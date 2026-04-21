/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nageoffer.ai.ragent.infra.chat;

import cn.hutool.core.collection.CollUtil;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.framework.errorcode.BaseErrorCode;
import com.nageoffer.ai.ragent.framework.exception.RemoteException;
import com.nageoffer.ai.ragent.framework.trace.RagTraceNode;
import com.nageoffer.ai.ragent.infra.enums.ModelCapability;
import com.nageoffer.ai.ragent.infra.model.ModelHealthStore;
import com.nageoffer.ai.ragent.infra.model.ModelRoutingExecutor;
import com.nageoffer.ai.ragent.infra.model.ModelSelector;
import com.nageoffer.ai.ragent.infra.model.ModelTarget;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 路由式 LLM 服务实现类 —— infra-ai 层的核心组件，{@link LLMService} 的 @Primary 实现
 * <p>
 * 该服务在 LLM 接口与具体模型提供商之间增加了一层智能路由和容错调度，
 * 是整个 RAG 管线中"LLM 流式生成"阶段的入口。
 * <p>
 * 核心能力：
 * <ol>
 *     <li><b>优先级调度</b>：通过 {@link ModelSelector} 按优先级排序候选模型列表，
 *         支持首选模型置顶、不可用模型过滤</li>
 *     <li><b>断路器/熔断</b>：通过 {@link ModelHealthStore} 维护每个模型的健康状态
 *         （CLOSED → OPEN → HALF_OPEN 三态），连续失败自动熔断</li>
 *     <li><b>流式首包探测</b>：流式调用时使用 {@link ProbeBufferingCallback} 缓冲探测阶段的事件，
 *         配合 {@link FirstPacketAwaiter} 等待首包到达。若首包超时或出错，透明切换下一个候选模型</li>
 *     <li><b>透明降级</b>：所有候选模型逐一尝试，失败自动切换下一个，调用方无感知</li>
 * </ol>
 * <p>
 * 同步调用通过 {@link ModelRoutingExecutor#executeWithFallback} 实现带降级的统一调度；
 * 流式调用则在本类中手动实现候选遍历 + 首包探测 + 降级逻辑。
 */
@Slf4j
@Service
@Primary
public class RoutingLLMService implements LLMService {

    /** 流式首包等待超时时间（秒），超时即认为该模型不可用 */
    private static final int FIRST_PACKET_TIMEOUT_SECONDS = 60;
    /** 各种流式错误的提示消息常量 */
    private static final String STREAM_INTERRUPTED_MESSAGE = "流式请求被中断";
    private static final String STREAM_NO_PROVIDER_MESSAGE = "无可用大模型提供者";
    private static final String STREAM_START_FAILED_MESSAGE = "流式请求启动失败";
    private static final String STREAM_TIMEOUT_MESSAGE = "流式首包超时";
    private static final String STREAM_NO_CONTENT_MESSAGE = "流式请求未返回内容";
    private static final String STREAM_ALL_FAILED_MESSAGE = "大模型调用失败，请稍后再试...";

    /** 模型优先级选择器，负责筛选和排序候选模型 */
    private final ModelSelector selector;
    /** 模型健康状态存储，维护断路器状态（成功/失败标记） */
    private final ModelHealthStore healthStore;
    /** 路由执行器，封装同步调用的降级循环逻辑 */
    private final ModelRoutingExecutor executor;
    /** 按提供商名称索引的 ChatClient 映射表，key 为 provider 标识 */
    private final Map<String, ChatClient> clientsByProvider;

    /**
     * 构造方法：注入依赖并构建按提供商名称索引的 ChatClient 映射表
     * <p>
     * Spring 会自动注入所有 {@link ChatClient} 实现（如 OllamaChatClient、BailianChatClient 等），
     * 通过 {@link ChatClient#provider()} 作为 key 建立快速查找表。
     *
     * @param selector    模型选择器
     * @param healthStore 模型健康状态存储
     * @param executor    路由执行器
     * @param clients     所有已注册的 ChatClient 实现列表
     */
    public RoutingLLMService(
            ModelSelector selector,
            ModelHealthStore healthStore,
            ModelRoutingExecutor executor,
            List<ChatClient> clients) {
        this.selector = selector;
        this.healthStore = healthStore;
        this.executor = executor;
        this.clientsByProvider = clients.stream()
                .collect(Collectors.toMap(ChatClient::provider, Function.identity()));
    }

    /**
     * 同步调用（带降级）：委托 {@link ModelRoutingExecutor#executeWithFallback} 实现
     * <p>
     * 执行流程：
     * <ol>
     *     <li>通过 {@link ModelSelector#selectChatCandidates} 获取优先级排序后的候选模型列表</li>
     *     <li>逐一尝试调用，成功则返回结果，失败则自动切换下一个候选</li>
     *     <li>所有候选均失败时抛出 {@link RemoteException}</li>
     * </ol>
     *
     * @param request 完整的 Chat 请求对象
     * @return 模型返回的完整回答文本
     */
    @Override
    @RagTraceNode(name = "llm-chat-routing", type = "LLM_ROUTING")
    public String chat(ChatRequest request) {
        return executor.executeWithFallback(
                ModelCapability.CHAT,
                selector.selectChatCandidates(Boolean.TRUE.equals(request.getThinking())),
                target -> clientsByProvider.get(target.candidate().getProvider()),
                (client, target) -> client.chat(request, target)
        );
    }

    @Override
    public String chat(ChatRequest request, String modelId) {
        if (!StringUtils.hasText(modelId)) {
            return chat(request);
        }
        return executor.executeWithFallback(
                ModelCapability.CHAT,
                List.of(resolveTarget(modelId, Boolean.TRUE.equals(request.getThinking()))),
                target -> clientsByProvider.get(target.candidate().getProvider()),
                (client, target) -> client.chat(request, target)
        );
    }

    /**
     * 流式调用（带首包探测和透明降级）
     * <p>
     * 核心流程：
     * <ol>
     *     <li>获取候选模型列表，若为空直接抛异常</li>
     *     <li>按优先级顺序逐一尝试每个候选模型：
     *         <ul>
     *             <li>解析对应的 {@link ChatClient}，若缺失则跳过</li>
     *             <li>创建 {@link FirstPacketAwaiter}（基于 CountDownLatch 的首包等待器）
     *                 和 {@link ProbeBufferingCallback}（探测阶段缓冲事件的装饰器回调）</li>
     *             <li>调用 client.streamChat() 启动流式请求</li>
     *             <li>阻塞等待首包到达（最多 {@value FIRST_PACKET_TIMEOUT_SECONDS} 秒）</li>
     *             <li>首包成功 → commit 回放缓冲事件 → 标记健康 → 返回取消句柄</li>
     *             <li>首包失败（超时/错误/无内容） → 标记不健康 → 取消当前请求 → 切换下一个候选</li>
     *         </ul>
     *     </li>
     *     <li>所有候选均失败 → 通过回调通知错误 → 抛出 {@link RemoteException}</li>
     * </ol>
     *
     * @param request  完整的 Chat 请求对象
     * @param callback 下游流式回调处理器（如 SSE 推送处理器）
     * @return 取消句柄，调用方可通过 handle.cancel() 中断生成
     * @throws RemoteException 当所有候选模型均失败时抛出
     */
    @Override
    @RagTraceNode(name = "llm-stream-routing", type = "LLM_ROUTING")
    public StreamCancellationHandle streamChat(ChatRequest request, StreamCallback callback) {
        // 获取按优先级排序的候选模型列表
        List<ModelTarget> targets = selector.selectChatCandidates(Boolean.TRUE.equals(request.getThinking()));
        if (CollUtil.isEmpty(targets)) {
            throw new RemoteException(STREAM_NO_PROVIDER_MESSAGE);
        }

        String label = ModelCapability.CHAT.getDisplayName();
        Throwable lastError = null;

        // 按优先级逐一尝试候选模型
        for (ModelTarget target : targets) {
            // 解析该候选对应的 ChatClient 实现
            ChatClient client = resolveClient(target, label);
            if (client == null) {
                continue;
            }
            if (!healthStore.allowCall(target.id())) {
                continue;
            }

            ProbeStreamBridge bridge = new ProbeStreamBridge(callback);

            // 启动流式请求
            StreamCancellationHandle handle;
            try {
                handle = client.streamChat(request, bridge, target);
            } catch (Exception e) {
                // 启动失败：标记不健康，记录错误，尝试下一个
                healthStore.markFailure(target.id());
                lastError = e;
                log.warn("{} 流式请求启动失败，切换下一个模型。modelId：{}，provider：{}",
                        label, target.id(), target.candidate().getProvider(), e);
                continue;
            }
            if (handle == null) {
                // 返回空句柄视为启动失败
                healthStore.markFailure(target.id());
                lastError = new RemoteException(STREAM_START_FAILED_MESSAGE, BaseErrorCode.REMOTE_ERROR);
                log.warn("{} 流式请求未返回取消句柄，切换下一个模型。modelId：{}，provider：{}",
                        label, target.id(), target.candidate().getProvider());
                continue;
            }

            ProbeStreamBridge.ProbeResult result = awaitFirstPacket(bridge, handle, callback);


            if (result.isSuccess()) {
                healthStore.markSuccess(target.id());
                return handle;
            }

            // 首包探测失败：标记不健康，取消当前请求，尝试下一个候选
            healthStore.markFailure(target.id());
            handle.cancel();

            lastError = buildLastErrorAndLog(result, target, label);
        }

        // 所有候选模型均失败：通知回调层错误并抛出异常
        throw notifyAllFailed(callback, lastError);
    }

    /**
     * 根据候选模型的 provider 标识查找对应的 {@link ChatClient} 实现
     *
     * @param target 候选模型目标
     * @param label  能力标签（用于日志输出）
     * @return 对应的 ChatClient；若未找到返回 null 并记录警告日志
     */
    private ChatClient resolveClient(ModelTarget target, String label) {
        ChatClient client = clientsByProvider.get(target.candidate().getProvider());
        if (client == null) {
            log.warn("{} 提供商客户端缺失: provider：{}，modelId：{}",
                    label, target.candidate().getProvider(), target.id());
        }
        return client;
    }

    private ProbeStreamBridge.ProbeResult awaitFirstPacket(ProbeStreamBridge bridge,
                                                           StreamCancellationHandle handle,
                                                           StreamCallback callback) {
        try {
            return bridge.awaitFirstPacket(FIRST_PACKET_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            handle.cancel();
            RemoteException interruptedException = new RemoteException(STREAM_INTERRUPTED_MESSAGE, e, BaseErrorCode.REMOTE_ERROR);
            callback.onError(interruptedException);
            throw interruptedException;
        }
    }

    private Throwable buildLastErrorAndLog(ProbeStreamBridge.ProbeResult result, ModelTarget target, String label) {
        switch (result.getType()) {
            case ERROR -> {
                Throwable error = result.getError() != null
                        ? result.getError()
                        : new RemoteException("流式请求失败", BaseErrorCode.REMOTE_ERROR);
                log.warn("{} 失败模型: modelId={}, provider={}，原因: 流式请求失败，切换下一个模型",
                        label, target.id(), target.candidate().getProvider(), error);
                return error;
            }
            case TIMEOUT -> {
                RemoteException timeout = new RemoteException(STREAM_TIMEOUT_MESSAGE, BaseErrorCode.REMOTE_ERROR);
                log.warn("{} 失败模型: modelId={}, provider={}，原因: 流式请求超时，切换下一个模型",
                        label, target.id(), target.candidate().getProvider());
                return timeout;
            }
            case NO_CONTENT -> {
                RemoteException noContent = new RemoteException(STREAM_NO_CONTENT_MESSAGE, BaseErrorCode.REMOTE_ERROR);
                log.warn("{} 失败模型: modelId={}, provider={}，原因: 流式请求无内容完成，切换下一个模型",
                        label, target.id(), target.candidate().getProvider());
                return noContent;
            }
            default -> {
                RemoteException unknown = new RemoteException("流式请求失败", BaseErrorCode.REMOTE_ERROR);
                log.warn("{} 失败模型: modelId={}, provider={}，原因: 流式请求失败（未知类型），切换下一个模型",
                        label, target.id(), target.candidate().getProvider());
                return unknown;
            }
        }
    }

    /**
     * 所有候选模型均失败后的最终处理：通知回调层错误并构建最终异常
     * <p>
     * 将最后一次错误作为 cause 包装到最终异常中，同时通过回调通知下游（如 SSE 推送层）
     *
     * @param callback  下游回调
     * @param lastError 最后一次失败的错误
     * @return 构建的最终异常对象
     */
    private RemoteException notifyAllFailed(StreamCallback callback, Throwable lastError) {
        RemoteException finalException = new RemoteException(
                STREAM_ALL_FAILED_MESSAGE,
                lastError,
                BaseErrorCode.REMOTE_ERROR
        );
        callback.onError(finalException);
        return finalException;
    }

    private ModelTarget resolveTarget(String modelId, boolean deepThinking) {
        return selector.selectChatCandidates(deepThinking).stream()
                .filter(target -> modelId.equals(target.id()))
                .findFirst()
                .orElseThrow(() -> new RemoteException("Chat 模型不可用: " + modelId));
    }
}
