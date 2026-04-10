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

import java.util.ArrayList;
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
                selector.selectChatCandidates(request.getThinking()),
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
        List<ModelTarget> targets = selector.selectChatCandidates(request.getThinking());
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

            // 创建首包探测器和缓冲回调装饰器
            // awaiter: CountDownLatch 机制，等待首个有效内容到达
            // wrapper: 探测阶段缓冲所有事件，避免失败模型的输出污染下游
            FirstPacketAwaiter awaiter = new FirstPacketAwaiter();
            ProbeBufferingCallback wrapper = new ProbeBufferingCallback(callback, awaiter);

            // 启动流式请求
            StreamCancellationHandle handle;
            try {
                handle = client.streamChat(request, wrapper, target);
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

            // 阻塞等待首包（最多 FIRST_PACKET_TIMEOUT_SECONDS 秒）
            FirstPacketAwaiter.Result result = awaitFirstPacket(awaiter, handle, callback);

            // 首包探测成功：提交缓冲事件回放，标记模型健康
            if (result.isSuccess()) {
                wrapper.commit();
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

    /**
     * 阻塞等待流式首包到达
     * <p>
     * 使用 {@link FirstPacketAwaiter#await} 基于 CountDownLatch 机制等待，
     * 若等待期间线程被中断，则取消当前流式请求并通过回调通知错误。
     *
     * @param awaiter  首包等待器
     * @param handle   当前流式请求的取消句柄（中断时用于清理）
     * @param callback 下游回调（中断时用于通知错误）
     * @return 首包等待结果（成功/超时/错误/无内容）
     * @throws RemoteException 当等待线程被中断时抛出
     */
    private FirstPacketAwaiter.Result awaitFirstPacket(FirstPacketAwaiter awaiter,
                                                       StreamCancellationHandle handle,
                                                       StreamCallback callback) {
        try {
            return awaiter.await(FIRST_PACKET_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            handle.cancel();
            RemoteException interruptedException = new RemoteException(STREAM_INTERRUPTED_MESSAGE, e, BaseErrorCode.REMOTE_ERROR);
            callback.onError(interruptedException);
            throw interruptedException;
        }
    }

    /**
     * 根据首包探测结果类型构建对应的错误对象并记录日志
     * <p>
     * 四种结果类型对应不同的错误语义：
     * <ul>
     *     <li>ERROR — 流式请求过程中发生异常</li>
     *     <li>TIMEOUT — 首包在规定时间内未到达</li>
     *     <li>NO_CONTENT — 流式请求正常完成但未返回任何内容</li>
     *     <li>default — 未知类型，兜底处理</li>
     * </ul>
     *
     * @param result 首包等待结果
     * @param target 当前失败的候选模型
     * @param label  能力标签（用于日志）
     * @return 构建的错误对象，将作为 lastError 传递给后续逻辑
     */
    private Throwable buildLastErrorAndLog(FirstPacketAwaiter.Result result, ModelTarget target, String label) {
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

    /**
     * 流式首包探测缓冲回调 —— 装饰器模式
     * <p>
     * 设计目的：在流式请求的探测阶段，将所有回调事件（onContent/onThinking/onComplete/onError）
     * 缓存在内存列表中，避免失败模型的部分输出泄露到下游（如 SSE 推送给前端）。
     * <p>
     * 工作机制：
     * <ul>
     *     <li><b>探测阶段（committed=false）</b>：所有事件存入 {@code bufferedEvents}，
     *         同时通过 {@link FirstPacketAwaiter} 标记首包状态</li>
     *     <li><b>提交后（committed=true）</b>：
     *         <ol>
     *             <li>{@link #commit()} 方法原子切换 committed 标记</li>
     *             <li>按顺序回放缓冲区中的所有事件到下游回调</li>
     *             <li>后续新事件直接透传，不再缓冲</li>
     *         </ol>
     *     </li>
     * </ul>
     * <p>
     * 线程安全：通过 synchronized(lock) 保护 committed 标记与缓冲列表的原子性操作
     */
    private static final class ProbeBufferingCallback implements StreamCallback {

        /** 下游真实回调（通常是 StreamChatEventHandler） */
        private final StreamCallback downstream;
        /** 首包等待器，用于通知路由层首包已到达或发生错误 */
        private final FirstPacketAwaiter awaiter;
        /** 同步锁，保护 committed 标记和 bufferedEvents 的原子操作 */
        private final Object lock = new Object();
        /** 探测阶段缓冲的事件列表，commit 后清空 */
        private final List<BufferedEvent> bufferedEvents = new ArrayList<>();
        /** 是否已提交：false=探测阶段（缓冲），true=已确认（透传） */
        private volatile boolean committed;

        private ProbeBufferingCallback(StreamCallback downstream, FirstPacketAwaiter awaiter) {
            this.downstream = downstream;
            this.awaiter = awaiter;
            this.committed = false;
        }

        /** 收到文本内容片段：标记首包到达，缓冲或透传 */
        @Override
        public void onContent(String content) {
            awaiter.markContent();
            bufferOrDispatch(BufferedEvent.content(content));
        }

        /** 收到思考内容片段：同样视为有效首包 */
        @Override
        public void onThinking(String content) {
            awaiter.markContent();
            bufferOrDispatch(BufferedEvent.thinking(content));
        }

        /** 流式请求正常完成：标记完成状态 */
        @Override
        public void onComplete() {
            awaiter.markComplete();
            bufferOrDispatch(BufferedEvent.complete());
        }

        /** 流式请求出错：标记错误状态，触发路由层切换下一个候选 */
        @Override
        public void onError(Throwable t) {
            awaiter.markError(t);
            bufferOrDispatch(BufferedEvent.error(t));
        }

        /**
         * 首包探测成功后提交：
         * 1. 原子切换为 committed
         * 2. 按事件顺序回放缓存，保证时序一致
         */
        private void commit() {
            List<BufferedEvent> snapshot;
            synchronized (lock) {
                if (committed) {
                    return;
                }
                committed = true;
                if (bufferedEvents.isEmpty()) {
                    return;
                }
                snapshot = new ArrayList<>(bufferedEvents);
                bufferedEvents.clear();
            }
            for (BufferedEvent event : snapshot) {
                dispatch(event);
            }
        }

        /**
         * 根据当前状态决定缓冲还是直接分发事件
         * <p>
         * 在 synchronized 块内检查 committed 标记，保证"检查 + 缓冲"的原子性，
         * 避免 commit 和 bufferOrDispatch 之间的竞态条件导致事件丢失或重复
         *
         * @param event 待处理的缓冲事件
         */
        private void bufferOrDispatch(BufferedEvent event) {
            boolean dispatchNow;
            synchronized (lock) {
                dispatchNow = committed;
                if (!dispatchNow) {
                    bufferedEvents.add(event);
                }
            }
            if (dispatchNow) {
                dispatch(event);
            }
        }

        /**
         * 将缓冲事件按类型分发到下游真实回调
         *
         * @param event 缓冲事件
         */
        private void dispatch(BufferedEvent event) {
            switch (event.type()) {
                case CONTENT -> downstream.onContent(event.content());
                case THINKING -> downstream.onThinking(event.content());
                case COMPLETE -> downstream.onComplete();
                case ERROR -> downstream.onError(event.error() != null
                        ? event.error()
                        : new RemoteException("流式请求失败", BaseErrorCode.REMOTE_ERROR));
            }
        }

        /**
         * 缓冲事件记录 —— 不可变数据载体
         * <p>
         * 使用 record 类型保证不可变性，包含事件类型、文本内容和异常信息三个字段。
         * 通过静态工厂方法创建不同类型的事件实例。
         */
        private record BufferedEvent(EventType type, String content, Throwable error) {

            private static BufferedEvent content(String content) {
                return new BufferedEvent(EventType.CONTENT, content, null);
            }

            private static BufferedEvent thinking(String content) {
                return new BufferedEvent(EventType.THINKING, content, null);
            }

            private static BufferedEvent complete() {
                return new BufferedEvent(EventType.COMPLETE, null, null);
            }

            private static BufferedEvent error(Throwable error) {
                return new BufferedEvent(EventType.ERROR, null, error);
            }
        }

        /** 缓冲事件类型枚举 */
        private enum EventType {
            CONTENT,
            THINKING,
            COMPLETE,
            ERROR
        }
    }
}
