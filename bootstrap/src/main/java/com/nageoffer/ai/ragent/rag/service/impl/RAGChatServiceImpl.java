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

package com.nageoffer.ai.ragent.rag.service.impl;

import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.framework.context.UserContext;
import com.nageoffer.ai.ragent.framework.trace.RagTraceContext;
import com.nageoffer.ai.ragent.infra.chat.StreamCallback;
import com.nageoffer.ai.ragent.rag.aop.ChatRateLimit;
import com.nageoffer.ai.ragent.rag.service.RAGChatService;
import com.nageoffer.ai.ragent.rag.service.handler.StreamCallbackFactory;
import com.nageoffer.ai.ragent.rag.service.handler.StreamTaskManager;
import com.nageoffer.ai.ragent.rag.service.pipeline.StreamChatContext;
import com.nageoffer.ai.ragent.rag.service.pipeline.StreamChatPipeline;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * RAG 对话服务的默认实现，是整个 RAG 对话管线的入口类。
 * <p>
 * 该类编排了完整的 RAG 对话流程，各阶段按顺序执行：
 * <ol>
 *   <li><b>记忆加载</b>：通过 {@link ConversationMemoryService} 加载对话历史（含摘要），并追加当前用户消息</li>
 *   <li><b>查询改写与拆分</b>：通过 {@link QueryRewriteService} 对用户问题进行术语归一化、指代消解和子问题拆分</li>
 *   <li><b>意图解析</b>：通过 {@link IntentResolver} 对各子问题并行进行意图识别（KB/MCP/SYSTEM 三种类型）</li>
 *   <li><b>歧义引导</b>：通过 {@link IntentGuidanceService} 检测是否存在意图歧义，若有则直接返回引导提示</li>
 *   <li><b>系统意图快捷路径</b>：若所有子问题均为 SYSTEM 类型意图，跳过检索阶段，直接走 LLM 对话</li>
 *   <li><b>多通道检索</b>：通过 {@link RetrievalEngine} 执行 MCP 工具调用和知识库向量检索</li>
 *   <li><b>Prompt 组装</b>：通过 {@link RAGPromptService} 根据检索结果和意图类型选择模板构建结构化消息</li>
 *   <li><b>流式输出</b>：通过 {@link LLMService} 以 SSE 方式流式生成回复</li>
 * </ol>
 * <p>
 * 流式任务通过 {@link StreamTaskManager} 管理生命周期，支持分布式取消。
 *
 * @see RAGChatService
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RAGChatServiceImpl implements RAGChatService {

    private final StreamChatPipeline chatPipeline;
    private final StreamCallbackFactory callbackFactory;
    private final StreamTaskManager taskManager;


    /**
     * RAG 对话管线入口方法，编排完整的流式对话流程。
     * <p>
     * 执行流程：
     * <ol>
     *   <li>初始化会话 ID 和任务 ID（若未提供则用雪花算法生成）</li>
     *   <li>创建 SSE 流式回调处理器</li>
     *   <li>加载对话历史（含摘要）并追加当前用户消息到持久化存储</li>
     *   <li>对用户问题进行改写和子问题拆分</li>
     *   <li>对拆分后的子问题并行进行意图识别</li>
     *   <li>歧义检测：若意图不明确，直接返回引导提示让用户澄清</li>
     *   <li>系统意图快捷路径：若所有意图均为 SYSTEM 类型（如闲聊），跳过检索直接调用 LLM</li>
     *   <li>多通道检索：执行 MCP 工具调用和/或知识库向量检索</li>
     *   <li>检索结果为空时直接返回兜底提示</li>
     *   <li>组装 Prompt 并流式调用 LLM 生成回复</li>
     * </ol>
     *
     * @param question       用户输入的原始问题文本
     * @param conversationId 会话 ID，为空时自动生成新会话
     * @param deepThinking   是否启用深度思考模式（开启后 LLM 会输出思考过程）
     * @param emitter        SSE 发射器，用于向前端推送流式响应事件
     */
    @Override
    @ChatRateLimit
    public void streamChat(String question, String conversationId, Boolean deepThinking, SseEmitter emitter) {
        // 若未传入会话 ID 或任务 ID，则使用雪花算法生成唯一标识
        String actualConversationId = StrUtil.isBlank(conversationId) ? IdUtil.getSnowflakeNextIdStr() : conversationId;
        String taskId = StrUtil.isBlank(RagTraceContext.getTaskId())
                ? IdUtil.getSnowflakeNextIdStr()
                : RagTraceContext.getTaskId();
        log.info("开始流式对话，会话ID：{}，任务ID：{}", actualConversationId, taskId);
        boolean thinkingEnabled = Boolean.TRUE.equals(deepThinking);

        // 创建 SSE 流式回调，负责将 LLM 输出事件推送到前端，并在完成时持久化助手回复
        StreamCallback callback = callbackFactory.createChatEventHandler(emitter, actualConversationId, taskId);

        StreamChatContext ctx = StreamChatContext.builder()
                .question(question)
                .conversationId(actualConversationId)
                .taskId(taskId)
                .deepThinking(thinkingEnabled)
                .userId(UserContext.getUserId())
                .callback(callback)
                .build();

        try {
            chatPipeline.execute(ctx);
        } catch (Exception e) {
            log.error("流式对话处理异常，会话ID：{}，任务ID：{}", actualConversationId, taskId, e);
            callback.onError(e);
        }
    }

    /**
     * 停止指定任务的流式输出。
     * <p>
     * 通过 {@link StreamTaskManager} 发布取消信号，底层使用 Redis Pub/Sub 实现分布式取消，
     * 即使 LLM 调用运行在其他实例上也能被正确取消。
     *
     * @param taskId 要取消的任务 ID
     */
    @Override
    public void stopTask(String taskId) {
        taskManager.cancel(taskId);
    }
}
