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

package com.nageoffer.ai.ragent.rag.service.handler;

import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.rag.dao.entity.ConversationDO;
import com.nageoffer.ai.ragent.rag.dto.CompletionPayload;
import com.nageoffer.ai.ragent.rag.dto.MessageDelta;
import com.nageoffer.ai.ragent.rag.dto.MetaPayload;
import com.nageoffer.ai.ragent.rag.enums.SSEEventType;
import com.nageoffer.ai.ragent.framework.context.UserContext;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.web.SseEmitterSender;
import com.nageoffer.ai.ragent.infra.chat.StreamCallback;
import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import com.nageoffer.ai.ragent.rag.core.memory.ConversationMemoryService;
import com.nageoffer.ai.ragent.rag.service.ConversationGroupService;

import java.util.Optional;

/**
 * 流式聊天事件处理器 —— RAG 管线中"LLM 流式生成"阶段的核心回调实现
 * <p>
 * 实现 {@link StreamCallback} 接口，作为 LLM 流式输出的下游消费者，
 * 承担从 LLM 到前端的"最后一公里"推送以及回复持久化职责。
 * <p>
 * 核心职责：
 * <ol>
 *     <li><b>SSE 推送</b>：将 LLM 输出的 token/片段通过 {@link SseEmitterSender}
 *         以 Server-Sent Events 协议实时推送给前端</li>
 *     <li><b>分块控制</b>：将模型输出按 {@code messageChunkSize}（可配置）分块推送，
 *         平衡网络效率和用户感知延迟</li>
 *     <li><b>回复持久化</b>：{@link #onComplete()} 时将累积的完整回复通过
 *         {@link ConversationMemoryService} 持久化到数据库</li>
 *     <li><b>标题生成</b>：对于新会话（无标题），在完成时查询或生成会话标题并推送给前端</li>
 *     <li><b>取消感知</b>：与 {@link StreamTaskManager} 配合，每次回调前检查任务是否已被取消，
 *         支持分布式取消场景下的优雅中止</li>
 * </ol>
 * <p>
 * 生命周期：每次流式请求创建一个新实例，请求结束后由 GC 回收。
 * 非 Spring 管理 Bean，通过 {@link StreamCallbackFactory} 创建。
 *
 * @see StreamCallbackFactory 工厂类，负责创建本处理器实例
 * @see StreamTaskManager     流式任务管理器，提供取消感知能力
 */
public class StreamChatEventHandler implements StreamCallback {

    /** SSE 事件中标识思考内容的类型标记 */
    private static final String TYPE_THINK = "think";
    /** SSE 事件中标识正式回复内容的类型标记 */
    private static final String TYPE_RESPONSE = "response";

    /** 每个 SSE 消息事件包含的最大字符数（按 Unicode code point 计算） */
    private final int messageChunkSize;
    /** SSE 发射器封装，保证连接只关闭一次的原子性 */
    private final SseEmitterSender sender;
    /** 当前会话 ID */
    private final String conversationId;
    /** 会话记忆服务，用于持久化消息 */
    private final ConversationMemoryService memoryService;
    /** 会话组服务，用于查询/更新会话元信息 */
    private final ConversationGroupService conversationGroupService;
    /** 当前流式任务 ID */
    private final String taskId;
    /** 当前用户 ID（从 UserContext 获取，支持跨线程传播） */
    private final String userId;
    /** 流式任务管理器 */
    private final StreamTaskManager taskManager;
    /** 是否需要在完成时推送会话标题（仅新会话或无标题会话需要） */
    private final boolean sendTitleOnComplete;
    /** 累积 LLM 输出的完整回复内容，用于最终持久化 */
    private final StringBuilder answer = new StringBuilder();
    private final StringBuilder thinking = new StringBuilder();
    private long thinkingStartMs;
    private int thinkingDurationSeconds;

    /**
     * 使用参数对象构造（推荐）
     * <p>
     * 构造过程：
     * <ol>
     *     <li>从参数对象提取依赖并初始化字段</li>
     *     <li>从 {@link UserContext} 获取当前用户 ID（需在请求线程中调用，
     *         或通过 TransmittableThreadLocal 跨线程传播）</li>
     *     <li>计算消息分块大小（从配置读取，最小为 1）</li>
     *     <li>判断是否需要在完成时发送标题（新会话或无标题会话）</li>
     *     <li>调用 {@link #initialize()} 发送初始化元数据并注册任务</li>
     * </ol>
     *
     * @param params 构建参数对象，包含所有必需的依赖
     */
    public StreamChatEventHandler(StreamChatHandlerParams params) {
        this.sender = new SseEmitterSender(params.getEmitter());
        this.conversationId = params.getConversationId();
        this.taskId = params.getTaskId();
        this.memoryService = params.getMemoryService();
        this.conversationGroupService = params.getConversationGroupService();
        this.taskManager = params.getTaskManager();
        this.userId = UserContext.getUserId();

        // 计算配置
        this.messageChunkSize = resolveMessageChunkSize(params.getModelProperties());
        this.sendTitleOnComplete = shouldSendTitle();

        // 初始化（发送初始事件、注册任务）
        initialize();
    }

    /**
     * 初始化：发送元数据事件并注册流式任务
     * <p>
     * 发送 META 事件让前端获知 conversationId 和 taskId，
     * 同时向 {@link StreamTaskManager} 注册任务，
     * 以支持后续的取消操作和取消时的内容保存回调。
     */
    private void initialize() {
        sender.sendEvent(SSEEventType.META.value(), new MetaPayload(conversationId, taskId));
        taskManager.register(taskId, sender, this::buildCompletionPayloadOnCancel);
    }

    /**
     * 解析消息分块大小
     * <p>
     * 从 AI 模型配置中读取 stream.messageChunkSize 值，最小为 1。
     * 分块大小决定每个 SSE 消息事件包含多少个字符，
     * 较小的值提供更流畅的打字机效果，较大的值减少网络开销。
     *
     * @param modelProperties AI 模型配置属性
     * @return 分块大小（>=1）
     */
    private int resolveMessageChunkSize(AIModelProperties modelProperties) {
        return Math.max(1, Optional.ofNullable(modelProperties.getStream())
                .map(AIModelProperties.Stream::getMessageChunkSize)
                .orElse(5));
    }

    /**
     * 判断是否需要在流式完成时推送会话标题
     * <p>
     * 仅当会话不存在（首次对话）或会话尚无标题时返回 true。
     * 这样可以在用户第一次发起对话后，自动为其生成会话标题。
     *
     * @return true 表示需要在 onComplete 时推送标题
     */
    private boolean shouldSendTitle() {
        ConversationDO existingConversation = conversationGroupService.findConversation(
                conversationId,
                userId
        );
        return existingConversation == null || StrUtil.isBlank(existingConversation.getTitle());
    }

    /**
     * 构造取消时的完成载荷
     * <p>
     * 当流式任务被取消时（如用户主动中止），此方法作为回调被 {@link StreamTaskManager} 调用。
     * 若已有部分回复内容，先将其持久化到数据库（避免内容丢失），再构建完成载荷。
     *
     * @return 包含消息 ID 和会话标题的完成载荷
     */
    private CompletionPayload buildCompletionPayloadOnCancel() {
        String content = answer.toString();
        String messageId = null;
        // 即使被取消，若已累积部分回复，也要持久化，避免用户看到的内容与数据库不一致
        if (StrUtil.isNotBlank(content)) {
            String thinkingContent = thinking.isEmpty() ? null : thinking.toString();
            ChatMessage message = ChatMessage.assistant(content, thinkingContent, resolveThinkingDuration());
            messageId = memoryService.append(conversationId, userId, message);
        }
        String title = resolveTitleForEvent();
        return new CompletionPayload(String.valueOf(messageId), title);
    }

    /**
     * 接收 LLM 输出的正式回复内容片段
     * <p>
     * 每次回调将内容累积到 {@code answer} 中（用于最终持久化），
     * 同时按分块大小通过 SSE 推送给前端。
     * 若任务已被取消或内容为空，则静默忽略。
     *
     * @param chunk LLM 输出的文本片段
     */
    @Override
    public void onContent(String chunk) {
        // 取消检查：避免在任务已取消后继续推送内容
        if (taskManager.isCancelled(taskId)) {
            return;
        }
        if (StrUtil.isBlank(chunk)) {
            return;
        }
        if (thinkingStartMs > 0 && thinkingDurationSeconds == 0) {
            thinkingDurationSeconds = Math.max(1, Math.round((System.currentTimeMillis() - thinkingStartMs) / 1000.0f));
        }
        // 累积完整回复用于后续持久化
        answer.append(chunk);
        // 按配置的分块大小推送 SSE 消息
        sendChunked(TYPE_RESPONSE, chunk);
    }

    /**
     * 接收 LLM 输出的思考过程内容片段
     * <p>
     * 思考内容仅推送给前端展示（如"思考中..."效果），不累积到 answer 中，
     * 不参与最终回复的持久化。
     *
     * @param chunk LLM 输出的思考文本片段
     */
    @Override
    public void onThinking(String chunk) {
        if (taskManager.isCancelled(taskId)) {
            return;
        }
        if (StrUtil.isBlank(chunk)) {
            return;
        }
        if (thinkingStartMs == 0) {
            thinkingStartMs = System.currentTimeMillis();
        }
        thinking.append(chunk);
        sendChunked(TYPE_THINK, chunk);
    }

    /**
     * LLM 流式输出正常完成时的处理
     * <p>
     * 完成时执行以下操作：
     * <ol>
     *     <li>将累积的完整回复持久化到数据库（通过 memoryService.append）</li>
     *     <li>解析会话标题（新会话时生成默认标题）</li>
     *     <li>发送 FINISH 事件，携带消息 ID 和标题</li>
     *     <li>发送 DONE 事件，标识 SSE 流结束</li>
     *     <li>从任务管理器中注销当前任务</li>
     *     <li>关闭 SSE 连接</li>
     * </ol>
     */
    @Override
    public void onComplete() {
        if (taskManager.isCancelled(taskId)) {
            return;
        }
        String thinkingContent = thinking.isEmpty() ? null : thinking.toString();
        ChatMessage message = ChatMessage.assistant(answer.toString(), thinkingContent, resolveThinkingDuration());
        String messageId = memoryService.append(conversationId, UserContext.getUserId(), message);
        String title = resolveTitleForEvent();
        String messageIdText = StrUtil.isBlank(messageId) ? null : messageId;
        // 发送完成事件和结束标记，然后注销任务并关闭 SSE 连接
        sender.sendEvent(SSEEventType.FINISH.value(), new CompletionPayload(messageIdText, title));
        sender.sendEvent(SSEEventType.DONE.value(), "[DONE]");
        taskManager.unregister(taskId);
        sender.complete();
    }

    /**
     * LLM 流式输出发生错误时的处理
     * <p>
     * 错误处理流程：先从任务管理器注销任务，再通过 SSE 发射器将错误通知前端。
     * 若任务已被取消，则忽略错误（取消本身会触发单独的完成流程）。
     *
     * @param t 发生的异常
     */
    @Override
    public void onError(Throwable t) {
        if (taskManager.isCancelled(taskId)) {
            return;
        }
        taskManager.unregister(taskId);
        sender.fail(t);
    }

    /**
     * 按配置的分块大小将文本内容拆分为多个 SSE 消息事件推送
     * <p>
     * 使用 {@link String#codePointAt} 按 Unicode code point 逐字遍历，
     * 正确处理 emoji 和 CJK 扩展字符等多字节字符（surrogate pair）。
     * 每积累 {@code messageChunkSize} 个 code point 后发送一个 SSE 消息事件。
     *
     * @param type    消息类型标记（"think" 或 "response"）
     * @param content 待推送的文本内容
     */
    private void sendChunked(String type, String content) {
        int length = content.length();
        int idx = 0;
        int count = 0;
        StringBuilder buffer = new StringBuilder();
        while (idx < length) {
            int codePoint = content.codePointAt(idx);
            buffer.appendCodePoint(codePoint);
            idx += Character.charCount(codePoint);
            count++;
            if (count >= messageChunkSize) {
                sender.sendEvent(SSEEventType.MESSAGE.value(), new MessageDelta(type, buffer.toString()));
                buffer.setLength(0);
                count = 0;
            }
        }
        if (!buffer.isEmpty()) {
            sender.sendEvent(SSEEventType.MESSAGE.value(), new MessageDelta(type, buffer.toString()));
        }
    }

    private Integer resolveThinkingDuration() {
        return thinkingDurationSeconds > 0 ? thinkingDurationSeconds : null;
    }

    /**
     * 解析当前会话的标题，用于在完成或取消时推送给前端
     * <p>
     * 如果不需要发送标题（已有标题的旧会话），直接返回 null。
     * 否则尝试从数据库查询最新标题；若仍无标题，返回默认值"新对话"。
     *
     * @return 会话标题字符串，或 null（表示不需要推送标题）
     */
    private String resolveTitleForEvent() {
        if (!sendTitleOnComplete) {
            return null;
        }
        ConversationDO conversation = conversationGroupService.findConversation(conversationId, userId);
        if (conversation != null && StrUtil.isNotBlank(conversation.getTitle())) {
            return conversation.getTitle();
        }
        return "新对话";
    }
}
