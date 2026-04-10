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

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.framework.context.UserContext;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.framework.trace.RagTraceContext;
import com.nageoffer.ai.ragent.infra.chat.LLMService;
import com.nageoffer.ai.ragent.infra.chat.StreamCallback;
import com.nageoffer.ai.ragent.infra.chat.StreamCancellationHandle;
import com.nageoffer.ai.ragent.rag.aop.ChatRateLimit;
import com.nageoffer.ai.ragent.rag.core.guidance.GuidanceDecision;
import com.nageoffer.ai.ragent.rag.core.guidance.IntentGuidanceService;
import com.nageoffer.ai.ragent.rag.core.intent.IntentResolver;
import com.nageoffer.ai.ragent.rag.core.memory.ConversationMemoryService;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptContext;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptTemplateLoader;
import com.nageoffer.ai.ragent.rag.core.prompt.RAGPromptService;
import com.nageoffer.ai.ragent.rag.core.retrieve.RetrievalEngine;
import com.nageoffer.ai.ragent.rag.core.rewrite.QueryRewriteService;
import com.nageoffer.ai.ragent.rag.core.rewrite.RewriteResult;
import com.nageoffer.ai.ragent.rag.dto.IntentGroup;
import com.nageoffer.ai.ragent.rag.dto.RetrievalContext;
import com.nageoffer.ai.ragent.rag.dto.SubQuestionIntent;
import com.nageoffer.ai.ragent.rag.service.RAGChatService;
import com.nageoffer.ai.ragent.rag.service.handler.StreamCallbackFactory;
import com.nageoffer.ai.ragent.rag.service.handler.StreamTaskManager;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.ArrayList;
import java.util.List;

import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.CHAT_SYSTEM_PROMPT_PATH;
import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.DEFAULT_TOP_K;

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

    /** LLM 服务，用于同步/流式调用大语言模型 */
    private final LLMService llmService;

    /** RAG Prompt 构建服务，根据意图类型和检索结果选择模板并组装消息列表 */
    private final RAGPromptService promptBuilder;

    /** Prompt 模板加载器，从文件系统加载 StringTemplate 模板 */
    private final PromptTemplateLoader promptTemplateLoader;

    /** 对话记忆服务，负责加载历史消息（含摘要）和追加新消息 */
    private final ConversationMemoryService memoryService;

    /** 流式任务管理器，绑定任务 ID 与取消句柄，支持 Redis Pub/Sub 分布式取消 */
    private final StreamTaskManager taskManager;

    /** 意图歧义引导服务，检测意图是否模糊并生成引导提示 */
    private final IntentGuidanceService guidanceService;

    /** 流式回调工厂，创建 SSE 事件处理器并负责消息持久化 */
    private final StreamCallbackFactory callbackFactory;

    /** 查询改写服务，执行术语归一化、LLM 改写和子问题拆分 */
    private final QueryRewriteService queryRewriteService;

    /** 意图解析器，对改写后的子问题进行并行意图识别 */
    private final IntentResolver intentResolver;

    /** 多通道检索引擎，编排 MCP 工具调用和知识库向量检索 */
    private final RetrievalEngine retrievalEngine;

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

        // 阶段1：记忆加载 —— 加载历史消息（含摘要），同时将当前用户消息追加到存储
        String userId = UserContext.getUserId();
        List<ChatMessage> history = memoryService.loadAndAppend(actualConversationId, userId, ChatMessage.user(question));

        // 阶段2：查询改写与拆分 —— 术语归一化 + LLM 改写（指代消解）+ 子问题拆分
        RewriteResult rewriteResult = queryRewriteService.rewriteWithSplit(question, history);
        // 阶段3：意图解析 —— 对每个子问题并行识别意图（KB/MCP/SYSTEM）
        List<SubQuestionIntent> subIntents = intentResolver.resolve(rewriteResult);

        // 阶段4：歧义检测 —— 若意图不够明确，返回引导性提示让用户进一步澄清
        GuidanceDecision guidanceDecision = guidanceService.detectAmbiguity(rewriteResult.rewrittenQuestion(), subIntents);
        if (guidanceDecision.isPrompt()) {
            callback.onContent(guidanceDecision.getPrompt());
            callback.onComplete();
            return;
        }

        // 阶段5：系统意图快捷路径 —— 若所有子问题的意图均为 SYSTEM 类型（如闲聊/打招呼），
        // 则无需检索知识库或调用 MCP 工具，直接走纯 LLM 对话即可
        boolean allSystemOnly = subIntents.stream()
                .allMatch(si -> intentResolver.isSystemOnly(si.nodeScores()));
        if (allSystemOnly) {
            // 优先使用意图节点上配置的自定义 Prompt 模板，若无则使用默认系统 Prompt
            String customPrompt = subIntents.stream()
                    .flatMap(si -> si.nodeScores().stream())
                    .map(ns -> ns.getNode().getPromptTemplate())
                    .filter(StrUtil::isNotBlank)
                    .findFirst()
                    .orElse(null);
            StreamCancellationHandle handle = streamSystemResponse(rewriteResult.rewrittenQuestion(), history, customPrompt, callback);
            taskManager.bindHandle(taskId, handle);
            return;
        }

        // 阶段6：多通道检索 —— 根据意图类型执行 MCP 工具调用和/或知识库向量检索
        RetrievalContext ctx = retrievalEngine.retrieve(subIntents, DEFAULT_TOP_K);
        if (ctx.isEmpty()) {
            // 检索结果为空时返回兜底提示，避免将空上下文传给 LLM 导致幻觉
            String emptyReply = "未检索到与问题相关的文档内容。";
            callback.onContent(emptyReply);
            callback.onComplete();
            return;
        }

        // 阶段7：聚合所有子问题的意图分组（KB 意图 + MCP 意图），用于 Prompt 模板选择
        IntentGroup mergedGroup = intentResolver.mergeIntentGroup(subIntents);

        // 阶段8：组装 Prompt 并流式调用 LLM 生成最终回复
        StreamCancellationHandle handle = streamLLMResponse(
                rewriteResult,
                ctx,
                mergedGroup,
                history,
                thinkingEnabled,
                callback
        );
        // 绑定取消句柄到任务管理器，支持通过 Redis Pub/Sub 进行分布式取消
        taskManager.bindHandle(taskId, handle);
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

    // ==================== LLM 响应 ====================

    /**
     * 系统意图响应 —— 不经过检索阶段，直接使用 LLM 进行对话。
     * <p>
     * 适用于所有子问题意图均为 SYSTEM 类型的场景（如闲聊、打招呼等），
     * 此时无需检索知识库或调用 MCP 工具，直接构建对话消息列表并流式调用 LLM。
     * <p>
     * 与 {@link #streamLLMResponse} 的核心区别：
     * <ul>
     *   <li>不携带检索上下文（无 KB/MCP 上下文注入）</li>
     *   <li>使用通用系统 Prompt 或意图节点自定义 Prompt</li>
     *   <li>温度固定为 0.7（更自然的对话风格），不启用深度思考</li>
     * </ul>
     *
     * @param question    改写后的用户问题
     * @param history     对话历史消息列表（含摘要）
     * @param customPrompt 意图节点上配置的自定义 Prompt 模板，为 null 时使用默认系统 Prompt
     * @param callback    流式回调处理器
     * @return 流式任务取消句柄
     */

    private StreamCancellationHandle streamSystemResponse(String question, List<ChatMessage> history,
                                                          String customPrompt, StreamCallback callback) {
        // 优先使用意图节点自定义 Prompt，否则加载默认的系统对话 Prompt 模板
        String systemPrompt = StrUtil.isNotBlank(customPrompt)
                ? customPrompt
                : promptTemplateLoader.load(CHAT_SYSTEM_PROMPT_PATH);

        // 构建消息列表：系统提示 + 历史消息（排除最后一条，因为最后一条是刚追加的用户消息） + 改写后的用户问题
        List<ChatMessage> messages = new ArrayList<>();
        messages.add(ChatMessage.system(systemPrompt));
        if (CollUtil.isNotEmpty(history)) {
            // 排除 history 中的最后一条消息，因为改写后的问题将替代它作为最新的用户输入
            messages.addAll(history.subList(0, history.size() - 1));
        }
        messages.add(ChatMessage.user(question));

        ChatRequest req = ChatRequest.builder()
                .messages(messages)
                .temperature(0.7D)   // 系统意图场景使用较高温度，回复更自然
                .thinking(false)     // 系统意图无需深度思考
                .build();
        return llmService.streamChat(req, callback);
    }

    /**
     * RAG 增强响应 —— 携带检索上下文调用 LLM 生成回复。
     * <p>
     * 适用于包含 KB（知识库）和/或 MCP（工具调用）意图的场景。
     * 通过 {@link RAGPromptService} 根据意图类型自动选择合适的 Prompt 模板
     * （KB_ONLY / MCP_ONLY / MIXED），并将检索到的文档片段和工具调用结果注入上下文。
     * <p>
     * 与 {@link #streamSystemResponse} 的核心区别：
     * <ul>
     *   <li>携带完整的检索上下文（KB 文档片段 + MCP 工具调用结果）</li>
     *   <li>使用 RAG 专用 Prompt 模板，而非通用系统 Prompt</li>
     *   <li>温度根据是否有 MCP 结果动态调整（MCP: 0.3, 纯KB: 0）</li>
     *   <li>支持深度思考模式</li>
     * </ul>
     *
     * @param rewriteResult 查询改写结果，包含改写后的问题和子问题列表
     * @param ctx           检索上下文，包含 KB 文档片段和 MCP 工具调用结果
     * @param intentGroup   聚合后的意图分组，包含 KB 意图列表和 MCP 意图列表
     * @param history       对话历史消息列表（含摘要）
     * @param deepThinking  是否启用深度思考模式
     * @param callback      流式回调处理器
     * @return 流式任务取消句柄
     */

    private StreamCancellationHandle streamLLMResponse(RewriteResult rewriteResult, RetrievalContext ctx,
                                                       IntentGroup intentGroup, List<ChatMessage> history,
                                                       boolean deepThinking, StreamCallback callback) {
        // 构建 Prompt 上下文对象，汇聚改写后的问题、检索结果和意图信息
        PromptContext promptContext = PromptContext.builder()
                .question(rewriteResult.rewrittenQuestion())
                .mcpContext(ctx.getMcpContext())         // MCP 工具调用结果
                .kbContext(ctx.getKbContext())           // 知识库检索到的文档片段
                .mcpIntents(intentGroup.mcpIntents())    // MCP 类型意图列表
                .kbIntents(intentGroup.kbIntents())      // KB 类型意图列表
                .intentChunks(ctx.getIntentChunks())     // 各意图对应的检索分块
                .build();

        // 通过 RAGPromptService 构建结构化消息列表（自动选择 KB_ONLY/MCP_ONLY/MIXED 模板）
        List<ChatMessage> messages = promptBuilder.buildStructuredMessages(
                promptContext,
                history,
                rewriteResult.rewrittenQuestion(),
                rewriteResult.subQuestions()  // 传入子问题列表
        );
        ChatRequest chatRequest = ChatRequest.builder()
                .messages(messages)
                .thinking(deepThinking)
                .temperature(ctx.hasMcp() ? 0.3D : 0D)  // MCP 场景稍微放宽温度，允许更灵活的工具结果整合；纯 KB 场景温度为 0，确保严格基于文档回答
                .topP(ctx.hasMcp() ? 0.8D : 1D)         // 与温度配合调整采样策略
                .build();

        return llmService.streamChat(chatRequest, callback);
    }
}
