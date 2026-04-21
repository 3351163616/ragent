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

package com.nageoffer.ai.ragent.rag.core.retrieve;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.rag.dto.KbResult;
import com.nageoffer.ai.ragent.rag.dto.RetrievalContext;
import com.nageoffer.ai.ragent.rag.dto.SubQuestionIntent;
import com.nageoffer.ai.ragent.framework.convention.RetrievedChunk;
import com.nageoffer.ai.ragent.framework.trace.RagTraceNode;
import com.nageoffer.ai.ragent.rag.core.intent.IntentNode;
import com.nageoffer.ai.ragent.rag.core.intent.NodeScore;
import com.nageoffer.ai.ragent.rag.core.intent.NodeScoreFilters;
import com.nageoffer.ai.ragent.rag.core.mcp.MCPParameterExtractor;
import com.nageoffer.ai.ragent.rag.core.mcp.MCPRequest;
import com.nageoffer.ai.ragent.rag.core.mcp.MCPResponse;
import com.nageoffer.ai.ragent.rag.core.mcp.MCPTool;
import com.nageoffer.ai.ragent.rag.core.mcp.MCPToolExecutor;
import com.nageoffer.ai.ragent.rag.core.mcp.MCPToolRegistry;
import com.nageoffer.ai.ragent.rag.core.prompt.ContextFormatter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.util.HashMap;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.DEFAULT_TOP_K;
import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.MULTI_CHANNEL_KEY;

/**
 * 检索引擎 —— RAG 管线的核心检索协调层
 * <p>
 * 在 RAG 管线中，位于意图识别 / 查询改写之后、Prompt 构建 / LLM 生成之前，
 * 负责将已分类的子问题意图列表转化为结构化的检索上下文（{@link RetrievalContext}）。
 * <p>
 * 核心职责：
 * <ol>
 *   <li><b>并行处理子问题</b>：使用 {@code ragContextExecutor} 线程池对多个子问题并行构建上下文</li>
 *   <li><b>意图分流</b>：将每个子问题的意图按类型分为 KB（知识库检索）和 MCP（工具调用）两路</li>
 *   <li><b>KB 检索</b>：委托 {@link MultiChannelRetrievalEngine} 执行多通道向量检索 + 后处理</li>
 *   <li><b>MCP 执行</b>：使用 {@code mcpBatchExecutor} 线程池并行调用 MCP 工具，获取结构化结果</li>
 *   <li><b>结果整合</b>：将所有子问题的 KB 上下文和 MCP 上下文分别拼接，供下游 Prompt 构建使用</li>
 * </ol>
 *
 * @see MultiChannelRetrievalEngine 多通道检索引擎
 * @see RetrievalContext 检索上下文（最终输出）
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RetrievalEngine {

    /** 上下文格式化器，将检索结果格式化为 LLM 可读的文本 */
    private final ContextFormatter contextFormatter;

    /** MCP 参数提取器，从用户问题中提取工具调用所需参数 */
    private final MCPParameterExtractor mcpParameterExtractor;

    /** MCP 工具注册表，管理所有可用的 MCP 工具执行器 */
    private final MCPToolRegistry mcpToolRegistry;

    /** 多通道检索引擎，负责 KB 场景下的向量检索和后处理 */
    private final MultiChannelRetrievalEngine multiChannelRetrievalEngine;

    /** 子问题上下文构建线程池，用于并行处理多个子问题 */
    @Qualifier("ragContextThreadPoolExecutor")
    private final Executor ragContextExecutor;

    /** MCP 工具批量执行线程池，用于并行调用多个 MCP 工具 */
    @Qualifier("mcpBatchThreadPoolExecutor")
    private final Executor mcpBatchExecutor;

    /**
     * 检索入口方法：根据子问题意图列表执行检索，整合知识库和 MCP 工具的结果
     * <p>
     * 执行流程：
     * <ol>
     *   <li>为每个子问题提交异步任务（{@code ragContextExecutor}），并行构建各自的上下文</li>
     *   <li>等待所有子问题的异步任务完成（{@code join}）</li>
     *   <li>将各子问题的 KB 上下文、MCP 上下文分别拼接，intentChunks 合并</li>
     *   <li>返回统一的 {@link RetrievalContext} 供下游 Prompt 构建使用</li>
     * </ol>
     *
     * @param subIntents 子问题意图列表，每个元素包含一个子问题及其关联的意图节点和得分
     * @param topK       期望返回的最相关文档块数量，若 <= 0 则使用默认值 {@code DEFAULT_TOP_K}
     * @return {@link RetrievalContext} 检索上下文，包含 KB 上下文文本、MCP 上下文文本和按意图分组的检索块
     */
    @RagTraceNode(name = "retrieval-engine", type = "RETRIEVE")
    public RetrievalContext retrieve(List<SubQuestionIntent> subIntents, int topK) {
        if (CollUtil.isEmpty(subIntents)) {
            return RetrievalContext.builder()
                    .intentChunks(Map.of())
                    .build();
        }

        // 确定实际 topK：优先使用传入值，无效则回退到全局默认值
        int finalTopK = topK > 0 ? topK : DEFAULT_TOP_K;
        // 为每个子问题提交异步上下文构建任务，利用线程池并行加速
        List<CompletableFuture<SubQuestionContext>> tasks = subIntents.stream()
                .map(si -> CompletableFuture.supplyAsync(
                        () -> {
                            try {
                                return buildSubQuestionContext(
                                        si,
                                        resolveSubQuestionTopK(si, finalTopK)
                                );
                            } catch (Exception e) {
                                log.error("子问题上下文构建失败，降级为空上下文，question：{}", si.subQuestion(), e);
                                return new SubQuestionContext(si.subQuestion(), "", "", Map.of());
                            }
                        },
                        ragContextExecutor
                ))
                .toList();
        // 等待所有子问题的异步任务完成，收集结果
        List<SubQuestionContext> contexts = tasks.stream()
                .map(CompletableFuture::join)
                .toList();

        // 分别拼接 KB 上下文和 MCP 上下文，合并 intentChunks
        StringBuilder kbBuilder = new StringBuilder();
        StringBuilder mcpBuilder = new StringBuilder();
        Map<String, List<RetrievedChunk>> mergedIntentChunks = new HashMap<>();

        for (SubQuestionContext context : contexts) {
            if (StrUtil.isNotBlank(context.kbContext())) {
                appendSection(kbBuilder, context.question(), context.kbContext());
            }
            if (StrUtil.isNotBlank(context.mcpContext())) {
                appendSection(mcpBuilder, context.question(), context.mcpContext());
            }
            if (CollUtil.isNotEmpty(context.intentChunks())) {
                mergedIntentChunks.putAll(context.intentChunks());
            }
        }

        return RetrievalContext.builder()
                .mcpContext(mcpBuilder.toString().trim())
                .kbContext(kbBuilder.toString().trim())
                .intentChunks(mergedIntentChunks)
                .build();
    }

    /**
     * 为单个子问题构建完整上下文
     * <p>
     * 将子问题的意图列表分为 KB 和 MCP 两路，分别执行检索/调用，
     * 最后将结果封装为 {@link SubQuestionContext}。
     *
     * @param intent 子问题意图（包含子问题文本和关联的意图节点列表）
     * @param topK   当前子问题的实际 topK
     * @return 子问题上下文，包含 KB 文本、MCP 文本和按意图分组的检索块
     */
    private SubQuestionContext buildSubQuestionContext(SubQuestionIntent intent, int topK) {
        // 按意图类型分流：KB 走向量检索，MCP 走工具调用
        List<NodeScore> kbIntents = NodeScoreFilters.kb(intent.nodeScores());
        List<NodeScore> mcpIntents = NodeScoreFilters.mcp(intent.nodeScores());

        // KB 检索：委托多通道检索引擎执行向量检索 + 后处理（去重、Rerank）
        KbResult kbResult = retrieveAndRerank(intent, kbIntents, topK);

        // MCP 执行：仅在存在 MCP 意图时触发工具调用
        String mcpContext = CollUtil.isNotEmpty(mcpIntents)
                ? executeMcpAndMerge(intent.subQuestion(), mcpIntents)
                : "";

        return new SubQuestionContext(intent.subQuestion(), kbResult.groupedContext(), mcpContext, kbResult.intentChunks());
    }

    /**
     * 子问题实际 TopK 计算规则
     */
    private int resolveSubQuestionTopK(SubQuestionIntent intent, int fallbackTopK) {
        return NodeScoreFilters.kb(intent.nodeScores()).stream()
                .map(NodeScore::getNode)
                .filter(Objects::nonNull)
                .map(IntentNode::getTopK)
                .filter(Objects::nonNull)
                .filter(topK -> topK > 0)
                .max(Integer::compareTo)
                .orElse(fallbackTopK);
    }

    /**
     * 拼接子问题上下文段落到 StringBuilder
     * <p>
     * 格式为 Markdown 结构：以分隔线开头，包含子问题和相关文档两部分。
     *
     * @param builder  目标 StringBuilder
     * @param question 子问题文本
     * @param context  该子问题对应的检索上下文文本
     */
    private void appendSection(StringBuilder builder, String question, String context) {
        builder.append("---\n")
                .append("**子问题**：").append(question).append("\n\n")
                .append("**相关文档**：\n")
                .append(context).append("\n\n");
    }

    /**
     * 执行 MCP 工具调用并合并结果为上下文文本
     * <p>
     * 如果所有工具调用均失败，则返回空字符串，不影响后续 Prompt 构建。
     *
     * @param question   用户子问题文本（用于参数提取）
     * @param mcpIntents MCP 意图列表
     * @return 格式化后的 MCP 上下文文本，全部失败时返回空字符串
     */
    private String executeMcpAndMerge(String question, List<NodeScore> mcpIntents) {
        if (CollUtil.isEmpty(mcpIntents)) {
            return "";
        }

        List<MCPResponse> responses = executeMcpTools(question, mcpIntents);
        if (responses.isEmpty() || responses.stream().noneMatch(MCPResponse::isSuccess)) {
            return "";
        }

        return contextFormatter.formatMcpContext(responses, mcpIntents);
    }

    /**
     * 执行 KB 检索并构建按意图分组的结果
     * <p>
     * 委托 {@link MultiChannelRetrievalEngine} 进行多通道检索（定向 + 全局兜底），
     * 然后将检索到的 Chunk 按意图节点 ID 分组，最后通过 {@link ContextFormatter} 格式化为文本。
     * <p>
     * 注意：多通道检索返回的 Chunk 无法精确对应到某个特定意图节点，
     * 因此采用"全量分配"策略 —— 将所有 Chunk 分配给每个意图节点。
     *
     * @param intent    子问题意图
     * @param kbIntents KB 类型的意图节点列表
     * @param topK      期望返回的结果数量
     * @return {@link KbResult} 包含格式化后的上下文文本和按意图分组的 Chunk 映射
     */
    private KbResult retrieveAndRerank(SubQuestionIntent intent, List<NodeScore> kbIntents, int topK) {
        // 使用多通道检索引擎（是否启用全局检索由置信度阈值决定）
        List<SubQuestionIntent> subIntents = List.of(intent);
        List<RetrievedChunk> chunks = multiChannelRetrievalEngine.retrieveKnowledgeChannels(subIntents, topK);

        if (CollUtil.isEmpty(chunks)) {
            return KbResult.empty();
        }

        // 按意图节点分组（用于格式化上下文）
        Map<String, List<RetrievedChunk>> intentChunks = new HashMap<>();

        // 如果有意图识别结果，按意图节点 ID 分组
        if (CollUtil.isNotEmpty(kbIntents)) {
            // 将所有 chunks 按意图节点 ID 分配
            // 注意：多通道检索返回的 chunks 无法精确对应到某个意图节点
            // 所以我们将所有 chunks 分配给每个意图节点
            for (NodeScore ns : kbIntents) {
                intentChunks.put(ns.getNode().getId(), chunks);
            }
        } else {
            // 如果没有意图识别结果，使用特殊 key
            intentChunks.put(MULTI_CHANNEL_KEY, chunks);
        }

        String groupedContext = contextFormatter.formatKbContext(kbIntents, intentChunks, topK);
        return new KbResult(groupedContext, intentChunks);
    }

    /**
     * 并行执行多个 MCP 工具调用
     * <p>
     * 先将意图节点转换为 {@link MCPRequest}，再使用 {@code mcpBatchExecutor} 线程池并行执行。
     * 每个工具调用互相独立，某个失败不影响其他工具的执行。
     *
     * @param question        用户子问题文本
     * @param mcpIntentScores MCP 意图得分列表
     * @return MCP 工具调用响应列表（包含成功和失败的响应）
     */
    private List<MCPResponse> executeMcpTools(String question, List<NodeScore> mcpIntentScores) {
        if (CollUtil.isEmpty(mcpIntentScores)) {
            return List.of();
        }

        List<CompletableFuture<MCPResponse>> futures = mcpIntentScores.stream()
                .map(ns -> CompletableFuture.supplyAsync(
                        () -> {
                            try {
                                MCPRequest request = buildMcpRequest(question, ns.getNode());
                                return request == null ? null : executeSingleMcpTool(request);
                            } catch (Exception e) {
                                String toolId = ns.getNode().getMcpToolId();
                                log.error("MCP 工具调用异常, toolId: {}", toolId, e);
                                return MCPResponse.error(toolId, "EXECUTION_ERROR", "工具调用异常: " + e.getMessage());
                            }
                        },
                        mcpBatchExecutor
                ))
                .toList();

        return futures.stream()
                .map(CompletableFuture::join)
                .filter(Objects::nonNull)
                .toList();
    }

    /**
     * 执行单个 MCP 工具调用
     * <p>
     * 从注册表中查找执行器并调用，异常时返回错误响应而非抛出异常，
     * 确保单个工具失败不会导致整个检索流程崩溃。
     *
     * @param request MCP 工具请求
     * @return MCP 工具响应（成功或错误）
     */
    private MCPResponse executeSingleMcpTool(MCPRequest request) {
        String toolId = request.getToolId();
        Optional<MCPToolExecutor> executorOpt = mcpToolRegistry.getExecutor(toolId);
        if (executorOpt.isEmpty()) {
            log.warn("MCP 工具执行失败, 工具不存在: {}", toolId);
            return MCPResponse.error(toolId, "TOOL_NOT_FOUND", "工具不存在: " + toolId);
        }

        try {
            return executorOpt.get().execute(request);
        } catch (Exception e) {
            log.error("MCP 工具执行异常, toolId: {}", toolId, e);
            return MCPResponse.error(toolId, "EXECUTION_ERROR", "工具调用异常: " + e.getMessage());
        }
    }

    /**
     * 构建单个 MCP 工具请求
     * <p>
     * 从意图节点获取工具 ID，查找工具定义，使用 {@link MCPParameterExtractor}
     * 从用户问题中提取工具所需参数。支持意图节点上配置的自定义参数提取 Prompt 模板。
     *
     * @param question   用户子问题文本
     * @param intentNode 关联的意图节点（含 mcpToolId 和可选的参数提取模板）
     * @return MCP 请求对象，工具不存在时返回 null
     */
    private MCPRequest buildMcpRequest(String question, IntentNode intentNode) {
        String toolId = intentNode.getMcpToolId();
        Optional<MCPToolExecutor> executorOpt = mcpToolRegistry.getExecutor(toolId);
        if (executorOpt.isEmpty()) {
            log.warn("MCP 工具不存在: {}", toolId);
            return null;
        }

        MCPTool tool = executorOpt.get().getToolDefinition();

        String customParamPrompt = intentNode.getParamPromptTemplate();
        Map<String, Object> params = mcpParameterExtractor.extractParameters(question, tool, customParamPrompt);

        return MCPRequest.builder()
                .toolId(toolId)
                .userQuestion(question)
                .parameters(params != null ? params : new HashMap<>())
                .build();
    }

    /**
     * 子问题上下文值对象
     * <p>
     * 封装单个子问题的检索结果，包含该子问题的原始文本、KB 上下文、MCP 上下文和按意图分组的检索块。
     *
     * @param question     子问题文本
     * @param kbContext    KB 检索格式化后的上下文文本
     * @param mcpContext   MCP 工具调用格式化后的上下文文本
     * @param intentChunks 按意图节点 ID 分组的检索块映射
     */
    private record SubQuestionContext(String question,
                                      String kbContext,
                                      String mcpContext,
                                      Map<String, List<RetrievedChunk>> intentChunks) {
    }
}
