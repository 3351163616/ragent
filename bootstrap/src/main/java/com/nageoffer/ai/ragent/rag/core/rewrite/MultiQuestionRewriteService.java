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

package com.nageoffer.ai.ragent.rag.core.rewrite;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.nageoffer.ai.ragent.infra.util.LLMResponseCleaner;
import com.nageoffer.ai.ragent.rag.config.RAGConfigProperties;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.framework.trace.RagTraceNode;
import com.nageoffer.ai.ragent.infra.chat.LLMService;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptTemplateLoader;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.QUERY_REWRITE_AND_SPLIT_PROMPT_PATH;

/**
 * 多问题查询改写服务 —— RAG 管线查询预处理阶段的核心实现。
 * <p>
 * 完整处理流程：
 * <ol>
 *     <li><b>术语归一化</b>：通过 {@link QueryTermMappingService} 将用户口语化表达映射为标准术语
 *         （例如"社保" → "社会保险"），采用长词优先匹配策略避免短词误替换。</li>
 *     <li><b>LLM 改写 + 拆分</b>：将归一化后的问题连同最近 2 轮对话历史一起发送给 LLM，
 *         由 LLM 完成指代消解（如"它" → "OA系统"）并将复合问句拆分为独立子问题。
 *         LLM 返回 JSON 格式：{@code {"rewrite": "...", "sub_questions": ["...", "..."]}}。</li>
 *     <li><b>兜底规则拆分</b>：当查询改写开关关闭或 LLM 调用/解析失败时，回退为基于标点符号的规则拆分。</li>
 * </ol>
 * <p>
 * 设计考量：
 * <ul>
 *     <li>LLM 请求使用低温度（0.1）和低 topP（0.3），确保改写结果稳定可控。</li>
 *     <li>会话历史只保留最近 2 轮（4 条消息），且过滤掉 System 摘要，控制 Token 消耗。</li>
 *     <li>整个链路通过 {@code @RagTraceNode} 注解进行链路追踪埋点。</li>
 * </ul>
 *
 * @see QueryRewriteService 查询改写服务接口
 * @see QueryTermMappingService 术语归一化服务
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MultiQuestionRewriteService implements QueryRewriteService {

    /** LLM 服务，用于调用大语言模型完成查询改写和拆分 */
    private final LLMService llmService;

    /** RAG 配置属性，包含查询改写开关等配置项 */
    private final RAGConfigProperties ragConfigProperties;

    /** 术语映射服务，负责将用户口语化表达归一化为标准检索术语 */
    private final QueryTermMappingService queryTermMappingService;

    /** Prompt 模板加载器，负责加载和渲染 StringTemplate 格式的提示词模板 */
    private final PromptTemplateLoader promptTemplateLoader;

    /**
     * 纯改写入口（不带会话历史）。
     * <p>
     * 委托给 {@link #rewriteAndSplit(String)} 完成改写+拆分，但只返回改写后的总查询。
     * 通过 {@code @RagTraceNode} 进行链路追踪埋点，记录改写阶段的耗时和结果。
     *
     * @param userQuestion 原始用户问题
     * @return 改写后的检索查询
     */
    @Override
    @RagTraceNode(name = "query-rewrite", type = "REWRITE")
    public String rewrite(String userQuestion) {
        return rewriteAndSplit(userQuestion).rewrittenQuestion();
    }

    /**
     * 改写 + 拆分入口（不带会话历史）。
     * <p>
     * 委托给内部的 {@link #rewriteAndSplit(String)} 方法。
     *
     * @param userQuestion 原始用户问题
     * @return 包含改写结果和拆分后子问题列表的 {@link RewriteResult}
     */
    @Override
    public RewriteResult rewriteWithSplit(String userQuestion) {
        return rewriteAndSplit(userQuestion);
    }

    /**
     * 改写 + 拆分入口（带会话历史版本，用于多轮对话的指代消解）。
     * <p>
     * 处理流程：
     * <ol>
     *     <li>检查查询改写开关，若关闭则仅做术语归一化 + 规则拆分</li>
     *     <li>先对用户问题进行术语归一化</li>
     *     <li>将归一化后的问题连同会话历史发送给 LLM 完成改写和拆分</li>
     * </ol>
     *
     * @param userQuestion 原始用户问题
     * @param history      最近的会话历史消息列表，用于 LLM 做指代消解
     * @return 包含改写结果和拆分后子问题列表的 {@link RewriteResult}
     */
    @Override
    @RagTraceNode(name = "query-rewrite-and-split", type = "REWRITE")
    public RewriteResult rewriteWithSplit(String userQuestion, List<ChatMessage> history) {
        if (!ragConfigProperties.getQueryRewriteEnabled()) {
            String normalized = queryTermMappingService.normalize(userQuestion);
            List<String> subs = ruleBasedSplit(normalized);
            return new RewriteResult(normalized, subs);
        }

        String normalizedQuestion = queryTermMappingService.normalize(userQuestion);

        return callLLMRewriteAndSplit(normalizedQuestion, userQuestion, history);
    }

    /**
     * 内部改写 + 拆分（不带会话历史的简化版本）。
     * <p>
     * 与 {@link #rewriteWithSplit(String, List)} 逻辑一致，但传入空的历史列表。
     * 当从 {@link #rewrite(String)} 调用时走此方法。
     *
     * @param userQuestion 原始用户问题
     * @return 改写 + 拆分结果
     */
    private RewriteResult rewriteAndSplit(String userQuestion) {
        // 开关关闭：直接做规则归一化 + 规则拆分，跳过 LLM 调用
        if (!ragConfigProperties.getQueryRewriteEnabled()) {
            String normalized = queryTermMappingService.normalize(userQuestion);
            List<String> subs = ruleBasedSplit(normalized);
            return new RewriteResult(normalized, subs);
        }

        String normalizedQuestion = queryTermMappingService.normalize(userQuestion);

        // 传入空历史列表，表示无多轮对话上下文
        return callLLMRewriteAndSplit(normalizedQuestion, userQuestion, List.of());

        // 兜底：使用归一化结果 + 规则拆分
    }

    /**
     * 调用 LLM 完成查询改写和多问句拆分的核心方法。
     * <p>
     * 流程：
     * <ol>
     *     <li>加载改写 + 拆分的 Prompt 模板</li>
     *     <li>构建包含系统提示、会话历史和归一化问题的 ChatRequest</li>
     *     <li>调用 LLM 获取原始响应</li>
     *     <li>解析 LLM 返回的 JSON 提取改写结果和子问题</li>
     *     <li>如果解析失败或 LLM 调用异常，统一回退到归一化问题作为兜底</li>
     * </ol>
     *
     * @param normalizedQuestion 经术语归一化后的用户问题（发送给 LLM 的实际输入）
     * @param originalQuestion   未经处理的原始用户问题（仅用于日志记录）
     * @param history            会话历史消息列表（可为空列表）
     * @return 改写 + 拆分结果；LLM 失败时返回归一化问题作为唯一子问题
     */
    private RewriteResult callLLMRewriteAndSplit(String normalizedQuestion,
                                                 String originalQuestion,
                                                 List<ChatMessage> history) {
        String systemPrompt = promptTemplateLoader.load(QUERY_REWRITE_AND_SPLIT_PROMPT_PATH);
        ChatRequest req = buildRewriteRequest(systemPrompt, normalizedQuestion, history);

        try {
            String raw = llmService.chat(req);
            RewriteResult parsed = parseRewriteAndSplit(raw);

            if (parsed != null) {
                log.info("""
                        RAG用户问题查询改写+拆分：
                        原始问题：{}
                        归一化后：{}
                        改写结果：{}
                        子问题：{}
                        """, originalQuestion, normalizedQuestion, parsed.rewrittenQuestion(), parsed.subQuestions());
                return parsed;
            }

            log.warn("查询改写+拆分解析失败，使用归一化问题兜底 - normalizedQuestion={}", normalizedQuestion);
        } catch (Exception e) {
            log.warn("查询改写+拆分 LLM 调用失败，使用归一化问题兜底 - question={}，normalizedQuestion={}", originalQuestion, normalizedQuestion, e);
        }

        // LLM 调用或解析失败时的统一兜底：直接使用归一化问题作为改写结果和唯一子问题
        return new RewriteResult(normalizedQuestion, List.of(normalizedQuestion));
    }

    /**
     * 构建发送给 LLM 的查询改写请求。
     * <p>
     * 消息组装顺序：System Prompt → 会话历史（最近 2 轮）→ 当前用户问题。
     * <p>
     * 设计考量：
     * <ul>
     *     <li>只保留 User 和 Assistant 角色的消息，过滤掉 System 摘要以节省 Token</li>
     *     <li>最多保留最近 4 条消息（约 2 轮对话），足够做指代消解又不会引入过多噪声</li>
     *     <li>使用低温度（0.1）和低 topP（0.3）确保改写结果的确定性和稳定性</li>
     *     <li>关闭 thinking 模式，减少不必要的推理 Token 消耗</li>
     * </ul>
     *
     * @param systemPrompt 系统提示词（改写 + 拆分的指令模板）
     * @param question     归一化后的用户问题
     * @param history      会话历史消息列表
     * @return 构建好的 {@link ChatRequest} 对象
     */
    private ChatRequest buildRewriteRequest(String systemPrompt,
                                            String question,
                                            List<ChatMessage> history) {
        List<ChatMessage> messages = new ArrayList<>();
        if (StrUtil.isNotBlank(systemPrompt)) {
            messages.add(ChatMessage.system(systemPrompt));
        }

        // 只保留最近 1-2 轮的 User 和 Assistant 消息，用于指代消解
        // 过滤掉 System 角色的摘要消息，避免额外 Token 浪费
        if (CollUtil.isNotEmpty(history)) {
            List<ChatMessage> recentHistory = history.stream()
                    .filter(msg -> msg.getRole() == ChatMessage.Role.USER
                            || msg.getRole() == ChatMessage.Role.ASSISTANT)
                    .skip(Math.max(0, history.size() - 4))  // 最多保留最近 4 条消息（即 2 轮对话）
                    .toList();
            messages.addAll(recentHistory);
        }

        // 当前用户问题放在消息列表末尾，作为 LLM 需要改写的目标
        messages.add(ChatMessage.user(question));

        // 使用低温度和低 topP 保证输出的确定性，关闭 thinking 模式减少 Token 消耗
        return ChatRequest.builder()
                .messages(messages)
                .temperature(0.1D)
                .topP(0.3D)
                .thinking(false)
                .build();
    }


    /**
     * 解析 LLM 返回的改写 + 拆分 JSON 响应。
     * <p>
     * 预期 JSON 格式：{@code {"rewrite": "改写后的问题", "sub_questions": ["子问题1", "子问题2"]}}
     * <p>
     * 解析规则：
     * <ul>
     *     <li>先通过 {@link LLMResponseCleaner} 移除可能存在的 Markdown 代码块标记（如 ```json ... ```）</li>
     *     <li>如果 rewrite 字段为空，返回 null 表示解析失败</li>
     *     <li>如果 sub_questions 为空或不存在，将 rewrite 作为唯一子问题</li>
     * </ul>
     *
     * @param raw LLM 返回的原始文本
     * @return 解析后的 {@link RewriteResult}；解析失败时返回 null
     */
    private RewriteResult parseRewriteAndSplit(String raw) {
        try {
            // 移除可能存在的 Markdown 代码块标记
            String cleaned = LLMResponseCleaner.stripMarkdownCodeFence(raw);

            JsonElement root = JsonParser.parseString(cleaned);
            if (!root.isJsonObject()) {
                return null;
            }
            JsonObject obj = root.getAsJsonObject();
            String rewrite = obj.has("rewrite") ? obj.get("rewrite").getAsString().trim() : "";
            List<String> subs = new ArrayList<>();
            if (obj.has("sub_questions") && obj.get("sub_questions").isJsonArray()) {
                JsonArray arr = obj.getAsJsonArray("sub_questions");
                for (JsonElement el : arr) {
                    if (el.isJsonPrimitive() && el.getAsJsonPrimitive().isString()) {
                        String s = el.getAsString().trim();
                        if (StrUtil.isNotBlank(s)) {
                            subs.add(s);
                        }
                    }
                }
            }
            if (StrUtil.isBlank(rewrite)) {
                return null;
            }
            if (CollUtil.isEmpty(subs)) {
                subs = List.of(rewrite);
            }
            return new RewriteResult(rewrite, subs);
        } catch (Exception e) {
            log.warn("解析改写+拆分结果失败，raw={}", raw, e);
            return null;
        }
    }

    /**
     * 基于规则的问句拆分（兜底方案）。
     * <p>
     * 当 LLM 改写不可用时，使用简单的标点分隔符将复合问句拆分为多个子问题。
     * 支持的分隔符包括：? ？ 。 ； ; 以及换行符。
     * <p>
     * 拆分后会确保每个子问题以问号结尾，方便后续意图识别。
     *
     * @param question 需要拆分的问题（通常已经过术语归一化）
     * @return 拆分后的子问题列表；如果无法拆分则返回原问题作为唯一元素
     */
    private List<String> ruleBasedSplit(String question) {
        // 按常见标点分隔符拆分，过滤空白片段
        List<String> parts = Arrays.stream(question.split("[?？。；;\\n]+"))
                .map(String::trim)
                .filter(StrUtil::isNotBlank)
                .collect(Collectors.toList());

        if (CollUtil.isEmpty(parts)) {
            return List.of(question);
        }
        // 确保每个子问题以问号结尾，统一格式便于后续处理
        return parts.stream()
                .map(s -> s.endsWith("？") || s.endsWith("?") ? s : s + "？")
                .toList();
    }
}
