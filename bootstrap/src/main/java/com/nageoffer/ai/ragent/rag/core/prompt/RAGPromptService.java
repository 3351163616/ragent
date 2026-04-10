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

package com.nageoffer.ai.ragent.rag.core.prompt;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.RetrievedChunk;
import com.nageoffer.ai.ragent.rag.core.intent.IntentNode;
import com.nageoffer.ai.ragent.rag.core.intent.NodeScore;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.MCP_KB_MIXED_PROMPT_PATH;
import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.MCP_ONLY_PROMPT_PATH;
import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.RAG_ENTERPRISE_PROMPT_PATH;

/**
 * RAG Prompt 编排服务 —— RAG 管线的"Prompt 构建"阶段
 * <p>
 * 该服务处于 RAG 管线的尾端：检索 + 后处理完成后、LLM 流式生成之前。
 * 核心职责：
 * <ol>
 *     <li>根据上下文中是否存在 KB（知识库检索结果）和 MCP（动态工具调用结果），
 *         判定当前属于哪种场景（{@link PromptScene#KB_ONLY}、{@link PromptScene#MCP_ONLY}、{@link PromptScene#MIXED}）</li>
 *     <li>单意图场景下优先使用意图节点自带的 promptTemplate；无自定义模板或多意图场景下回退到默认模板</li>
 *     <li>将系统提示词、KB/MCP 上下文、历史消息和用户问题组装为有序的 {@link ChatMessage} 列表，
 *         交给下游 LLM 服务进行生成</li>
 * </ol>
 *
 * @see PromptContext  封装检索阶段产出的全部上下文
 * @see PromptScene   场景枚举：KB_ONLY / MCP_ONLY / MIXED / EMPTY
 * @see PromptBuildPlan 内部决策结果，记录选定的场景和基础模板
 */
@Service
@RequiredArgsConstructor
public class RAGPromptService {

    /** MCP 动态数据片段在 Prompt 中的标题前缀 */
    private static final String MCP_CONTEXT_HEADER = "## 动态数据片段";
    /** KB 知识库文档内容在 Prompt 中的标题前缀 */
    private static final String KB_CONTEXT_HEADER = "## 文档内容";

    /** 模板加载器，负责从 classpath 加载 .st 模板文件并缓存 */
    private final PromptTemplateLoader promptTemplateLoader;

    /**
     * 根据上下文决策场景并生成系统提示词
     * <p>
     * 逻辑流程：
     * <ol>
     *     <li>通过 {@link #plan(PromptContext)} 判定场景并获取可能的自定义模板</li>
     *     <li>若自定义模板为空，则按场景加载默认 classpath 模板</li>
     *     <li>最终对模板做清理（去除多余空行、占位符残留等）</li>
     * </ol>
     *
     * @param context 包含 KB/MCP 上下文、意图列表等信息的 Prompt 上下文对象
     * @return 清理后的系统提示词字符串；若无可用模板则返回空字符串
     */
    public String buildSystemPrompt(PromptContext context) {
        // 执行场景判定，获取决策方案（含场景类型和可能的自定义模板）
        PromptBuildPlan plan = plan(context);
        // 优先使用意图节点自带的 Prompt 模板，否则回退到对应场景的默认模板
        String template = StrUtil.isNotBlank(plan.getBaseTemplate())
                ? plan.getBaseTemplate()
                : defaultTemplate(plan.getScene());
        return StrUtil.isBlank(template) ? "" : PromptTemplateUtils.cleanupPrompt(template);
    }

    /**
     * 构造发送给 LLM 的完整消息列表
     * <p>
     * 按照如下固定顺序组装消息序列：
     * <ol>
     *     <li><b>system 消息</b> — 系统提示词（由 {@link #buildSystemPrompt} 生成）</li>
     *     <li><b>MCP 证据（system 角色）</b> — 如果存在 MCP 动态数据片段，以 system 消息注入</li>
     *     <li><b>KB 证据（user 角色）</b> — 如果存在 KB 检索结果，以 user 消息注入。
     *         这里使用 user 角色是为了让模型更关注文档内容，提高引用准确率</li>
     *     <li><b>历史消息</b> — 多轮对话的历史上下文</li>
     *     <li><b>当前用户问题</b> — 单问题直接追加；多子问题时编号列出以降低模型漏答风险</li>
     * </ol>
     *
     * @param context      Prompt 上下文，含 KB/MCP 检索结果
     * @param history      历史对话消息列表，可为空
     * @param question     用户原始问题
     * @param subQuestions 查询重写/拆分后的子问题列表，可为空
     * @return 有序的 {@link ChatMessage} 列表，可直接传给 LLM
     */
    public List<ChatMessage> buildStructuredMessages(PromptContext context,
                                                     List<ChatMessage> history,
                                                     String question,
                                                     List<String> subQuestions) {
        List<ChatMessage> messages = new ArrayList<>();
        String systemPrompt = buildSystemPrompt(context);
        if (StrUtil.isNotBlank(systemPrompt)) {
            messages.add(ChatMessage.system(systemPrompt));
        }
        // MCP 上下文以 system 角色注入，让模型将其视为可信参考数据
        if (StrUtil.isNotBlank(context.getMcpContext())) {
            messages.add(ChatMessage.system(formatEvidence(MCP_CONTEXT_HEADER, context.getMcpContext())));
        }
        // KB 上下文以 user 角色注入，引导模型在回答时优先引用文档
        if (StrUtil.isNotBlank(context.getKbContext())) {
            messages.add(ChatMessage.user(formatEvidence(KB_CONTEXT_HEADER, context.getKbContext())));
        }
        if (CollUtil.isNotEmpty(history)) {
            messages.addAll(history);
        }

        // 多子问题场景下，显式编号以降低模型漏答风险
        if (CollUtil.isNotEmpty(subQuestions) && subQuestions.size() > 1) {
            StringBuilder userMessage = new StringBuilder();
            userMessage.append("请基于上述文档内容，回答以下问题：\n\n");
            for (int i = 0; i < subQuestions.size(); i++) {
                userMessage.append(i + 1).append(". ").append(subQuestions.get(i)).append("\n");
            }
            messages.add(ChatMessage.user(userMessage.toString().trim()));
        } else if (StrUtil.isNotBlank(question)) {
            messages.add(ChatMessage.user(question));
        }

        return messages;
    }

    /**
     * 内部 Prompt 规划：根据意图列表和对应检索结果，决定最终使用的模板策略
     * <p>
     * 规划逻辑：
     * <ol>
     *     <li>过滤掉没有命中检索结果的意图（即 intentChunks 中无对应 chunk 的意图）</li>
     *     <li>若过滤后没有任何可用意图，返回空模板，上层可做 fallback 处理</li>
     *     <li>单意图且该意图节点配有自定义 promptTemplate，则直接使用该模板</li>
     *     <li>单意图无自定义模板或多意图场景，统一使用默认模板</li>
     * </ol>
     *
     * @param intents      意图识别阶段输出的 {@link NodeScore} 列表
     * @param intentChunks 每个意图节点对应的检索结果，key 为 nodeKey
     * @return 包含有效意图列表和基础模板的 {@link PromptPlan}
     */
    private PromptPlan planPrompt(List<NodeScore> intents, Map<String, List<RetrievedChunk>> intentChunks) {
        // 防御性处理：意图列表为 null 时视为空列表
        List<NodeScore> safeIntents = intents == null ? Collections.emptyList() : intents;

        // 第一步：剔除在检索阶段未命中任何文档块的意图
        // 只有实际拿到检索结果的意图才有意义参与 Prompt 构建
        List<NodeScore> retained = safeIntents.stream()
                .filter(ns -> {
                    IntentNode node = ns.getNode();
                    String key = nodeKey(node);
                    List<RetrievedChunk> chunks = intentChunks == null ? null : intentChunks.get(key);
                    return CollUtil.isNotEmpty(chunks);
                })
                .toList();

        if (retained.isEmpty()) {
            // 没有任何可用意图：返回空模板，上层根据业务场景决定 fallback 策略
            return new PromptPlan(Collections.emptyList(), null);
        }

        // 第二步：根据有效意图数量决定模板策略
        if (retained.size() == 1) {
            // 单意图场景：检查意图节点是否携带了自定义 Prompt 模板
            IntentNode only = retained.get(0).getNode();
            String tpl = StrUtil.emptyIfNull(only.getPromptTemplate()).trim();

            if (StrUtil.isNotBlank(tpl)) {
                // 使用意图节点级别的自定义模板（更精准的 Prompt 控制）
                return new PromptPlan(retained, tpl);
            } else {
                // 无自定义模板，后续将使用场景级默认模板
                return new PromptPlan(retained, null);
            }
        } else {
            // 多意图场景：不同意图模板可能冲突，统一使用默认模板
            return new PromptPlan(retained, null);
        }
    }

    /**
     * 场景判定入口：根据 PromptContext 中 MCP/KB 的有无，路由到对应的规划方法
     * <p>
     * 三种互斥场景：
     * <ul>
     *     <li>仅 MCP → {@link #planMcpOnly}</li>
     *     <li>仅 KB → {@link #planKbOnly}</li>
     *     <li>MCP + KB 混合 → {@link #planMixed}</li>
     * </ul>
     * 若两者均不存在，抛出 {@link IllegalStateException}（属于编程错误，不应到达）
     *
     * @param context Prompt 上下文
     * @return 场景决策结果
     */
    private PromptBuildPlan plan(PromptContext context) {
        if (context.hasMcp() && !context.hasKb()) {
            return planMcpOnly(context);
        }
        if (!context.hasMcp() && context.hasKb()) {
            return planKbOnly(context);
        }
        if (context.hasMcp() && context.hasKb()) {
            return planMixed(context);
        }
        throw new IllegalStateException("PromptContext requires MCP or KB context.");
    }

    /**
     * 纯 KB 场景规划：委托 {@link #planPrompt} 判断意图模板策略
     *
     * @param context Prompt 上下文
     * @return 场景为 KB_ONLY 的决策结果
     */
    private PromptBuildPlan planKbOnly(PromptContext context) {
        PromptPlan plan = planPrompt(context.getKbIntents(), context.getIntentChunks());
        return PromptBuildPlan.builder()
                .scene(PromptScene.KB_ONLY)
                .baseTemplate(plan.getBaseTemplate())
                .mcpContext(context.getMcpContext())
                .kbContext(context.getKbContext())
                .question(context.getQuestion())
                .build();
    }

    /**
     * 纯 MCP 场景规划：仅在单意图且有自定义模板时使用节点模板
     * <p>
     * 与 KB 不同，MCP 不走 planPrompt 通用流程，因为 MCP 意图没有检索结果需要过滤
     *
     * @param context Prompt 上下文
     * @return 场景为 MCP_ONLY 的决策结果
     */
    private PromptBuildPlan planMcpOnly(PromptContext context) {
        List<NodeScore> intents = context.getMcpIntents();
        String baseTemplate = null;
        if (CollUtil.isNotEmpty(intents) && intents.size() == 1) {
            IntentNode node = intents.get(0).getNode();
            String tpl = StrUtil.emptyIfNull(node.getPromptTemplate()).trim();
            if (StrUtil.isNotBlank(tpl)) {
                baseTemplate = tpl;
            }
        }

        return PromptBuildPlan.builder()
                .scene(PromptScene.MCP_ONLY)
                .baseTemplate(baseTemplate)
                .mcpContext(context.getMcpContext())
                .kbContext(context.getKbContext())
                .question(context.getQuestion())
                .build();
    }

    /**
     * 混合场景规划：MCP 与 KB 同时存在，不使用意图级模板，统一走混合默认模板
     * <p>
     * 混合场景的模板需要同时引导模型处理知识库文档和动态工具数据，
     * 因此不适合使用单意图节点级的自定义模板
     *
     * @param context Prompt 上下文
     * @return 场景为 MIXED 的决策结果
     */
    private PromptBuildPlan planMixed(PromptContext context) {
        return PromptBuildPlan.builder()
                .scene(PromptScene.MIXED)
                .mcpContext(context.getMcpContext())
                .kbContext(context.getKbContext())
                .question(context.getQuestion())
                .build();
    }

    /**
     * 根据场景枚举加载对应的默认 Prompt 模板文件
     * <p>
     * 模板文件存放于 classpath 下，通过 {@link PromptTemplateLoader} 缓存加载：
     * <ul>
     *     <li>KB_ONLY → rag-enterprise.st（企业知识问答模板）</li>
     *     <li>MCP_ONLY → mcp-only.st（纯工具调用模板）</li>
     *     <li>MIXED → mcp-kb-mixed.st（混合模板）</li>
     *     <li>EMPTY → 空字符串</li>
     * </ul>
     *
     * @param scene 当前判定的 Prompt 场景
     * @return 模板内容字符串
     */
    private String defaultTemplate(PromptScene scene) {
        return switch (scene) {
            case KB_ONLY -> promptTemplateLoader.load(RAG_ENTERPRISE_PROMPT_PATH);
            case MCP_ONLY -> promptTemplateLoader.load(MCP_ONLY_PROMPT_PATH);
            case MIXED -> promptTemplateLoader.load(MCP_KB_MIXED_PROMPT_PATH);
            case EMPTY -> "";
        };
    }

    /**
     * 将标题和正文拼接为一段带格式的证据文本
     *
     * @param header 证据区域标题（如 "## 文档内容"）
     * @param body   证据正文内容
     * @return 拼接后的证据文本
     */
    private String formatEvidence(String header, String body) {
        return header + "\n" + body.trim();
    }

    // === 工具方法 ===

    /**
     * 从意图节点提取用于映射检索结果的 key
     */
    private static String nodeKey(IntentNode node) {
        if (node == null) return "";
        if (StrUtil.isNotBlank(node.getId())) return node.getId();
        return String.valueOf(node.getId());
    }

}
