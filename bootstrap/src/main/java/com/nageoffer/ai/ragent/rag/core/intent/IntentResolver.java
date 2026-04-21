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

package com.nageoffer.ai.ragent.rag.core.intent;

import cn.hutool.core.collection.CollUtil;
import com.nageoffer.ai.ragent.rag.dto.IntentCandidate;
import com.nageoffer.ai.ragent.rag.dto.IntentGroup;
import com.nageoffer.ai.ragent.rag.dto.SubQuestionIntent;
import com.nageoffer.ai.ragent.framework.trace.RagTraceNode;
import com.nageoffer.ai.ragent.rag.core.rewrite.RewriteResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.HashMap;
import java.util.concurrent.Executor;

import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.INTENT_MIN_SCORE;
import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.MAX_INTENT_COUNT;
import static com.nageoffer.ai.ragent.rag.enums.IntentKind.SYSTEM;

/**
 * 意图解析编排器 —— RAG 管线意图识别阶段的顶层编排类。
 * <p>
 * 职责：协调查询改写产出的多个子问题，并行调用 {@link IntentClassifier} 进行意图识别，
 * 然后通过分数过滤和总量控制策略，输出最终的意图识别结果。
 * <p>
 * 核心流程：
 * <ol>
 *     <li><b>并行意图识别</b>：通过 {@code intentClassifyExecutor} 线程池，对每个子问题并行调用
 *         {@link IntentClassifier#classifyTargets(String)}，降低多子问题场景下的总延迟。</li>
 *     <li><b>分数阈值过滤</b>：过滤掉 score 低于 {@code INTENT_MIN_SCORE}（0.35）的低置信度意图。</li>
 *     <li><b>总意图数限制</b>：通过 {@link #capTotalIntents(List)} 竞争策略，将全部子问题的意图总数
 *         限制在 {@code MAX_INTENT_COUNT}（3）以内，避免下游检索和 Prompt 过载。</li>
 * </ol>
 * <p>
 * 竞争策略（{@link #capTotalIntents}）：
 * <ul>
 *     <li>每个子问题至少保留 1 个最高分意图（保底策略），确保不会有子问题"完全失声"</li>
 *     <li>剩余配额在所有子问题的候选意图中按分数从高到低竞争分配</li>
 * </ul>
 * <p>
 * 此外提供 {@link #mergeIntentGroup(List)} 方法，将识别结果按类型分组为 MCP 意图和 KB 意图，
 * 供后续检索和 Prompt 构建阶段使用不同的处理策略。
 *
 * @see IntentClassifier 意图分类器接口
 * @see SubQuestionIntent 子问题及其关联意图的封装
 * @see IntentGroup 按类型分组后的意图结果
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class IntentResolver {

    /** 意图分类器，负责对单个问题进行意图识别打分 */
    @Qualifier("defaultIntentClassifier")
    private final IntentClassifier intentClassifier;

    /** 意图分类专用线程池，用于多子问题的并行意图识别 */
    @Qualifier("intentClassifyThreadPoolExecutor")
    private final Executor intentClassifyExecutor;

    /**
     * 对改写结果中的多个子问题并行执行意图识别，并进行总量控制。
     * <p>
     * 处理流程：
     * <ol>
     *     <li>从 {@link RewriteResult} 中提取子问题列表（若为空则使用改写后的总查询作为唯一子问题）</li>
     *     <li>通过 {@code intentClassifyExecutor} 线程池对每个子问题并行调用意图分类器</li>
     *     <li>等待所有并行任务完成，收集各子问题的意图识别结果</li>
     *     <li>调用 {@link #capTotalIntents(List)} 进行总意图数量控制</li>
     * </ol>
     *
     * @param rewriteResult 查询改写阶段的输出结果，包含改写后的总查询和拆分后的子问题列表
     * @return 每个子问题及其关联意图的列表，总意图数不超过 {@code MAX_INTENT_COUNT}
     */
    @RagTraceNode(name = "intent-resolve", type = "INTENT")
    public List<SubQuestionIntent> resolve(RewriteResult rewriteResult) {
        // 优先使用拆分后的子问题列表；若为空，退化为使用改写后的总查询作为唯一子问题
        List<String> subQuestions = CollUtil.isNotEmpty(rewriteResult.subQuestions())
                ? rewriteResult.subQuestions()
                : List.of(rewriteResult.rewrittenQuestion());

        // 对每个子问题提交异步意图分类任务，并行执行以降低总延迟
        List<CompletableFuture<SubQuestionIntent>> tasks = subQuestions.stream()
                .map(q -> CompletableFuture.supplyAsync(
                        () -> {
                            try {
                                return new SubQuestionIntent(q, classifyIntents(q));
                            } catch (Exception e) {
                                log.error("子问题意图分类失败，降级为空意图，question：{}", q, e);
                                return new SubQuestionIntent(q, List.of());
                            }
                        },
                        intentClassifyExecutor
                ))
                .toList();

        // 等待所有并行任务完成并收集结果
        List<SubQuestionIntent> subIntents = tasks.stream()
                .map(CompletableFuture::join)
                .toList();

        // 对汇总后的意图进行总量控制，防止意图数过多导致下游过载
        return capTotalIntents(subIntents);
    }

    /**
     * 将多个子问题的意图识别结果按类型合并分组。
     * <p>
     * 将所有子问题产出的意图分为两类：
     * <ul>
     *     <li><b>MCP 意图</b>：需要调用外部工具（MCP Server）完成的意图，
     *         要求节点类型为 MCP 且关联了有效的 toolId</li>
     *     <li><b>KB 意图</b>：需要从知识库检索文档回答的意图，
     *         节点类型为 KB 或类型未指定（兼容旧数据）</li>
     * </ul>
     * <p>
     * 分组结果供后续 {@code RAGChatServiceImpl} 根据不同类型选择不同的处理策略
     * （如 KB_ONLY / MCP_ONLY / MIXED 三种 Prompt 模板）。
     *
     * @param subIntents 意图识别结果列表（每个元素包含子问题及其关联的意图打分）
     * @return 按 MCP / KB 分组的 {@link IntentGroup}
     */
    public IntentGroup mergeIntentGroup(List<SubQuestionIntent> subIntents) {
        List<NodeScore> mcpIntents = new ArrayList<>();
        List<NodeScore> kbIntents = new ArrayList<>();
        for (SubQuestionIntent si : subIntents) {
            mcpIntents.addAll(NodeScoreFilters.mcp(si.nodeScores()));
            kbIntents.addAll(NodeScoreFilters.kb(si.nodeScores()));
        }
        return new IntentGroup(mcpIntents, kbIntents);
    }

    /**
     * 判断意图识别结果是否为"纯系统指令"类型。
     * <p>
     * 当且仅当只有一个意图且该意图类型为 SYSTEM 时返回 true。
     * SYSTEM 类型的意图（如打招呼、闲聊）不需要走 RAG 检索流程，
     * 可以直接由 LLM 生成回复。
     *
     * @param nodeScores 意图打分列表
     * @return 是否为纯系统指令意图
     */
    public boolean isSystemOnly(List<NodeScore> nodeScores) {
        return nodeScores.size() == 1
                && nodeScores.get(0).getNode() != null
                && nodeScores.get(0).getNode().getKind() == SYSTEM;
    }

    /**
     * 对单个子问题执行意图分类并过滤低分结果。
     * <p>
     * 过滤规则：
     * <ul>
     *     <li>score 低于 {@code INTENT_MIN_SCORE}（0.35）的意图被认为置信度不足，直接丢弃</li>
     *     <li>单个子问题最多保留 {@code MAX_INTENT_COUNT}（3）个意图</li>
     * </ul>
     *
     * @param question 需要识别意图的子问题
     * @return 过滤后的意图打分列表（按 score 降序）
     */
    private List<NodeScore> classifyIntents(String question) {
        List<NodeScore> scores = intentClassifier.classifyTargets(question);
        return scores.stream()
                .filter(ns -> ns.getScore() >= INTENT_MIN_SCORE)
                .limit(MAX_INTENT_COUNT)
                .toList();
    }

    /**
     * 限制所有子问题的总意图数量不超过 {@code MAX_INTENT_COUNT}（竞争策略）。
     * <p>
     * 当多个子问题的意图总数超过上限时，需要在公平性和准确性之间做权衡。
     * 本方法采用"保底 + 竞争"的两阶段策略：
     * <ol>
     *     <li><b>保底阶段</b>：每个子问题至少保留 1 个最高分意图，确保不会有子问题
     *         完全没有意图（避免部分用户问题被"静默忽略"）</li>
     *     <li><b>竞争阶段</b>：将剩余配额（{@code MAX_INTENT_COUNT - 保底数量}）分配给
     *         所有子问题的剩余候选意图，按 score 从高到低竞争获取</li>
     * </ol>
     * <p>
     * 如果总意图数未超限，直接返回原结果，不做任何裁剪。
     *
     * @param subIntents 各子问题的意图识别结果列表
     * @return 裁剪后的意图识别结果列表，总意图数不超过 {@code MAX_INTENT_COUNT}
     */
    private List<SubQuestionIntent> capTotalIntents(List<SubQuestionIntent> subIntents) {
        int totalIntents = subIntents.stream()
                .mapToInt(si -> si.nodeScores().size())
                .sum();

        // 未超限，直接返回
        if (totalIntents <= MAX_INTENT_COUNT) {
            return subIntents;
        }

        // 步骤1：将所有子问题的意图展开为扁平候选列表，并按 score 降序排列
        List<IntentCandidate> allCandidates = collectAllCandidates(subIntents);

        // 步骤2：保底策略——每个子问题至少保留 1 个最高分意图
        List<IntentCandidate> guaranteedIntents = selectTopIntentPerSubQuestion(allCandidates, subIntents.size());

        // 步骤3：计算竞争阶段的剩余配额
        int remaining = MAX_INTENT_COUNT - guaranteedIntents.size();

        // 步骤4：从未被保底选中的候选中，按 score 降序竞争分配剩余配额
        List<IntentCandidate> additionalIntents = selectAdditionalIntents(allCandidates, guaranteedIntents, remaining);

        // 步骤5：合并保底意图和竞争意图，按子问题索引重建结果结构
        return rebuildSubIntents(subIntents, guaranteedIntents, additionalIntents);
    }

    /**
     * 收集所有子问题的意图候选，标记每个候选所属的子问题索引，并按 score 降序排列。
     * <p>
     * 将树形的 {@code List<SubQuestionIntent>} 展开为扁平的 {@code List<IntentCandidate>}，
     * 便于后续的全局排序和竞争选择。
     *
     * @param subIntents 各子问题的意图识别结果
     * @return 按 score 降序排列的全部意图候选列表
     */
    private List<IntentCandidate> collectAllCandidates(List<SubQuestionIntent> subIntents) {
        List<IntentCandidate> candidates = new ArrayList<>();
        for (int i = 0; i < subIntents.size(); i++) {
            List<NodeScore> nodeScores = subIntents.get(i).nodeScores();
            if (CollUtil.isEmpty(nodeScores)) {
                continue;
            }
            for (NodeScore ns : nodeScores) {
                candidates.add(new IntentCandidate(i, ns));
            }
        }
        // 按分数降序排序
        candidates.sort((a, b) -> Double.compare(b.nodeScore().getScore(), a.nodeScore().getScore()));
        return candidates;
    }

    /**
     * 保底策略：为每个子问题选择得分最高的一个意图。
     * <p>
     * 由于 {@code allCandidates} 已按 score 降序排列，对于每个子问题索引，
     * 第一个出现的候选就是该子问题得分最高的意图。
     * <p>
     * 这确保了即使总配额紧张，每个子问题至少有一个意图被保留，
     * 避免部分用户问题被完全忽略。
     *
     * @param allCandidates    按 score 降序排列的全部候选
     * @param subQuestionCount 子问题总数
     * @return 每个子问题的最高分意图列表
     */
    private List<IntentCandidate> selectTopIntentPerSubQuestion(List<IntentCandidate> allCandidates, int subQuestionCount) {
        List<IntentCandidate> topIntents = new ArrayList<>();
        boolean[] selected = new boolean[subQuestionCount];

        for (IntentCandidate candidate : allCandidates) {
            int index = candidate.subQuestionIndex();
            if (!selected[index]) {
                topIntents.add(candidate);
                selected[index] = true;
            }
            // 所有子问题都有了保底意图，提前退出
            if (topIntents.size() == subQuestionCount) {
                break;
            }
        }
        return topIntents;
    }

    /**
     * 竞争策略：从剩余候选中按 score 降序选择额外意图，填充剩余配额。
     * <p>
     * 跳过已在保底阶段被选中的候选，从剩余候选中按 score 从高到低依次选取，
     * 直到填满剩余配额或候选耗尽。
     *
     * @param allCandidates     按 score 降序排列的全部候选
     * @param guaranteedIntents 保底阶段已选中的意图（需排除）
     * @param remaining         剩余配额数量
     * @return 额外选中的意图列表
     */
    private List<IntentCandidate> selectAdditionalIntents(List<IntentCandidate> allCandidates,
                                                          List<IntentCandidate> guaranteedIntents,
                                                          int remaining) {
        if (remaining <= 0) {
            return List.of();
        }

        List<IntentCandidate> additional = new ArrayList<>();
        for (IntentCandidate candidate : allCandidates) {
            // 跳过已经被选为保底的意图
            if (guaranteedIntents.contains(candidate)) {
                continue;
            }
            additional.add(candidate);
            if (additional.size() >= remaining) {
                break;
            }
        }
        return additional;
    }

    /**
     * 根据选中的意图候选重建 {@link SubQuestionIntent} 列表。
     * <p>
     * 将保底阶段和竞争阶段选出的意图合并，按子问题索引重新分组，
     * 重建与原始结构一致的 {@code SubQuestionIntent} 列表。
     * 未被分配到任何意图的子问题将获得空的 nodeScores 列表。
     *
     * @param originalSubIntents 原始的子问题意图列表（用于提取子问题文本）
     * @param guaranteedIntents  保底阶段选中的意图
     * @param additionalIntents  竞争阶段选中的额外意图
     * @return 裁剪后的 {@link SubQuestionIntent} 列表
     */
    private List<SubQuestionIntent> rebuildSubIntents(List<SubQuestionIntent> originalSubIntents,
                                                      List<IntentCandidate> guaranteedIntents,
                                                      List<IntentCandidate> additionalIntents) {
        // 合并保底意图和竞争意图为统一列表
        List<IntentCandidate> allSelected = new ArrayList<>(guaranteedIntents);
        allSelected.addAll(additionalIntents);

        // 按子问题索引分组，便于重建与原始结构对应的结果
        Map<Integer, List<NodeScore>> groupedByIndex = new HashMap<>();
        for (IntentCandidate candidate : allSelected) {
            groupedByIndex.computeIfAbsent(candidate.subQuestionIndex(), k -> new ArrayList<>())
                    .add(candidate.nodeScore());
        }

        // 按原始子问题顺序重建结果，保持与输入一致的索引对应关系
        List<SubQuestionIntent> result = new ArrayList<>();
        for (int i = 0; i < originalSubIntents.size(); i++) {
            SubQuestionIntent original = originalSubIntents.get(i);
            List<NodeScore> retained = groupedByIndex.getOrDefault(i, List.of());
            result.add(new SubQuestionIntent(original.subQuestion(), retained));
        }
        return result;
    }
}
