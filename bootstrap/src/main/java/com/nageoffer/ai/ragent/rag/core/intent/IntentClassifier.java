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

import java.util.List;

/**
 * 意图分类器接口 —— RAG 管线意图识别阶段的核心抽象。
 * <p>
 * 在查询改写和拆分完成后，意图分类器负责对用户问题（或拆分后的子问题）进行意图识别，
 * 判断用户问题属于哪个业务领域或操作类型（如知识库查询 KB、工具调用 MCP、系统指令 SYSTEM 等）。
 * <p>
 * 意图分类的结果将直接影响后续的检索通道选择和 Prompt 构建策略。
 * <p>
 * 支持两种实现策略：
 * <ul>
 *     <li><b>串行分类</b>：所有意图节点在单次 LLM 调用中完成识别打分，适用于意图数量较少的场景。
 *         参见 {@link DefaultIntentClassifier}。</li>
 *     <li><b>并行分类</b>：按 Domain 拆分意图子树，并行调用多个 LLM 分别识别，
 *         适用于意图数量多、需要降低单次调用延迟的场景。</li>
 * </ul>
 *
 * @see DefaultIntentClassifier 基于 LLM 的串行意图分类器实现
 * @see IntentResolver 意图解析编排器，负责对多子问题并行调用本接口
 * @see NodeScore 意图节点打分结果
 */
public interface IntentClassifier {

    /**
     * 对所有叶子分类节点进行意图识别打分。
     * <p>
     * 将用户问题发送给 LLM，LLM 对预定义的意图树中的每个叶子节点进行打分，
     * 返回按 score 从高到低排序的打分列表。
     * <p>
     * 叶子节点代表最细粒度的意图分类，例如"OA系统 > 审批流程"。
     *
     * @param question 用户问题（通常是经过改写后的子问题）
     * @return 按 score 降序排列的 {@link NodeScore} 列表；若识别失败则返回空列表
     */
    List<NodeScore> classifyTargets(String question);

    /**
     * 获取 Top-K 个且分数高于阈值的意图分类结果。
     * <p>
     * 在 {@link #classifyTargets(String)} 的基础上做双重过滤：
     * <ol>
     *     <li>过滤掉 score 低于 {@code minScore} 的低置信度结果</li>
     *     <li>只保留前 {@code topN} 个高分结果</li>
     * </ol>
     * <p>
     * 默认实现基于流式过滤，子类可覆盖以提供更高效的实现。
     *
     * @param question 用户问题
     * @param topN     最多返回的结果数量
     * @param minScore 最低分数阈值，低于此值的结果将被过滤
     * @return 过滤后的 {@link NodeScore} 列表，按 score 降序排列
     */
    default List<NodeScore> topKAboveThreshold(String question, int topN, double minScore) {
        return classifyTargets(question).stream()
                .filter(ns -> ns.getScore() >= minScore)
                .limit(topN)
                .toList();
    }
}
