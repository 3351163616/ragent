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

import com.nageoffer.ai.ragent.framework.convention.ChatMessage;

import java.util.List;

/**
 * 查询改写服务接口 —— RAG 管线的第一阶段。
 * <p>
 * 职责：将用户输入的自然语言问题改写为更适合向量检索 / 关键字检索的查询语句，
 * 同时支持将复合问句拆分为多个独立子问题，以便后续意图识别和多通道检索各自命中精准上下文。
 * <p>
 * 设计要点：
 * <ul>
 *     <li>提供三层递进的能力：纯改写 {@link #rewrite} → 改写+拆分 {@link #rewriteWithSplit(String)}
 *         → 带会话历史的改写+拆分 {@link #rewriteWithSplit(String, List)}。</li>
 *     <li>默认实现通过 {@code default} 方法逐层委托，子类只需覆盖最顶层方法即可获得全套能力。</li>
 * </ul>
 *
 * @see MultiQuestionRewriteService 基于 LLM 的多问题改写实现
 * @see RewriteResult 改写 + 拆分的统一返回结构
 */
public interface QueryRewriteService {

    /**
     * 将用户问题改写为适合向量 / 关键字检索的简洁查询。
     * <p>
     * 这是最基础的改写能力，不做多问句拆分。如果改写失败，实现方应回退至原始问题。
     *
     * @param userQuestion 原始用户问题（未经任何预处理的自然语言文本）
     * @return 改写后的检索查询；若改写失败则返回原始问题作为兜底
     */
    String rewrite(String userQuestion);

    /**
     * 改写 + 拆分多问句（无会话历史版本）。
     * <p>
     * 默认实现：调用 {@link #rewrite(String)} 获取改写结果，然后将其作为唯一子问题返回。
     * 子类（如 {@link MultiQuestionRewriteService}）可覆盖此方法以支持真正的多问句拆分。
     *
     * @param userQuestion 原始用户问题
     * @return {@link RewriteResult} 包含改写后的总查询和拆分后的子问题列表
     */
    default RewriteResult rewriteWithSplit(String userQuestion) {
        String rewritten = rewrite(userQuestion);
        return new RewriteResult(rewritten, List.of(rewritten));
    }

    /**
     * 改写 + 拆分多问句，支持会话历史（用于指代消解）。
     * <p>
     * 在多轮对话场景中，用户问题可能存在代词引用（如"它"、"这个系统"），
     * 需要结合最近几轮历史消息进行指代消解后再改写。
     * <p>
     * 默认实现忽略历史，直接委托给 {@link #rewriteWithSplit(String)}。
     *
     * @param userQuestion 原始用户问题
     * @param history      最近的会话历史消息列表（包含 User 和 Assistant 角色的消息）
     * @return {@link RewriteResult} 包含改写后的总查询和拆分后的子问题列表
     */
    default RewriteResult rewriteWithSplit(String userQuestion, List<ChatMessage> history) {
        return rewriteWithSplit(userQuestion);
    }
}
