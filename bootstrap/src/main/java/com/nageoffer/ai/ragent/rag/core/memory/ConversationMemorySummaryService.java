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

package com.nageoffer.ai.ragent.rag.core.memory;

import com.nageoffer.ai.ragent.framework.convention.ChatMessage;

/**
 * 对话摘要服务接口 —— 负责对话历史的摘要生成、加载和装饰。
 * <p>
 * 在 RAG 管线的"记忆加载"阶段，摘要服务与消息存储协同工作，实现"摘要 + 滑动窗口"的记忆策略：
 * <ul>
 *   <li>滑动窗口保留最近 N 轮的完整消息（由 {@link ConversationMemoryStore} 管理）</li>
 *   <li>摘要压缩更早期的对话内容为简短文本（由本接口管理）</li>
 * </ul>
 * <p>
 * 摘要的生命周期：
 * <ol>
 *   <li><b>触发压缩</b>：每次追加 ASSISTANT 消息后，通过 {@link #compressIfNeeded} 异步检查是否需要生成/更新摘要</li>
 *   <li><b>加载摘要</b>：在 load 阶段通过 {@link #loadLatestSummary} 获取最新摘要</li>
 *   <li><b>装饰摘要</b>：通过 {@link #decorateIfNeeded} 为摘要添加前缀标识，便于 LLM 区分摘要与普通消息</li>
 * </ol>
 *
 * @see JdbcConversationMemorySummaryService
 */
public interface ConversationMemorySummaryService {

    /**
     * 检查是否需要压缩对话历史并生成/更新摘要。
     * <p>
     * 触发条件：
     * <ul>
     *   <li>摘要功能已启用（配置开关）</li>
     *   <li>当前消息为 ASSISTANT 角色（确保一轮对话完整后才触发）</li>
     *   <li>会话中的用户消息总数达到配置的触发阈值</li>
     * </ul>
     * <p>
     * 该方法为异步执行，不会阻塞当前请求。使用 Redisson 分布式锁防止并发压缩。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @param message        刚追加的消息，用于判断角色（仅 ASSISTANT 触发）
     */
    void compressIfNeeded(String conversationId, String userId, ChatMessage message);

    /**
     * 加载指定会话的最新摘要。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @return 摘要消息（SYSTEM 角色），若无摘要则返回 null
     */
    ChatMessage loadLatestSummary(String conversationId, String userId);

    /**
     * 为摘要消息添加装饰前缀（如"对话摘要："）。
     * <p>
     * 添加前缀的目的是让 LLM 明确知道这是一段压缩后的历史摘要而非普通系统指令，
     * 避免 LLM 将摘要内容误当作操作指令执行。
     * 若摘要已有前缀则不重复添加。
     *
     * @param summary 原始摘要消息
     * @return 装饰后的摘要消息，若输入为 null 或空则原样返回
     */
    ChatMessage decorateIfNeeded(ChatMessage summary);
}
