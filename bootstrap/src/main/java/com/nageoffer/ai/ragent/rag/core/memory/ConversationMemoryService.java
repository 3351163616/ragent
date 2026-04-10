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

import java.util.List;

/**
 * 对话记忆服务接口 —— RAG 对话管线中"记忆加载"阶段的核心抽象。
 * <p>
 * 该接口负责管理对话历史的加载和追加，是 RAG 管线的第一个阶段。
 * 在 {@code RAGChatServiceImpl.streamChat()} 中被调用，为后续的查询改写和意图识别提供对话上下文。
 * <p>
 * 设计要点：
 * <ul>
 *   <li>{@link #load} 和 {@link #append} 是两个基本操作，由实现类负责具体的存储和检索逻辑</li>
 *   <li>{@link #loadAndAppend} 是一个 default 方法，采用<b>模板方法模式</b>组合了 load 和 append 两个操作，
 *       提供了一个便捷的原子语义：先加载历史再追加新消息。子类可以按需重写以实现更高效的实现</li>
 * </ul>
 *
 * @see DefaultConversationMemoryService
 */
public interface ConversationMemoryService {

    /**
     * 加载指定会话的对话历史记录。
     * <p>
     * 返回的列表中可能包含以下内容（按顺序）：
     * <ol>
     *   <li>对话摘要（如果存在，以 SYSTEM 消息形式出现在列表开头）</li>
     *   <li>最近 N 轮的历史消息（USER 和 ASSISTANT 交替）</li>
     * </ol>
     *
     * @param conversationId 会话 ID，标识一个独立的对话会话
     * @param userId         用户 ID，用于隔离不同用户的对话数据
     * @return 对话历史消息列表（包含摘要和历史记录），若无历史则返回空列表
     */
    List<ChatMessage> load(String conversationId, String userId);

    /**
     * 追加一条消息到对话历史存储。
     * <p>
     * 持久化消息后，若消息为 ASSISTANT 类型，还会异步触发摘要压缩检查。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @param message        要追加的消息（USER 或 ASSISTANT 角色）
     * @return 持久化后的消息 ID（雪花算法生成）
     */
    String append(String conversationId, String userId, ChatMessage message);

    /**
     * 加载历史记录并追加新消息的便捷方法（default 实现）。
     * <p>
     * 该方法组合了 {@link #load} 和 {@link #append} 两个操作，语义为：
     * "先获取当前已有的历史记录，再将新消息追加到存储中"。
     * <p>
     * 返回的是追加<b>之前</b>的历史记录（不含刚追加的消息），
     * 这样调用方可以将历史记录与当前用户消息分开处理（如改写时需要历史做指代消解）。
     * <p>
     * 使用 default 方法的设计意图：
     * <ul>
     *   <li>提供开箱即用的默认实现，避免每个实现类都重复编写组合逻辑</li>
     *   <li>子类可以重写此方法以实现批量操作优化（如减少 DB 查询次数）</li>
     * </ul>
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @param message        要追加的消息
     * @return 追加前的历史记录列表（不含刚追加的消息）
     */
    default List<ChatMessage> loadAndAppend(String conversationId, String userId, ChatMessage message) {
        List<ChatMessage> history = load(conversationId, userId);
        append(conversationId, userId, message);
        return history;
    }
}
