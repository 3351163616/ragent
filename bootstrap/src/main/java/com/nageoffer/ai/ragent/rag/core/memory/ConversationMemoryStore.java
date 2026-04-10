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
 * 对话消息持久化存储接口 —— 定义对话历史消息的底层存储操作。
 * <p>
 * 该接口是 {@link ConversationMemoryService} 的下层依赖，负责消息的实际读写操作。
 * 与 {@link ConversationMemoryService} 的区别在于：
 * <ul>
 *   <li>{@code ConversationMemoryService} 是上层编排接口，负责摘要+历史的并行加载与合并逻辑</li>
 *   <li>{@code ConversationMemoryStore} 是底层存储接口，只关注消息本身的持久化和读取</li>
 * </ul>
 * <p>
 * 当前实现：{@link JdbcConversationMemoryStore}（基于 JDBC 直读模式）。
 * 可扩展为 Redis 缓存实现等其他存储后端。
 *
 * @see JdbcConversationMemoryStore
 */
public interface ConversationMemoryStore {

    /**
     * 从存储中加载指定会话的历史消息。
     * <p>
     * 实现类需保证返回的消息按时间正序排列（最早的在前），
     * 并且只包含 USER 和 ASSISTANT 角色的消息（不含 SYSTEM 消息）。
     * 历史消息的数量由配置的保留轮次数决定。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @return 时间正序排列的历史消息列表，无历史时返回空列表
     */
    List<ChatMessage> loadHistory(String conversationId, String userId);

    /**
     * 将一条消息追加到对话历史存储中。
     * <p>
     * 若消息角色为 USER，还需同步更新或创建会话记录（conversation 表）。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @param message        要追加的消息（包含角色和内容）
     * @return 持久化后的消息 ID（雪花算法生成），可用于后续引用
     */
    String append(String conversationId, String userId, ChatMessage message);

    /**
     * 刷新指定会话的缓存。
     * <p>
     * 在使用缓存层（如 Redis）的实现中，当底层数据发生变更时调用此方法使缓存失效。
     * 在 JDBC 直读模式的实现中，此方法为空操作（no-op）。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     */
    void refreshCache(String conversationId, String userId);
}
