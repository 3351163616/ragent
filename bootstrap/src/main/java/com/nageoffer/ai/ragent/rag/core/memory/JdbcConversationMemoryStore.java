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

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.rag.config.MemoryProperties;
import com.nageoffer.ai.ragent.rag.controller.request.ConversationCreateRequest;
import com.nageoffer.ai.ragent.rag.controller.vo.ConversationMessageVO;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.rag.enums.ConversationMessageOrder;
import com.nageoffer.ai.ragent.rag.service.ConversationMessageService;
import com.nageoffer.ai.ragent.rag.service.ConversationService;
import com.nageoffer.ai.ragent.rag.service.bo.ConversationMessageBO;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 基于 JDBC 的对话消息持久化存储实现。
 * <p>
 * 该实现采用<b>直读模式</b>（无缓存层），每次加载历史时直接查询数据库。
 * <p>
 * 核心设计要点：
 * <ul>
 *   <li><b>DESC + reverse 取最新 N 条</b>：{@link #loadHistory} 先以 DESC 排序取最新的 N 条消息，
 *       再通过 stream 处理转为时间正序。这样比 ASC + OFFSET 效率更高，
 *       因为数据库可以利用索引快速定位最新记录而无需跳过大量旧记录</li>
 *   <li><b>历史消息规范化</b>：{@link #normalizeHistory} 确保返回的消息列表以 USER 消息开头，
 *       避免上下文中出现孤立的 ASSISTANT 消息（没有对应的 USER 问题），
 *       这对 LLM 的上下文理解至关重要</li>
 *   <li><b>轮次到消息数的转换</b>：配置的是保留轮次数（turns），每轮包含一问一答两条消息，
 *       因此实际查询条数 = 轮次数 * 2</li>
 *   <li><b>会话自动创建</b>：追加 USER 消息时自动创建或更新会话记录</li>
 * </ul>
 *
 * @see ConversationMemoryStore
 */
@Slf4j
@Service
public class JdbcConversationMemoryStore implements ConversationMemoryStore {

    /** 会话管理服务，负责会话记录的创建和更新 */
    private final ConversationService conversationService;

    /** 会话消息服务，负责消息的 CRUD 操作 */
    private final ConversationMessageService conversationMessageService;

    /** 记忆相关配置属性，包含历史保留轮次数等参数 */
    private final MemoryProperties memoryProperties;

    /**
     * 构造方法，注入所需的服务和配置。
     *
     * @param conversationService        会话管理服务
     * @param conversationMessageService 会话消息服务
     * @param memoryProperties           记忆配置属性
     */

    public JdbcConversationMemoryStore(ConversationService conversationService,
                                       ConversationMessageService conversationMessageService,
                                       MemoryProperties memoryProperties) {
        this.conversationService = conversationService;
        this.conversationMessageService = conversationMessageService;
        this.memoryProperties = memoryProperties;
    }

    /**
     * 从数据库加载最近 N 轮的对话历史消息。
     * <p>
     * 加载策略：
     * <ol>
     *   <li>按 DESC 排序查询最新的 maxMessages 条消息（利用索引快速定位）</li>
     *   <li>将数据库记录转换为 {@link ChatMessage} 对象</li>
     *   <li>过滤无效消息（null、空内容、非 USER/ASSISTANT 角色）</li>
     *   <li>通过 {@link #normalizeHistory} 规范化：确保列表以 USER 消息开头</li>
     * </ol>
     * <p>
     * 注意：{@code ConversationMessageOrder.DESC} 在查询层面保证按时间倒序返回，
     * 但 stream 的 map/filter 操作会保持原始顺序，最终由 normalizeHistory 处理正序问题。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @return 时间正序排列的历史消息列表，以 USER 消息开头
     */
    @Override
    public List<ChatMessage> loadHistory(String conversationId, String userId) {
        // 将配置的轮次数转换为消息条数（每轮 = 1条USER + 1条ASSISTANT）
        int maxMessages = resolveMaxHistoryMessages();
        // 按 DESC 排序查询，数据库可利用 (conversation_id, id DESC) 索引快速取最新记录
        List<ConversationMessageVO> dbMessages = conversationMessageService.listMessages(
                conversationId,
                userId,
                maxMessages,
                ConversationMessageOrder.DESC
        );
        if (CollUtil.isEmpty(dbMessages)) {
            return List.of();
        }

        // 转换并过滤：DB 记录 → ChatMessage，移除无效消息
        List<ChatMessage> result = dbMessages.stream()
                .map(this::toChatMessage)
                .filter(this::isHistoryMessage)
                .collect(Collectors.toList());

        // 规范化：确保以 USER 消息开头，截掉开头可能存在的孤立 ASSISTANT 消息
        return normalizeHistory(result);
    }

    /**
     * 将消息追加到数据库，并在必要时创建/更新会话记录。
     * <p>
     * 处理逻辑：
     * <ol>
     *   <li>构建消息 BO 并通过 {@link ConversationMessageService} 持久化</li>
     *   <li>若消息角色为 USER，额外调用 {@link ConversationService#createOrUpdate}
     *       创建新会话或更新会话的最后活跃时间和最新问题</li>
     * </ol>
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @param message        要追加的消息
     * @return 持久化后的消息 ID
     */

    @Override
    public String append(String conversationId, String userId, ChatMessage message) {
        ConversationMessageBO conversationMessage = ConversationMessageBO.builder()
                .conversationId(conversationId)
                .userId(userId)
                .role(message.getRole().name().toLowerCase())
                .content(message.getContent())
                .build();
        String messageId = conversationMessageService.addMessage(conversationMessage);

        // 仅 USER 消息触发会话记录的创建/更新，ASSISTANT 消息不影响会话元数据
        if (message.getRole() == ChatMessage.Role.USER) {
            ConversationCreateRequest conversation = ConversationCreateRequest.builder()
                    .conversationId(conversationId)
                    .userId(userId)
                    .question(message.getContent())
                    .lastTime(new Date())
                    .build();
            conversationService.createOrUpdate(conversation);
        }
        return messageId;
    }

    /**
     * JDBC 直读模式无需缓存刷新，此方法为空操作。
     */
    @Override
    public void refreshCache(String conversationId, String userId) {
        // JDBC 直读模式，无需刷新缓存
    }

    /**
     * 将数据库消息记录转换为 {@link ChatMessage} 对象。
     *
     * @param record 数据库消息记录 VO
     * @return 对应的 ChatMessage，记录无效时返回 null
     */

    private ChatMessage toChatMessage(ConversationMessageVO record) {
        if (record == null || StrUtil.isBlank(record.getContent())) {
            return null;
        }
        ChatMessage.Role role = ChatMessage.Role.fromString(record.getRole());
        return new ChatMessage(role, record.getContent());
    }

    /**
     * 规范化历史消息列表，确保以 USER 消息开头。
     * <p>
     * 为什么需要这样做：LLM 的对话上下文要求 USER/ASSISTANT 消息交替出现，
     * 若列表以 ASSISTANT 消息开头，LLM 会将其视为"无问题的回答"，导致上下文混乱。
     * 当历史消息因 DESC 排序截断或消息丢失导致开头是 ASSISTANT 消息时，
     * 此方法会跳过这些孤立的 ASSISTANT 消息，直到找到第一条 USER 消息。
     *
     * @param messages 待规范化的消息列表
     * @return 以 USER 消息开头的规范化消息列表
     */
    private List<ChatMessage> normalizeHistory(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty()) {
            return List.of();
        }
        // 先过滤掉无效消息
        List<ChatMessage> cleaned = messages.stream()
                .filter(this::isHistoryMessage)
                .toList();
        if (cleaned.isEmpty()) {
            return List.of();
        }
        // 从头部跳过连续的 ASSISTANT 消息，找到第一条 USER 消息的位置
        int start = 0;
        while (start < cleaned.size() && cleaned.get(start).getRole() == ChatMessage.Role.ASSISTANT) {
            start++;
        }
        // 若全部都是 ASSISTANT 消息（极端情况），返回空列表
        if (start >= cleaned.size()) {
            return List.of();
        }
        return cleaned.subList(start, cleaned.size());
    }

    /**
     * 判断消息是否为有效的历史消息。
     * <p>
     * 有效条件：消息不为 null、角色为 USER 或 ASSISTANT、内容不为空。
     * SYSTEM 角色的消息（如摘要）不算历史消息，由摘要服务单独管理。
     *
     * @param message 待检查的消息
     * @return 是否为有效历史消息
     */
    private boolean isHistoryMessage(ChatMessage message) {
        return message != null
                && (message.getRole() == ChatMessage.Role.USER || message.getRole() == ChatMessage.Role.ASSISTANT)
                && StrUtil.isNotBlank(message.getContent());
    }

    /**
     * 将配置的历史保留轮次数转换为实际的消息查询条数。
     * <p>
     * 每一轮对话包含一条 USER 消息和一条 ASSISTANT 消息，因此消息条数 = 轮次数 * 2。
     *
     * @return 需要查询的最大消息条数
     */
    private int resolveMaxHistoryMessages() {
        int maxTurns = memoryProperties.getHistoryKeepTurns();
        return maxTurns * 2;
    }
}
