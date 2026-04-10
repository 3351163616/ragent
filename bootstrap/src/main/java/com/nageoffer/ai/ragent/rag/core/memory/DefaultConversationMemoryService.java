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
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * 对话记忆服务的默认实现，负责编排对话历史和摘要的加载与合并。
 * <p>
 * 该类在 RAG 管线的"记忆加载"阶段被调用，是 {@link ConversationMemoryService} 接口的主要实现。
 * <p>
 * 核心设计要点：
 * <ul>
 *   <li><b>并行加载</b>：{@link #load} 方法使用 {@link CompletableFuture} 并行加载摘要和历史记录，
 *       减少总加载时间（摘要加载和历史加载互不依赖）</li>
 *   <li><b>降级容错</b>：摘要加载失败时返回 null（跳过摘要），历史加载失败时返回空列表，
 *       确保记忆加载阶段不会因部分失败而阻塞整个管线</li>
 *   <li><b>摘要合并</b>：{@link #attachSummary} 将摘要以 SYSTEM 消息形式插入历史消息列表头部，
 *       使 LLM 能够感知更早期的对话上下文</li>
 *   <li><b>异步摘要压缩</b>：在 {@link #append} 中追加消息后，异步触发摘要压缩检查</li>
 * </ul>
 *
 * @see ConversationMemoryStore 消息持久化存储
 * @see ConversationMemorySummaryService 摘要生成与加载
 */
@Slf4j
@Service
public class DefaultConversationMemoryService implements ConversationMemoryService {

    /** 消息持久化存储，负责历史消息的读写 */
    private final ConversationMemoryStore memoryStore;

    /** 摘要服务，负责摘要的生成、加载和装饰 */
    private final ConversationMemorySummaryService summaryService;

    /**
     * 构造方法，注入消息存储和摘要服务。
     *
     * @param memoryStore    消息持久化存储实现
     * @param summaryService 摘要服务实现
     */
    public DefaultConversationMemoryService(ConversationMemoryStore memoryStore,
                                            ConversationMemorySummaryService summaryService) {
        this.memoryStore = memoryStore;
        this.summaryService = summaryService;
    }

    /**
     * 并行加载对话摘要和历史消息，合并后返回完整的对话上下文。
     * <p>
     * 执行流程：
     * <ol>
     *   <li>参数校验：会话 ID 或用户 ID 为空时直接返回空列表</li>
     *   <li>并行加载：使用 {@link CompletableFuture} 同时发起摘要加载和历史加载，
     *       两者互不依赖，可并行执行以减少总耗时</li>
     *   <li>降级处理：摘要加载失败返回 null（跳过摘要），历史加载失败返回空列表</li>
     *   <li>结果合并：通过 {@link #attachSummary} 将摘要插入历史消息列表头部</li>
     * </ol>
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @return 合并后的对话上下文列表（摘要在前 + 历史消息在后），加载失败时返回空列表
     */
    @Override
    public List<ChatMessage> load(String conversationId, String userId) {
        // 参数校验：避免无效的数据库查询
        if (StrUtil.isBlank(conversationId) || StrUtil.isBlank(userId)) {
            return List.of();
        }

        long startTime = System.currentTimeMillis();
        try {
            // 并行加载摘要和历史记录：两个查询互不依赖，并行执行可将总耗时从 T1+T2 降至 max(T1,T2)
            CompletableFuture<ChatMessage> summaryFuture = CompletableFuture.supplyAsync(
                    () -> loadSummaryWithFallback(conversationId, userId)
            );
            CompletableFuture<List<ChatMessage>> historyFuture = CompletableFuture.supplyAsync(
                    () -> loadHistoryWithFallback(conversationId, userId)
            );

            // 等待两个并行任务都完成后，合并摘要与历史记录
            return CompletableFuture.allOf(summaryFuture, historyFuture)
                    .thenApply(v -> {
                        ChatMessage summary = summaryFuture.join();
                        List<ChatMessage> history = historyFuture.join();
                        log.debug("加载对话记忆 - conversationId: {}, userId: {}, 摘要: {}, 历史消息数: {}, 耗时: {}ms",
                                conversationId, userId, summary != null, history.size(), System.currentTimeMillis() - startTime);
                        return attachSummary(summary, history);
                    })
                    .join();
        } catch (Exception e) {
            log.error("加载对话记忆失败 - conversationId: {}, userId: {}", conversationId, userId, e);
            return List.of();
        }
    }

    /**
     * 加载摘要的降级方法 —— 加载失败时返回 null 而非抛出异常。
     * <p>
     * 摘要是可选的辅助信息，即使加载失败也不应阻塞整个记忆加载流程。
     * 返回 null 后，{@link #attachSummary} 会跳过摘要合并。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @return 摘要消息，加载失败时返回 null
     */
    private ChatMessage loadSummaryWithFallback(String conversationId, String userId) {
        try {
            return summaryService.loadLatestSummary(conversationId, userId);
        } catch (Exception e) {
            log.warn("加载摘要失败，将跳过摘要 - conversationId: {}, userId: {}", conversationId, userId, e);
            return null;
        }
    }

    /**
     * 加载历史记录的降级方法 —— 加载失败时返回空列表而非抛出异常。
     * <p>
     * 历史记录加载失败时，管线仍可继续运行（仅丧失上下文连续性），
     * 不会导致整个对话请求失败。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @return 历史消息列表，加载失败时返回空列表
     */
    private List<ChatMessage> loadHistoryWithFallback(String conversationId, String userId) {
        try {
            List<ChatMessage> history = memoryStore.loadHistory(conversationId, userId);
            return history != null ? history : List.of();
        } catch (Exception e) {
            log.error("加载历史记录失败 - conversationId: {}, userId: {}", conversationId, userId, e);
            return List.of();
        }
    }

    /**
     * 追加消息到对话历史存储，并在必要时触发异步摘要压缩。
     * <p>
     * 执行流程：
     * <ol>
     *   <li>参数校验</li>
     *   <li>通过 {@link ConversationMemoryStore#append} 持久化消息</li>
     *   <li>调用 {@link ConversationMemorySummaryService#compressIfNeeded} 检查是否需要触发摘要压缩
     *       （仅 ASSISTANT 消息会实际触发，USER 消息会被跳过）</li>
     * </ol>
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @param message        要追加的消息
     * @return 持久化后的消息 ID，参数无效时返回 null
     */
    @Override
    public String append(String conversationId, String userId, ChatMessage message) {
        if (StrUtil.isBlank(conversationId) || StrUtil.isBlank(userId)) {
            return null;
        }
        String messageId = memoryStore.append(conversationId, userId, message);
        // 追加消息后异步检查是否需要生成/更新摘要（仅 ASSISTANT 消息触发）
        summaryService.compressIfNeeded(conversationId, userId, message);
        return messageId;
    }

    /**
     * 将摘要合并到历史消息列表的头部。
     * <p>
     * 合并策略：
     * <ul>
     *   <li>若历史消息为空，直接返回空列表（即使有摘要也没有意义）</li>
     *   <li>若摘要为 null，直接返回原始历史消息列表</li>
     *   <li>若摘要存在，先通过 {@link ConversationMemorySummaryService#decorateIfNeeded} 装饰摘要
     *       （添加"对话摘要："前缀），然后将其作为 SYSTEM 消息插入列表头部</li>
     * </ul>
     * <p>
     * 这样做的目的是让 LLM 在接收到消息列表时，首先看到更早期的对话摘要作为背景知识，
     * 然后再看到近期的具体历史消息，从而实现"摘要 + 滑动窗口"的记忆策略。
     *
     * @param summary  摘要消息（SYSTEM 角色），可能为 null
     * @param messages 历史消息列表
     * @return 合并后的消息列表（摘要在前 + 历史在后）
     */
    private List<ChatMessage> attachSummary(ChatMessage summary, List<ChatMessage> messages) {
        // 确保返回值不为 null：历史为空时无需合并
        if (CollUtil.isEmpty(messages)) {
            return List.of();
        }
        if (summary == null) {
            return messages;
        }
        List<ChatMessage> result = new ArrayList<>();
        // 将装饰后的摘要插入列表头部，作为 LLM 的背景知识
        result.add(summaryService.decorateIfNeeded(summary));
        result.addAll(messages);
        return result;
    }
}
