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
import com.nageoffer.ai.ragent.rag.dao.entity.ConversationMessageDO;
import com.nageoffer.ai.ragent.rag.dao.entity.ConversationSummaryDO;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.infra.chat.LLMService;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptTemplateLoader;
import com.nageoffer.ai.ragent.rag.service.ConversationGroupService;
import com.nageoffer.ai.ragent.rag.service.ConversationMessageService;
import com.nageoffer.ai.ragent.rag.service.bo.ConversationSummaryBO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.CONVERSATION_SUMMARY_PROMPT_PATH;

/**
 * 基于 JDBC 的对话摘要服务实现。
 * <p>
 * 该类负责对话历史的增量摘要压缩，是"摘要 + 滑动窗口"记忆策略的核心实现。
 * <p>
 * 核心设计要点：
 * <ul>
 *   <li><b>异步触发</b>：{@link #compressIfNeeded} 通过专用线程池异步执行摘要任务，
 *       不阻塞主对话流程</li>
 *   <li><b>Redisson 分布式锁</b>：使用 {@code ragent:memory:summary:lock:{userId}:{conversationId}} 作为锁 Key，
 *       防止同一会话的多个摘要任务并发执行（如用户快速连续发送消息时）</li>
 *   <li><b>增量摘要合并策略</b>：不是每次都从头压缩所有历史，而是：
 *     <ol>
 *       <li>加载上一次摘要的内容</li>
 *       <li>查找上次摘要覆盖的最后一条消息 ID 到当前滑动窗口起点之间的新增消息</li>
 *       <li>将旧摘要和新增消息一起交给 LLM 合并压缩，生成更新后的摘要</li>
 *     </ol>
 *   </li>
 *   <li><b>摘要边界计算</b>：通过 cutoffId（滑动窗口最早消息的 ID）和 afterId（上次摘要覆盖到的消息 ID）
 *       精确确定需要压缩的消息范围，避免重复压缩</li>
 *   <li><b>LLM 摘要生成</b>：使用低温度（0.3）的 LLM 调用，指令要求合并去重并控制字符数上限</li>
 * </ul>
 *
 * @see ConversationMemorySummaryService
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class JdbcConversationMemorySummaryService implements ConversationMemorySummaryService {

    /** 摘要消息的前缀标识，用于让 LLM 区分摘要与普通系统消息 */
    private static final String SUMMARY_PREFIX = "对话摘要：";

    /** Redisson 分布式锁的 Key 前缀 */
    private static final String SUMMARY_LOCK_PREFIX = "ragent:memory:summary:lock:";

    /** 分布式锁的最大持有时间（TTL），防止锁泄漏 */
    private static final Duration SUMMARY_LOCK_TTL = Duration.ofMinutes(5);

    /** 会话分组服务，提供消息计数、摘要查询和消息范围查询等操作 */
    private final ConversationGroupService conversationGroupService;

    /** 会话消息服务，负责摘要记录的持久化 */
    private final ConversationMessageService conversationMessageService;

    /** 记忆相关配置属性（摘要开关、触发阈值、历史保留轮次、摘要最大字符数等） */
    private final MemoryProperties memoryProperties;

    /** LLM 服务，用于调用大语言模型生成摘要文本 */
    private final LLMService llmService;

    /** Prompt 模板加载器，加载摘要生成用的 Prompt 模板 */
    private final PromptTemplateLoader promptTemplateLoader;

    /** Redisson 客户端，用于获取分布式锁 */
    private final RedissonClient redissonClient;

    /** 摘要专用线程池，避免摘要任务占用主业务线程 */
    @Qualifier("memorySummaryThreadPoolExecutor")
    private final Executor memorySummaryExecutor;

    /**
     * 检查是否需要压缩对话历史并异步触发摘要生成。
     * <p>
     * 前置过滤：
     * <ul>
     *   <li>摘要功能未启用时直接返回</li>
     *   <li>仅 ASSISTANT 消息触发 —— 确保一轮对话（USER + ASSISTANT）完整后才触发，
     *       避免在 USER 消息刚追加时就压缩（此时 ASSISTANT 回复尚未生成）</li>
     * </ul>
     * <p>
     * 异步执行：通过专用线程池 {@code memorySummaryExecutor} 提交任务，
     * 异常通过 {@code exceptionally} 捕获并记录日志，不影响主流程。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @param message        刚追加的消息
     */
    @Override
    public void compressIfNeeded(String conversationId, String userId, ChatMessage message) {
        // 摘要功能开关检查
        if (!memoryProperties.getSummaryEnabled()) {
            return;
        }
        // 仅在 ASSISTANT 回复完成后触发，确保一轮对话完整
        if (message.getRole() != ChatMessage.Role.ASSISTANT) {
            return;
        }
        // 异步提交摘要任务到专用线程池，不阻塞当前对话响应
        CompletableFuture.runAsync(() -> doCompressIfNeeded(conversationId, userId), memorySummaryExecutor)
                .exceptionally(ex -> {
                    log.error("对话记忆摘要异步任务失败 - conversationId: {}, userId: {}",
                            conversationId, userId, ex);
                    return null;
                });
    }

    /**
     * 加载指定会话的最新摘要记录。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @return 摘要消息（SYSTEM 角色），若无摘要记录则返回 null
     */
    @Override
    public ChatMessage loadLatestSummary(String conversationId, String userId) {
        ConversationSummaryDO summary = conversationGroupService.findLatestSummary(conversationId, userId);
        return toChatMessage(summary);
    }

    /**
     * 为摘要消息添加装饰前缀。
     * <p>
     * 若摘要内容已以"对话摘要："或"摘要："开头，则不重复添加前缀。
     * 装饰后的摘要以 SYSTEM 角色返回，便于在消息列表中与其他 SYSTEM 消息区分。
     *
     * @param summary 原始摘要消息
     * @return 添加前缀后的摘要消息，输入为 null 或空内容时原样返回
     */
    @Override
    public ChatMessage decorateIfNeeded(ChatMessage summary) {
        if (summary == null || StrUtil.isBlank(summary.getContent())) {
            return summary;
        }

        String content = summary.getContent().trim();
        // 幂等检查：若已有前缀则不重复添加
        if (content.startsWith(SUMMARY_PREFIX) || content.startsWith("摘要：")) {
            return summary;
        }
        return ChatMessage.system(SUMMARY_PREFIX + content);
    }

    /**
     * 摘要压缩的核心执行逻辑（在异步线程中运行）。
     * <p>
     * 完整流程：
     * <ol>
     *   <li><b>配置检查</b>：获取触发阈值（triggerTurns）和历史保留轮次（maxTurns）</li>
     *   <li><b>获取分布式锁</b>：使用 Redisson 的 tryLock（非阻塞），防止同一会话的并发摘要</li>
     *   <li><b>消息计数检查</b>：统计会话中的 USER 消息总数，未达阈值则跳过</li>
     *   <li><b>确定压缩边界</b>：
     *     <ul>
     *       <li>cutoffId：滑动窗口中最早的 USER 消息 ID（窗口之前的消息需要被压缩）</li>
     *       <li>afterId：上次摘要覆盖到的最后一条消息 ID（避免重复压缩）</li>
     *     </ul>
     *   </li>
     *   <li><b>查询待压缩消息</b>：获取 afterId 到 cutoffId 之间的所有消息</li>
     *   <li><b>调用 LLM 生成摘要</b>：将旧摘要和新增消息合并，通过 LLM 去重压缩</li>
     *   <li><b>持久化摘要</b>：保存新摘要并记录其覆盖到的最后消息 ID</li>
     * </ol>
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     */
    private void doCompressIfNeeded(String conversationId, String userId) {
        long startTime = System.currentTimeMillis();
        // 获取配置：触发摘要的最小 USER 消息数、历史保留的轮次数
        int triggerTurns = memoryProperties.getSummaryStartTurns();
        int maxTurns = memoryProperties.getHistoryKeepTurns();
        if (maxTurns <= 0 || triggerTurns <= 0) {
            return;
        }

        // 获取分布式锁：防止同一会话的多个摘要任务并发执行
        // 使用 tryLock(0, TTL) 非阻塞模式，获取失败则直接放弃（下次 ASSISTANT 消息时会重试）
        String lockKey = SUMMARY_LOCK_PREFIX + buildLockKey(conversationId, userId);
        RLock lock = redissonClient.getLock(lockKey);
        if (!tryLock(lock)) {
            return;
        }
        try {
            // 统计该会话中的 USER 消息总数，未达触发阈值则跳过
            long total = conversationGroupService.countUserMessages(conversationId, userId);
            if (total < triggerTurns) {
                return;
            }

            // 加载上次的摘要记录，用于增量合并
            ConversationSummaryDO latestSummary = conversationGroupService.findLatestSummary(conversationId, userId);
            // 获取滑动窗口中保留的最新 maxTurns 条 USER 消息（用于确定窗口起始位置）
            List<ConversationMessageDO> latestUserTurns = conversationGroupService.listLatestUserOnlyMessages(
                    conversationId,
                    userId,
                    maxTurns
            );
            if (latestUserTurns.isEmpty()) {
                return;
            }
            // cutoffId：滑动窗口中最早的 USER 消息 ID，此 ID 之前的消息需要被压缩成摘要
            String cutoffId = resolveCutoffId(latestUserTurns);
            if (StrUtil.isBlank(cutoffId)) {
                return;
            }

            // afterId：上次摘要覆盖到的最后一条消息 ID，避免重复压缩已摘要过的消息
            String afterId = resolveSummaryStartId(conversationId, userId, latestSummary);
            // 若上次摘要已覆盖到 cutoff 位置或更新的位置，说明无新增消息需要压缩
            if (afterId != null && Long.parseLong(afterId) >= Long.parseLong(cutoffId)) {
                return;
            }

            // 查询 afterId 到 cutoffId 之间的所有消息，这些就是需要被压缩的新增消息
            List<ConversationMessageDO> toSummarize = conversationGroupService.listMessagesBetweenIds(
                    conversationId,
                    userId,
                    afterId,
                    cutoffId
            );
            if (CollUtil.isEmpty(toSummarize)) {
                return;
            }

            // 记录待压缩消息中的最后一条 ID，作为本次摘要覆盖的边界
            String lastMessageId = resolveLastMessageId(toSummarize);
            if (StrUtil.isBlank(lastMessageId)) {
                return;
            }

            // 增量摘要合并：将旧摘要内容和新增消息一起交给 LLM 合并压缩
            String existingSummary = latestSummary == null ? "" : latestSummary.getContent();
            String summary = summarizeMessages(toSummarize, existingSummary);
            if (StrUtil.isBlank(summary)) {
                return;
            }

            // 持久化新摘要，记录覆盖到的最后消息 ID（供下次增量压缩使用）
            createSummary(conversationId, userId, summary, lastMessageId);
            log.info("摘要成功 - conversationId：{}，userId：{}，消息数：{}，耗时：{}ms",
                    conversationId, userId, toSummarize.size(),
                    System.currentTimeMillis() - startTime);
        } catch (Exception e) {
            log.error("摘要失败 - conversationId：{}，userId：{}", conversationId, userId, e);
        } finally {
            // 安全释放锁：仅当当前线程持有锁时才释放，避免误释放其他线程的锁
            if (lock.isHeldByCurrentThread()) {
                lock.unlock();
            }
        }
    }

    /**
     * 尝试获取分布式锁（非阻塞模式）。
     * <p>
     * 使用 waitTime=0 表示不等待，若锁已被其他线程/实例持有则立即返回 false。
     * leaseTime 设置为 {@link #SUMMARY_LOCK_TTL}（5分钟），作为锁的最大持有时间，
     * 防止因异常导致锁泄漏（deadlock protection）。
     *
     * @param lock Redisson 锁对象
     * @return 是否成功获取锁
     */
    private boolean tryLock(RLock lock) {
        try {
            return lock.tryLock(0, SUMMARY_LOCK_TTL.toMillis(), TimeUnit.MILLISECONDS);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    /**
     * 调用 LLM 将待压缩消息与已有摘要合并，生成更新后的摘要文本。
     * <p>
     * 增量摘要合并策略：
     * <ol>
     *   <li>加载摘要 Prompt 模板，注入摘要最大字符数限制</li>
     *   <li>若存在旧摘要，将其作为 ASSISTANT 消息注入（标注"仅用于合并去重"），
     *       避免 LLM 将旧摘要内容当作新事实</li>
     *   <li>将待压缩的历史消息按原始角色拼接</li>
     *   <li>最后添加 USER 指令消息，要求 LLM 合并去重并控制字符数</li>
     *   <li>使用低温度（0.3）确保摘要稳定性和准确性</li>
     * </ol>
     * <p>
     * 降级策略：LLM 调用失败时返回旧摘要内容，避免丢失已有的摘要数据。
     *
     * @param messages        待压缩的消息列表
     * @param existingSummary 已有的旧摘要文本，可能为空字符串
     * @return 更新后的摘要文本，LLM 调用失败时返回旧摘要
     */
    private String summarizeMessages(List<ConversationMessageDO> messages, String existingSummary) {
        List<ChatMessage> histories = toHistoryMessages(messages);
        if (CollUtil.isEmpty(histories)) {
            return existingSummary;
        }

        // 构建摘要生成的消息列表
        int summaryMaxChars = memoryProperties.getSummaryMaxChars();
        List<ChatMessage> summaryMessages = new ArrayList<>();
        // 加载摘要 Prompt 模板，注入字符数上限参数
        String summaryPrompt = promptTemplateLoader.render(
                CONVERSATION_SUMMARY_PROMPT_PATH,
                Map.of("summary_max_chars", String.valueOf(summaryMaxChars))
        );
        summaryMessages.add(ChatMessage.system(summaryPrompt));

        // 若存在旧摘要，将其作为 ASSISTANT 消息注入，并明确标注仅用于合并去重
        // 这样做是为了防止 LLM 将旧摘要中的信息当作新事实，同时允许冲突时以新对话为准
        if (StrUtil.isNotBlank(existingSummary)) {
            summaryMessages.add(ChatMessage.assistant(
                    "历史摘要（仅用于合并去重，不得作为事实新增来源；若与本轮对话冲突，以本轮对话为准）：\n"
                            + existingSummary.trim()
            ));
        }
        // 拼接待压缩的历史消息
        summaryMessages.addAll(histories);
        // 添加最终指令：要求 LLM 合并去重并控制字符数上限
        summaryMessages.add(ChatMessage.user(
                "合并以上对话与历史摘要，去重后输出更新摘要。要求：严格≤" + summaryMaxChars + "字符；仅一行。"
        ));

        ChatRequest request = ChatRequest.builder()
                .messages(summaryMessages)
                .temperature(0.3D)   // 低温度确保摘要准确稳定
                .topP(0.9D)
                .thinking(false)     // 摘要任务无需深度思考
                .build();
        try {
            String result = llmService.chat(request);
            log.info("对话摘要生成 - resultChars: {}", result.length());

            return result;
        } catch (Exception e) {
            // 降级策略：LLM 调用失败时返回旧摘要，保证不丢失已有的摘要数据
            log.error("对话记忆摘要生成失败, conversationId相关消息数: {}", messages.size(), e);
            return existingSummary;
        }
    }

    /**
     * 将数据库消息记录列表转换为 {@link ChatMessage} 列表。
     * <p>
     * 仅保留 USER 和 ASSISTANT 角色的有效消息（内容和角色不为空），
     * 其他角色（如 SYSTEM）的消息会被过滤掉。
     *
     * @param messages 数据库消息记录列表
     * @return ChatMessage 列表，输入为空时返回空列表
     */
    private List<ChatMessage> toHistoryMessages(List<ConversationMessageDO> messages) {
        if (CollUtil.isEmpty(messages)) {
            return List.of();
        }
        return messages.stream()
                .filter(item -> item != null
                        && StrUtil.isNotBlank(item.getContent())
                        && StrUtil.isNotBlank(item.getRole()))
                .map(item -> {
                    String role = item.getRole().toLowerCase();
                    if ("user".equals(role)) {
                        return ChatMessage.user(item.getContent());
                    } else if ("assistant".equals(role)) {
                        return ChatMessage.assistant(item.getContent());
                    }
                    return null;
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }

    /**
     * 将摘要数据库记录转换为 SYSTEM 角色的 {@link ChatMessage}。
     *
     * @param record 摘要数据库记录
     * @return SYSTEM 角色的摘要消息，记录无效时返回 null
     */
    private ChatMessage toChatMessage(ConversationSummaryDO record) {
        if (record == null || StrUtil.isBlank(record.getContent())) {
            return null;
        }
        return new ChatMessage(ChatMessage.Role.SYSTEM, record.getContent());
    }

    /**
     * 确定增量摘要的起始位置（afterId）—— 即上次摘要覆盖到的最后一条消息 ID。
     * <p>
     * 查找策略（按优先级）：
     * <ol>
     *   <li>直接使用摘要记录中保存的 {@code lastMessageId}（首选，精确）</li>
     *   <li>若 {@code lastMessageId} 不存在（兼容旧数据），则根据摘要的更新/创建时间
     *       查找时间点之前的最大消息 ID（近似）</li>
     * </ol>
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @param summary        上次的摘要记录，可能为 null
     * @return 上次摘要覆盖到的最后一条消息 ID，无上次摘要时返回 null（表示从头开始压缩）
     */
    private String resolveSummaryStartId(String conversationId, String userId, ConversationSummaryDO summary) {
        if (summary == null) {
            return null;
        }
        if (summary.getLastMessageId() != null) {
            return summary.getLastMessageId();
        }

        // 兼容旧数据：lastMessageId 不存在时，根据时间回溯查找对应的消息 ID
        Date after = summary.getUpdateTime();
        if (after == null) {
            after = summary.getCreateTime();
        }
        return conversationGroupService.findMaxMessageIdAtOrBefore(conversationId, userId, after);
    }

    /**
     * 确定摘要压缩的截止位置（cutoffId）—— 即滑动窗口中最早的 USER 消息 ID。
     * <p>
     * cutoffId 之前的消息需要被压缩成摘要，cutoffId 及之后的消息保留在滑动窗口中。
     * 输入列表是按时间倒序排列的，因此最后一个元素就是最早的消息。
     *
     * @param latestUserTurns 按时间倒序排列的最近 N 条 USER 消息
     * @return 最早的 USER 消息 ID（即窗口起始位置），列表为空时返回 null
     */
    private String resolveCutoffId(List<ConversationMessageDO> latestUserTurns) {
        if (CollUtil.isEmpty(latestUserTurns)) {
            return null;
        }

        // 倒序列表的最后一个就是最早的
        ConversationMessageDO oldest = latestUserTurns.get(latestUserTurns.size() - 1);
        return oldest == null ? null : oldest.getId();
    }

    /**
     * 从待压缩消息列表中获取最后一条有效消息的 ID。
     * <p>
     * 该 ID 将作为本次摘要覆盖的边界，保存到摘要记录中供下次增量压缩使用。
     * 从列表末尾向前遍历，跳过可能存在的 null 元素。
     *
     * @param toSummarize 待压缩的消息列表
     * @return 最后一条有效消息的 ID，列表为空或全部无效时返回 null
     */
    private String resolveLastMessageId(List<ConversationMessageDO> toSummarize) {
        for (int i = toSummarize.size() - 1; i >= 0; i--) {
            ConversationMessageDO item = toSummarize.get(i);
            if (item != null && item.getId() != null) {
                return item.getId();
            }
        }
        return null;
    }

    /**
     * 将摘要文本持久化到数据库。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @param content        摘要文本内容
     * @param lastMessageId  本次摘要覆盖到的最后一条消息 ID
     */
    private void createSummary(String conversationId,
                               String userId,
                               String content,
                               String lastMessageId) {
        ConversationSummaryBO summaryRecord = ConversationSummaryBO.builder()
                .conversationId(conversationId)
                .userId(userId)
                .content(content)
                .lastMessageId(lastMessageId)
                .build();
        conversationMessageService.addMessageSummary(summaryRecord);
    }

    /**
     * 构建分布式锁的 Key，格式为 {@code userId:conversationId}。
     *
     * @param conversationId 会话 ID
     * @param userId         用户 ID
     * @return 锁 Key 字符串
     */
    private String buildLockKey(String conversationId, String userId) {
        return userId.trim() + ":" + conversationId.trim();
    }
}
