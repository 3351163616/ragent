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

package com.nageoffer.ai.ragent.rag.service.handler;

import cn.hutool.core.util.StrUtil;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.nageoffer.ai.ragent.rag.enums.SSEEventType;
import com.nageoffer.ai.ragent.rag.dto.CompletionPayload;
import com.nageoffer.ai.ragent.framework.web.SseEmitterSender;
import com.nageoffer.ai.ragent.infra.chat.StreamCancellationHandle;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBucket;
import org.redisson.api.RTopic;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

/**
 * 流式任务管理器 —— RAG 管线中"LLM 流式生成"阶段的分布式任务管控组件
 * <p>
 * 核心职责：管理流式对话任务的注册、注销和分布式取消。
 * <p>
 * 分布式取消机制：
 * <ol>
 *     <li><b>Redis Pub/Sub 广播</b>：取消请求通过 Redis Topic 广播到所有节点，
 *         确保在多实例部署时，不论请求落在哪个节点都能正确取消</li>
 *     <li><b>Redis Key 标记</b>：同时在 Redis 中设置取消标记（带 TTL），
 *         用于处理"取消消息先于任务注册"的竞态条件（register 时检查 Redis 标记）</li>
 *     <li><b>本地缓存</b>：使用 Guava Cache 存储任务信息，通过 {@link java.util.concurrent.atomic.AtomicBoolean}
 *         的 CAS 操作确保取消回调只执行一次</li>
 * </ol>
 * <p>
 * 任务生命周期：
 * <pre>
 * register(taskId) → bindHandle(taskId) → [正常完成] → unregister(taskId)
 *                                        ↘ [被取消] → cancelLocal(taskId) → 保存已累积内容 → 通知前端
 * </pre>
 * <p>
 * 线程安全保证：
 * <ul>
 *     <li>任务取消状态使用 {@link java.util.concurrent.atomic.AtomicBoolean#compareAndSet} 保证只执行一次</li>
 *     <li>Guava Cache 本身是线程安全的</li>
 *     <li>Redis 操作通过 Redisson 保证原子性</li>
 * </ul>
 *
 * @see StreamChatEventHandler 流式回调处理器，在 register 时提供取消回调
 */
@Slf4j
@Component
public class StreamTaskManager {

    /** Redis Pub/Sub 取消通知的 Topic 名称 */
    private static final String CANCEL_TOPIC = "ragent:stream:cancel";
    /** Redis 取消标记 Key 的前缀 */
    private static final String CANCEL_KEY_PREFIX = "ragent:stream:cancel:";
    /** Redis 取消标记的 TTL（30 分钟），超时自动清理 */
    private static final Duration CANCEL_TTL = Duration.ofMinutes(30);

    /**
     * 本地任务信息缓存
     * <p>
     * 使用 Guava Cache 而非 ConcurrentHashMap，利用其自动过期和容量限制能力，
     * 避免因异常场景（如 unregister 未被调用）导致的内存泄漏。
     */
    private final Cache<String, StreamTaskInfo> tasks = CacheBuilder.newBuilder()
            .expireAfterWrite(CANCEL_TTL)
            .maximumSize(10000)  // 限制最大数量，基本上不可能超出这个数量。如果觉得不稳妥，可以把值调大并在配置文件声明
            .build();

    /** Redisson 客户端，用于 Redis Pub/Sub 和 Key 操作 */
    private final RedissonClient redissonClient;
    /** Redis Topic 监听器 ID，用于销毁时移除监听器 */
    private int listenerId = -1;

    /**
     * 构造方法
     *
     * @param redissonClient Redisson 客户端实例
     */
    public StreamTaskManager(RedissonClient redissonClient) {
        this.redissonClient = redissonClient;
    }

    /**
     * 应用启动时订阅 Redis 取消通知 Topic
     * <p>
     * 监听器收到取消消息后，调用 {@link #cancelLocal(String)} 执行本地取消逻辑。
     * 所有节点（包括发布取消消息的节点自身）都会收到通知，统一通过监听器处理。
     */
    @PostConstruct
    public void subscribe() {
        RTopic topic = redissonClient.getTopic(CANCEL_TOPIC);
        listenerId = topic.addListener(String.class, (channel, taskId) -> {
            if (StrUtil.isBlank(taskId)) {
                return;
            }
            cancelLocal(taskId);
        });
    }

    /**
     * 应用关闭时取消订阅 Redis Topic，释放资源
     */
    @PreDestroy
    public void unsubscribe() {
        if (listenerId == -1) {
            return;
        }
        redissonClient.getTopic(CANCEL_TOPIC).removeListener(listenerId);
    }

    /**
     * 注册流式任务
     * <p>
     * 将 SSE 发射器和取消回调绑定到任务信息中。
     * 注册时会检查 Redis 中是否已存在取消标记（处理"取消先于注册"的竞态），
     * 若已被取消则立即执行取消回调、发送取消事件并关闭连接。
     *
     * @param taskId           任务唯一标识
     * @param sender           SSE 发射器封装，用于向前端推送事件
     * @param onCancelSupplier 取消时的回调，返回包含已保存消息 ID 和标题的完成载荷
     */
    public void register(String taskId, SseEmitterSender sender, Supplier<CompletionPayload> onCancelSupplier) {
        StreamTaskInfo taskInfo = getOrCreate(taskId);
        taskInfo.sender = sender;
        taskInfo.onCancelSupplier = onCancelSupplier;
        // 竞态处理：注册时检查 Redis 中是否已有取消标记
        // 场景：用户在流式请求启动前就点击了取消按钮
        if (isTaskCancelledInRedis(taskId, taskInfo)) {
            CompletionPayload payload = taskInfo.onCancelSupplier.get();
            sendCancelAndDone(sender, payload);
            sender.complete();
        }
    }

    /**
     * 绑定流式取消句柄
     * <p>
     * 在流式请求启动后，将取消句柄绑定到任务。
     * 如果在绑定时任务已被标记为取消，则立即通过句柄取消流式请求。
     *
     * @param taskId 任务唯一标识
     * @param handle 流式请求的取消句柄
     */
    public void bindHandle(String taskId, StreamCancellationHandle handle) {
        StreamTaskInfo taskInfo = getOrCreate(taskId);
        taskInfo.handle = handle;
        // 若绑定句柄时任务已被取消，立即取消底层的流式请求
        if (taskInfo.cancelled.get() && handle != null) {
            handle.cancel();
        }
    }

    /**
     * 检查指定任务是否已被取消
     * <p>
     * 仅检查本地缓存中的取消状态，不查询 Redis（性能优先）。
     * 该方法会在每次 {@link StreamChatEventHandler} 的回调中调用，
     * 因此必须保证极高的性能。
     *
     * @param taskId 任务唯一标识
     * @return true 表示任务已被取消
     */
    public boolean isCancelled(String taskId) {
        StreamTaskInfo info = tasks.getIfPresent(taskId);
        return info != null && info.cancelled.get();
    }

    /**
     * 发起分布式取消操作
     * <p>
     * 两步操作保证可靠取消：
     * <ol>
     *     <li>先在 Redis 中设置取消标记（带 TTL），用于处理竞态条件</li>
     *     <li>再通过 Redis Pub/Sub 广播取消消息，通知所有节点执行本地取消</li>
     * </ol>
     * <p>
     * 本地节点也通过监听器统一处理，不直接调用 cancelLocal，
     * 避免重复执行取消逻辑。
     *
     * @param taskId 要取消的任务标识
     */
    public void cancel(String taskId) {
        // 第一步：设置 Redis 取消标记，解决"取消先于注册"的竞态问题
        RBucket<Boolean> bucket = redissonClient.getBucket(cancelKey(taskId));
        bucket.set(Boolean.TRUE, CANCEL_TTL);

        // 第二步：通过 Pub/Sub 广播取消消息到所有节点（包括本地）
        // 所有节点通过 subscribe() 中注册的监听器统一处理
        redissonClient.getTopic(CANCEL_TOPIC).publish(taskId);
    }

    /**
     * 检查任务是否在 Redis 中被标记为已取消
     * <p>
     * 如果 Redis 中存在取消标记，则同步更新本地缓存的取消状态。
     * 用于 register 时的竞态检测。
     *
     * @param taskId   任务唯一标识
     * @param taskInfo 本地任务信息对象
     * @return true 表示任务已在 Redis 中被标记为取消
     */
    private boolean isTaskCancelledInRedis(String taskId, StreamTaskInfo taskInfo) {
        if (taskInfo.cancelled.get()) {
            return true;
        }

        RBucket<Boolean> bucket = redissonClient.getBucket(cancelKey(taskId));
        Boolean cancelled = bucket.get();
        if (Boolean.TRUE.equals(cancelled)) {
            taskInfo.cancelled.set(true);
            return true;
        }
        return false;
    }

    /**
     * 执行本地取消逻辑
     * <p>
     * 由 Redis Pub/Sub 监听器触发。使用 CAS（compareAndSet）保证取消回调只执行一次：
     * <ol>
     *     <li>通过 AtomicBoolean.compareAndSet(false, true) 原子标记为已取消</li>
     *     <li>若存在流式取消句柄，调用 handle.cancel() 中断底层 HTTP 请求</li>
     *     <li>执行取消回调（保存已累积的回复内容），发送 CANCEL 和 DONE 事件</li>
     *     <li>关闭 SSE 连接</li>
     * </ol>
     *
     * @param taskId 要取消的任务标识
     */
    private void cancelLocal(String taskId) {
        StreamTaskInfo taskInfo = tasks.getIfPresent(taskId);
        if (taskInfo == null) {
            return;
        }

        // 使用 CAS 确保取消逻辑只执行一次，防止 Pub/Sub 重复投递或并发调用
        if (!taskInfo.cancelled.compareAndSet(false, true)) {
            return;
        }

        // 取消底层的流式 HTTP 请求
        if (taskInfo.handle != null) {
            taskInfo.handle.cancel();
        }

        // 执行取消回调：保存已累积内容，通知前端取消完成
        // 在取消时执行回调，保存已累积的内容
        if (taskInfo.sender != null) {
            CompletionPayload payload = taskInfo.onCancelSupplier.get();
            sendCancelAndDone(taskInfo.sender, payload);
            taskInfo.sender.complete();
        }
    }

    /**
     * 注销流式任务
     * <p>
     * 在任务正常完成或出错后调用，清理本地缓存和 Redis 中的取消标记。
     * Redis 清理使用异步删除（deleteAsync），避免阻塞业务线程。
     *
     * @param taskId 要注销的任务标识
     */
    public void unregister(String taskId) {
        // 清理本地 Guava Cache
        tasks.invalidate(taskId);

        // 异步清理 Redis 取消标记，避免阻塞
        redissonClient.getBucket(cancelKey(taskId)).deleteAsync();
    }

    /**
     * 构建 Redis 取消标记的完整 Key
     *
     * @param taskId 任务标识
     * @return 格式为 "ragent:stream:cancel:{taskId}" 的 Redis Key
     */
    private String cancelKey(String taskId) {
        return CANCEL_KEY_PREFIX + taskId;
    }

    /**
     * 向前端发送取消完成事件和流结束标记
     *
     * @param sender  SSE 发射器封装
     * @param payload 完成载荷（含消息 ID 和标题），可为 null
     */
    private void sendCancelAndDone(SseEmitterSender sender, CompletionPayload payload) {
        CompletionPayload actualPayload = payload == null ? new CompletionPayload(null, null) : payload;
        sender.sendEvent(SSEEventType.CANCEL.value(), actualPayload);
        sender.sendEvent(SSEEventType.DONE.value(), "[DONE]");
    }

    /**
     * 获取或创建任务信息对象
     * <p>
     * 使用 Guava Cache 的 get(key, callable) 方法保证原子性：
     * 同一 taskId 只会创建一个 StreamTaskInfo 实例。
     *
     * @param taskId 任务标识
     * @return 任务信息对象（已存在则返回现有的，不存在则新建）
     */
    @SneakyThrows
    private StreamTaskInfo getOrCreate(String taskId) {
        return tasks.get(taskId, StreamTaskInfo::new);
    }

    /**
     * 流式任务信息内部类
     * <p>
     * 封装单个流式任务的运行时状态和回调引用。
     * 所有字段使用 volatile 或 AtomicBoolean 保证多线程可见性。
     */
    private static final class StreamTaskInfo {
        /** 取消状态标记，使用 AtomicBoolean 的 CAS 操作保证取消逻辑只执行一次 */
        private final AtomicBoolean cancelled = new AtomicBoolean(false);
        /** 底层流式请求的取消句柄，调用 cancel() 可中断 HTTP 请求 */
        private volatile StreamCancellationHandle handle;
        /** SSE 发射器封装，用于向前端推送取消事件 */
        private volatile SseEmitterSender sender;
        /** 取消时的回调供应者，返回包含已保存消息 ID 和标题的载荷 */
        private volatile Supplier<CompletionPayload> onCancelSupplier;
    }
}
