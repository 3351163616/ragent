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

import com.nageoffer.ai.ragent.infra.chat.StreamCallback;
import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import com.nageoffer.ai.ragent.rag.core.memory.ConversationMemoryService;
import com.nageoffer.ai.ragent.rag.service.ConversationGroupService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * 流式回调工厂 —— RAG 管线中"LLM 流式生成"阶段的组件工厂
 * <p>
 * 核心职责：创建 {@link StreamChatEventHandler} 实例，并将所需的依赖
 * （SSE 发射器、会话服务、记忆服务、任务管理器等）通过参数对象注入。
 * <p>
 * 使用工厂模式的原因：{@link StreamChatEventHandler} 是非 Spring 管理的对象
 * （每次流式请求都会创建一个新实例），但它需要依赖多个 Spring Bean。
 * 通过工厂将这些 Bean 集中注入，避免在业务代码中手动组装依赖。
 *
 * @see StreamChatEventHandler  实际的流式回调处理器
 * @see StreamChatHandlerParams 参数对象，封装创建处理器所需的全部依赖
 */
@Component
@RequiredArgsConstructor
public class StreamCallbackFactory {

    /** AI 模型配置属性，包含流式推送的分块大小等参数 */
    private final AIModelProperties modelProperties;
    /** 会话记忆服务，用于在流式完成后持久化 AI 回复消息 */
    private final ConversationMemoryService memoryService;
    /** 会话组服务，用于查询和更新会话标题等元信息 */
    private final ConversationGroupService conversationGroupService;
    /** 流式任务管理器，用于注册/注销任务以及处理分布式取消 */
    private final StreamTaskManager taskManager;

    /**
     * 创建聊天场景的流式事件处理器
     * <p>
     * 每次 RAG 流式对话请求都会调用此方法创建一个新的处理器实例。
     * 处理器负责：
     * <ul>
     *     <li>接收 LLM 流式输出并通过 SSE 推送给前端</li>
     *     <li>流式完成后将完整回复持久化到数据库</li>
     *     <li>自动为新会话生成标题</li>
     *     <li>支持通过 {@link StreamTaskManager} 实现的分布式取消</li>
     * </ul>
     *
     * @param emitter        Spring SSE 发射器，用于向前端推送流式事件
     * @param conversationId 当前会话的唯一标识
     * @param taskId         当前流式任务的唯一标识（用于任务注册和取消）
     * @return 实现了 {@link StreamCallback} 接口的事件处理器实例
     */
    public StreamCallback createChatEventHandler(SseEmitter emitter,
                                                 String conversationId,
                                                 String taskId) {
        StreamChatHandlerParams params = StreamChatHandlerParams.builder()
                .emitter(emitter)
                .conversationId(conversationId)
                .taskId(taskId)
                .modelProperties(modelProperties)
                .memoryService(memoryService)
                .conversationGroupService(conversationGroupService)
                .taskManager(taskManager)
                .build();

        return new StreamChatEventHandler(params);
    }
}
