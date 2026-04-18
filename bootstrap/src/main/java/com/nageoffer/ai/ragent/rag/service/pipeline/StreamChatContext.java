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

package com.nageoffer.ai.ragent.rag.service.pipeline;

import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.infra.chat.StreamCallback;
import com.nageoffer.ai.ragent.rag.core.rewrite.RewriteResult;
import com.nageoffer.ai.ragent.rag.dto.SubQuestionIntent;
import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * 流式对话上下文
 */
@Data
@Builder
public class StreamChatContext {

    /**
     * 原始问题
     */
    private final String question;

    /**
     * 会话ID
     */
    private final String conversationId;

    /**
     * 任务ID
     */
    private final String taskId;

    /**
     * 是否启用深度思考
     */
    private final boolean deepThinking;

    /**
     * 用户ID
     */
    private final String userId;

    /**
     * 流式回调（用于输出给前端）
     */
    private final StreamCallback callback;

    /**
     * 对话历史（过程中填充）
     */
    private List<ChatMessage> history;

    /**
     * 查询改写结果（过程中填充）
     */
    private RewriteResult rewriteResult;

    /**
     * 子问题意图列表（过程中填充）
     */
    private List<SubQuestionIntent> subIntents;
}
