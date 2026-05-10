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

package com.nageoffer.ai.ragent.infra.chat;

import com.google.gson.JsonObject;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import com.nageoffer.ai.ragent.infra.model.ModelTarget;
import okhttp3.OkHttpClient;

import java.util.Locale;
import java.util.concurrent.Executor;

/**
 * 配置驱动的 OpenAI 格式 ChatClient
 * 由 {@link com.nageoffer.ai.ragent.infra.config.ChatClientRegistrar} 根据 YAML 配置自动注册
 */
public class GenericOpenAIChatClient extends AbstractOpenAIStyleChatClient {

    private static final String THINKING_PARAMETER_ENABLE_THINKING = "enable_thinking";
    private static final String THINKING_PARAMETER_REASONING_EFFORT = "reasoning_effort";
    private static final String THINKING_PARAMETER_NONE = "none";
    private static final String DEFAULT_REASONING_EFFORT = "medium";

    private final String providerName;

    public GenericOpenAIChatClient(String providerName,
                                   OkHttpClient syncHttpClient,
                                   OkHttpClient streamingHttpClient,
                                   Executor modelStreamExecutor) {
        super(syncHttpClient, streamingHttpClient, modelStreamExecutor);
        this.providerName = providerName;
    }

    @Override
    public String provider() {
        return providerName;
    }

    @Override
    public String chat(ChatRequest request, ModelTarget target) {
        return doChat(request, target);
    }

    @Override
    public StreamCancellationHandle streamChat(ChatRequest request, StreamCallback callback, ModelTarget target) {
        return doStreamChat(request, callback, target);
    }

    @Override
    protected void customizeRequestBody(JsonObject body, ChatRequest request, ModelTarget target) {
        if (!Boolean.TRUE.equals(request.getThinking())) {
            return;
        }

        AIModelProperties.ProviderConfig provider = target == null ? null : target.provider();
        String thinkingParameter = normalizeThinkingParameter(provider == null ? null : provider.getThinkingParameter());
        if (THINKING_PARAMETER_REASONING_EFFORT.equals(thinkingParameter)) {
            body.addProperty(THINKING_PARAMETER_REASONING_EFFORT, resolveReasoningEffort(provider));
            return;
        }
        if (THINKING_PARAMETER_NONE.equals(thinkingParameter) || "disabled".equals(thinkingParameter)) {
            return;
        }

        body.addProperty(THINKING_PARAMETER_ENABLE_THINKING, true);
    }

    private String normalizeThinkingParameter(String parameter) {
        if (parameter == null || parameter.isBlank()) {
            return THINKING_PARAMETER_ENABLE_THINKING;
        }
        return parameter.trim().toLowerCase(Locale.ROOT).replace('-', '_');
    }

    private String resolveReasoningEffort(AIModelProperties.ProviderConfig provider) {
        if (provider == null || provider.getReasoningEffort() == null || provider.getReasoningEffort().isBlank()) {
            return DEFAULT_REASONING_EFFORT;
        }
        return provider.getReasoningEffort().trim();
    }
}
