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

import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.infra.model.ModelTarget;
import okhttp3.OkHttpClient;

import java.util.concurrent.Executor;

/**
 * 配置驱动的 OpenAI 格式 ChatClient
 * 由 {@link com.nageoffer.ai.ragent.infra.config.ChatClientRegistrar} 根据 YAML 配置自动注册
 */
public class GenericOpenAIChatClient extends AbstractOpenAIStyleChatClient {

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
}
