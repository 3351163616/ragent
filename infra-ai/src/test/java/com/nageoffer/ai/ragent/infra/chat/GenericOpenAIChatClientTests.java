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
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import com.nageoffer.ai.ragent.infra.model.ModelTarget;
import okhttp3.OkHttpClient;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GenericOpenAIChatClientTests {

    private final GenericOpenAIChatClient client = new GenericOpenAIChatClient(
            "gpt55",
            new OkHttpClient(),
            new OkHttpClient(),
            Runnable::run
    );

    @Test
    void shouldUseConfiguredReasoningEffortParameter() {
        AIModelProperties.ProviderConfig provider = new AIModelProperties.ProviderConfig();
        provider.setThinkingParameter("reasoning_effort");
        provider.setReasoningEffort("high");

        JsonObject body = client.buildRequestBody(thinkingRequest(), target(provider), true);

        assertEquals("high", body.get("reasoning_effort").getAsString());
        assertFalse(body.has("enable_thinking"));
    }

    @Test
    void shouldKeepEnableThinkingAsDefaultParameter() {
        JsonObject body = client.buildRequestBody(thinkingRequest(), target(new AIModelProperties.ProviderConfig()), true);

        assertTrue(body.get("enable_thinking").getAsBoolean());
        assertFalse(body.has("reasoning_effort"));
    }

    private ChatRequest thinkingRequest() {
        return ChatRequest.builder()
                .messages(List.of(ChatMessage.user("hello")))
                .thinking(true)
                .build();
    }

    private ModelTarget target(AIModelProperties.ProviderConfig provider) {
        AIModelProperties.ModelCandidate candidate = new AIModelProperties.ModelCandidate();
        candidate.setModel("gpt-5.5");
        return new ModelTarget("gpt-5.5", candidate, provider);
    }
}
