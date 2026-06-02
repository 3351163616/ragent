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
import com.nageoffer.ai.ragent.infra.chat.log.LLMRequestLogProperties;
import com.nageoffer.ai.ragent.infra.chat.log.LLMRequestLogger;
import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import com.nageoffer.ai.ragent.infra.model.ModelTarget;
import okhttp3.OkHttpClient;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class GenericAnthropicChatClientTests {

    @Test
    void shouldDisableMimoThinkingWhenNotRequested() {
        JsonObject body = client("mimo").buildRequestBody(nonThinkingRequest(), target("mimo-v2.5-pro"), false);

        assertEquals("disabled", body.getAsJsonObject("thinking").get("type").getAsString());
    }

    @Test
    void shouldDisableMimoThinkingWhenThinkingIsUnset() {
        JsonObject body = client("mimo").buildRequestBody(noThinkingFlagRequest(), target("mimo-v2.5-pro"), false);

        assertEquals("disabled", body.getAsJsonObject("thinking").get("type").getAsString());
    }

    @Test
    void shouldKeepThinkingEnabledWhenRequested() {
        JsonObject body = client("mimo").buildRequestBody(thinkingRequest(), target("mimo-v2.5-pro"), false);

        assertEquals("enabled", body.getAsJsonObject("thinking").get("type").getAsString());
    }

    @Test
    void shouldNotAddDisabledThinkingForOtherAnthropicProviders() {
        JsonObject body = client("glm").buildRequestBody(nonThinkingRequest(), target("glm-5.1"), false);

        assertFalse(body.has("thinking"));
    }

    private GenericAnthropicChatClient client(String provider) {
        return new GenericAnthropicChatClient(
                provider,
                new OkHttpClient(),
                new OkHttpClient(),
                Runnable::run,
                new LLMRequestLogger(new LLMRequestLogProperties())
        );
    }

    private ChatRequest thinkingRequest() {
        return ChatRequest.builder()
                .messages(List.of(ChatMessage.user("hello")))
                .thinking(true)
                .build();
    }

    private ChatRequest nonThinkingRequest() {
        return ChatRequest.builder()
                .messages(List.of(ChatMessage.user("hello")))
                .thinking(false)
                .build();
    }

    private ChatRequest noThinkingFlagRequest() {
        return ChatRequest.builder()
                .messages(List.of(ChatMessage.user("hello")))
                .build();
    }

    private ModelTarget target(String model) {
        AIModelProperties.ModelCandidate candidate = new AIModelProperties.ModelCandidate();
        candidate.setModel(model);
        return new ModelTarget(model, candidate, new AIModelProperties.ProviderConfig());
    }
}
