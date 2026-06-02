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

package com.nageoffer.ai.ragent.infra.chat.log;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import com.nageoffer.ai.ragent.infra.http.HttpMediaTypes;
import com.nageoffer.ai.ragent.infra.model.ModelTarget;
import okhttp3.Headers;
import okhttp3.Request;
import okhttp3.RequestBody;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LLMRequestLoggerTests {

    @TempDir
    private Path tempDir;

    @Test
    void shouldKeepResponseBodyDisabledByDefault() throws Exception {
        LLMRequestLogProperties properties = enabledProperties(false);
        LLMRequestLogger logger = new LLMRequestLogger(properties);

        LLMRequestLogger.LogContext context = logger.logChatRequest(
                chatRequest(),
                modelTarget(),
                httpRequest(),
                requestBody(),
                false
        );
        logger.logChatResponse(context, 200, Headers.of(), "{\"content\":\"hello\"}", null);

        JsonObject logRecord = readLogRecord();

        assertFalse(logRecord.has("response"));
    }

    @Test
    void shouldWriteSyncResponseBodyWhenEnabled() throws Exception {
        LLMRequestLogProperties properties = enabledProperties(true);
        LLMRequestLogger logger = new LLMRequestLogger(properties);

        LLMRequestLogger.LogContext context = logger.logChatRequest(
                chatRequest(),
                modelTarget(),
                httpRequest(),
                requestBody(),
                false
        );
        logger.logChatResponse(
                context,
                200,
                Headers.of("x-request-id", "req-1"),
                "{\"choices\":[{\"message\":{\"content\":\"pong\"}}]}",
                null
        );

        JsonObject response = readLogRecord().getAsJsonObject("response");

        assertEquals("sync", response.get("type").getAsString());
        assertEquals(200, response.get("statusCode").getAsInt());
        assertTrue(response.get("successful").getAsBoolean());
        assertEquals("req-1", response.getAsJsonObject("headers").getAsJsonArray("x-request-id").get(0).getAsString());
        assertEquals(
                "pong",
                response.getAsJsonObject("body")
                        .getAsJsonArray("choices")
                        .get(0)
                        .getAsJsonObject()
                        .getAsJsonObject("message")
                        .get("content")
                        .getAsString()
        );
    }

    @Test
    void shouldWriteStreamResponseSummaryWhenEnabled() throws Exception {
        LLMRequestLogProperties properties = enabledProperties(true);
        LLMRequestLogger logger = new LLMRequestLogger(properties);

        LLMRequestLogger.LogContext context = logger.logChatRequest(
                chatRequest(),
                modelTarget(),
                httpRequest(),
                requestBody(),
                true
        );
        logger.logChatStreamResponse(
                context,
                200,
                Headers.of(),
                "hello world",
                "thinking",
                null,
                3,
                true,
                false,
                null,
                null
        );

        JsonObject response = readLogRecord().getAsJsonObject("response");
        JsonObject body = response.getAsJsonObject("body");

        assertEquals("stream", response.get("type").getAsString());
        assertTrue(response.get("successful").getAsBoolean());
        assertEquals("hello world", body.get("content").getAsString());
        assertEquals("thinking", body.get("reasoning").getAsString());
        assertEquals(3, body.get("eventCount").getAsInt());
        assertTrue(body.get("completed").getAsBoolean());
        assertFalse(body.get("cancelled").getAsBoolean());
    }

    private LLMRequestLogProperties enabledProperties(boolean includeResponseBody) {
        LLMRequestLogProperties properties = new LLMRequestLogProperties();
        properties.setEnabled(true);
        properties.setDirectory(tempDir.toString());
        properties.setIncludeHeaders(true);
        properties.setIncludeResponseBody(includeResponseBody);
        properties.setPrettyPrint(true);
        return properties;
    }

    private ChatRequest chatRequest() {
        return ChatRequest.builder()
                .scene("unit-test")
                .messages(List.of(ChatMessage.user("ping")))
                .build();
    }

    private ModelTarget modelTarget() {
        AIModelProperties.ModelCandidate candidate = new AIModelProperties.ModelCandidate();
        candidate.setId("test-chat");
        candidate.setProvider("test-provider");
        candidate.setModel("test-model");
        candidate.setPriority(1);

        AIModelProperties.ProviderConfig provider = new AIModelProperties.ProviderConfig();
        return new ModelTarget("test-chat", candidate, provider);
    }

    private Request httpRequest() {
        return new Request.Builder()
                .url("http://localhost/v1/chat/completions")
                .post(RequestBody.create("{}", HttpMediaTypes.JSON))
                .build();
    }

    private JsonObject requestBody() {
        JsonObject body = new JsonObject();
        body.addProperty("model", "test-model");
        body.addProperty("stream", false);
        return body;
    }

    private JsonObject readLogRecord() throws IOException {
        Path logFile;
        try (var paths = Files.walk(tempDir)) {
            logFile = paths
                    .filter(path -> path.getFileName().toString().endsWith(".json"))
                    .findFirst()
                    .orElseThrow();
        }
        return JsonParser.parseString(Files.readString(logFile)).getAsJsonObject();
    }
}
