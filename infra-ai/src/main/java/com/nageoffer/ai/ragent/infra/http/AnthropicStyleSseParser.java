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

package com.nageoffer.ai.ragent.infra.http;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import lombok.NoArgsConstructor;

/**
 * Anthropic 协议风格 SSE 解析器
 * 处理 Anthropic Messages API 的流式事件格式
 */
@NoArgsConstructor(access = lombok.AccessLevel.PRIVATE)
public final class AnthropicStyleSseParser {

    private static final String DATA_PREFIX = "data:";
    private static final String EVENT_PREFIX = "event:";

    public static ParsedEvent parseLine(String line, Gson gson) {
        if (line == null || line.isBlank()) {
            return ParsedEvent.empty();
        }

        String trimmed = line.trim();

        if (trimmed.startsWith(EVENT_PREFIX)) {
            String eventType = trimmed.substring(EVENT_PREFIX.length()).trim();
            return switch (eventType) {
                case "message_stop" -> ParsedEvent.done();
                case "error" -> ParsedEvent.error();
                default -> ParsedEvent.empty();
            };
        }

        if (!trimmed.startsWith(DATA_PREFIX)) {
            return ParsedEvent.empty();
        }

        String payload = trimmed.substring(DATA_PREFIX.length()).trim();
        if (payload.isBlank()) {
            return ParsedEvent.empty();
        }

        JsonObject obj = gson.fromJson(payload, JsonObject.class);
        if (obj == null) {
            return ParsedEvent.empty();
        }

        if ("error".equals(optString(obj, "type"))) {
            return ParsedEvent.error();
        }

        if (!obj.has("delta") || !obj.get("delta").isJsonObject()) {
            return ParsedEvent.empty();
        }

        JsonObject delta = obj.getAsJsonObject("delta");
        String deltaType = optString(delta, "type");
        if (deltaType == null) {
            return ParsedEvent.empty();
        }

        return switch (deltaType) {
            case "text_delta" -> ParsedEvent.content(optString(delta, "text"));
            case "thinking_delta" -> ParsedEvent.thinking(optString(delta, "thinking"));
            default -> ParsedEvent.empty();
        };
    }

    private static String optString(JsonObject obj, String key) {
        if (obj == null || !obj.has(key) || obj.get(key).isJsonNull()) {
            return null;
        }
        return obj.get(key).getAsString();
    }

    public record ParsedEvent(String content, String reasoning, boolean completed, boolean isError) {

        public static ParsedEvent empty() {
            return new ParsedEvent(null, null, false, false);
        }

        public static ParsedEvent done() {
            return new ParsedEvent(null, null, true, false);
        }

        public static ParsedEvent error() {
            return new ParsedEvent(null, null, false, true);
        }

        public static ParsedEvent content(String text) {
            return new ParsedEvent(text, null, false, false);
        }

        public static ParsedEvent thinking(String text) {
            return new ParsedEvent(null, text, false, false);
        }

        public boolean hasContent() {
            return content != null && !content.isEmpty();
        }

        public boolean hasReasoning() {
            return reasoning != null && !reasoning.isEmpty();
        }
    }
}
