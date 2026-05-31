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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.nageoffer.ai.ragent.framework.context.UserContext;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.framework.trace.RagTraceContext;
import com.nageoffer.ai.ragent.infra.model.ModelTarget;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import okhttp3.Headers;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * 将每次 Chat LLM HTTP 请求写入独立 JSON 文件
 */
@Slf4j
@Component("llmRequestLogger")
@RequiredArgsConstructor
public class LLMRequestLogger {

    private static final DateTimeFormatter DAY_FORMATTER = DateTimeFormatter.ISO_LOCAL_DATE;
    private static final DateTimeFormatter FILE_TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss-SSS");

    private final LLMRequestLogProperties properties;
    private final Gson prettyGson = new GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create();
    private final Gson compactGson = new GsonBuilder().disableHtmlEscaping().create();

    public LogContext logChatRequest(ChatRequest chatRequest,
                                     ModelTarget target,
                                     Request httpRequest,
                                     JsonObject requestBody,
                                     boolean stream) {
        if (!properties.isEnabled()) {
            return LogContext.noop();
        }

        try {
            OffsetDateTime now = OffsetDateTime.now();
            String requestId = UUID.randomUUID().toString();
            Map<String, Object> logRecord = new LinkedHashMap<>();
            logRecord.put("requestId", requestId);
            logRecord.put("timestamp", now.toString());
            logRecord.put("scene", defaultIfBlank(chatRequest.getScene(), "unknown"));
            logRecord.put("mode", stream ? "stream" : "sync");
            logRecord.put("provider", target.candidate().getProvider());
            logRecord.put("modelId", target.id());
            logRecord.put("model", target.candidate().getModel());
            logRecord.put("priority", target.candidate().getPriority());
            logRecord.put("url", httpRequest.url().toString());
            logRecord.put("method", httpRequest.method());
            if (properties.isIncludeHeaders()) {
                logRecord.put("headers", extractHeaders(httpRequest));
            }
            logRecord.put("body", requestBody);
            logRecord.put("metadata", chatRequest.getMetadata());
            logRecord.put("traceId", RagTraceContext.getTraceId());
            logRecord.put("taskId", RagTraceContext.getTaskId());
            logRecord.put("userId", UserContext.getUserId());
            logRecord.put("threadName", Thread.currentThread().getName());

            Path file = buildLogFile(now, logRecord, requestId);
            Files.createDirectories(file.getParent());
            writeLogRecord(file, logRecord);
            return new LogContext(file, logRecord);
        } catch (Exception ex) {
            log.warn("记录 LLM 请求日志失败", ex);
            return LogContext.noop();
        }
    }

    public void logChatResponse(LogContext context, Response response, String responseBody, Throwable error) {
        Integer statusCode = response == null ? null : response.code();
        Headers headers = response == null ? null : response.headers();
        logChatResponse(context, statusCode, headers, responseBody, error);
    }

    public void logChatResponse(LogContext context,
                                Integer statusCode,
                                Headers headers,
                                String responseBody,
                                Throwable error) {
        if (!shouldWriteResponse(context)) {
            return;
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("type", "sync");
        putCommonResponseFields(response, statusCode, headers, error);
        response.put("body", parseBodyForLog(responseBody));
        JsonElement usage = extractUsage(responseBody);
        if (usage != null) {
            response.put("usage", usage);
            log.info(
                    "LLM 响应 Token 用量: requestId={}, modelId={}, usage={}",
                    context.logRecord.get("requestId"),
                    context.logRecord.get("modelId"),
                    compactGson.toJson(usage)
            );
        }
        writeResponse(context, response);
    }

    public void logChatStreamResponse(LogContext context,
                                      Integer statusCode,
                                      Headers headers,
                                      String content,
                                      String reasoning,
                                      JsonElement usage,
                                      int eventCount,
                                      boolean completed,
                                      boolean cancelled,
                                      String rawBody,
                                      Throwable error) {
        if (!shouldWriteResponse(context)) {
            return;
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("content", defaultString(content));
        body.put("reasoning", defaultString(reasoning));
        body.put("eventCount", eventCount);
        body.put("completed", completed);
        body.put("cancelled", cancelled);
        if (rawBody != null && !rawBody.isBlank()) {
            body.put("raw", parseBodyForLog(rawBody));
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("type", "stream");
        putCommonResponseFields(response, statusCode, headers, error);
        response.put("body", body);
        if (usage != null) {
            response.put("usage", usage);
            log.info(
                    "LLM 流式响应 Token 用量: requestId={}, modelId={}, usage={}",
                    context.logRecord.get("requestId"),
                    context.logRecord.get("modelId"),
                    compactGson.toJson(usage)
            );
        }
        writeResponse(context, response);
    }

    private boolean shouldWriteResponse(LogContext context) {
        return properties.isEnabled()
                && properties.isIncludeResponseBody()
                && context != null
                && context.isEnabled();
    }

    private void putCommonResponseFields(Map<String, Object> response,
                                         Integer statusCode,
                                         Headers headers,
                                         Throwable error) {
        if (statusCode != null) {
            response.put("statusCode", statusCode);
            response.put("successful", statusCode >= 200 && statusCode < 300);
        }
        if (properties.isIncludeHeaders() && headers != null) {
            response.put("headers", extractHeaders(headers));
        }
        if (error != null) {
            response.put("error", error.getMessage());
            response.put("errorType", error.getClass().getName());
        }
    }

    private void writeResponse(LogContext context, Map<String, Object> response) {
        try {
            synchronized (context) {
                context.logRecord.put("response", response);
                writeLogRecord(context.file, context.logRecord);
            }
        } catch (Exception ex) {
            log.warn("记录 LLM 响应日志失败", ex);
        }
    }

    private Map<String, List<String>> extractHeaders(Request request) {
        Map<String, List<String>> result = new LinkedHashMap<>();
        Headers headers = request.headers();
        for (String name : headers.names()) {
            result.put(name, headers.values(name));
        }

        RequestBody body = request.body();
        if (body != null && body.contentType() != null && !result.containsKey("Content-Type")) {
            result.put("Content-Type", List.of(body.contentType().toString()));
        }
        return result;
    }

    private Map<String, List<String>> extractHeaders(Headers headers) {
        Map<String, List<String>> result = new LinkedHashMap<>();
        for (String name : headers.names()) {
            result.put(name, headers.values(name));
        }
        return result;
    }

    private Path buildLogFile(OffsetDateTime now, Map<String, Object> logRecord, String requestId) {
        Path root = Path.of(properties.getDirectory());
        if (!root.isAbsolute()) {
            root = Path.of("").toAbsolutePath().resolve(root).normalize();
        }

        String timestamp = FILE_TIME_FORMATTER.format(now);
        String scene = sanitize((String) logRecord.get("scene"));
        String provider = sanitize((String) logRecord.get("provider"));
        String modelId = sanitize((String) logRecord.get("modelId"));
        String suffix = requestId.substring(0, 8);
        String fileName = String.format(
                "%s_%s_%s_%s_%s.json",
                timestamp,
                scene,
                provider,
                modelId,
                suffix
        );
        return root.resolve(DAY_FORMATTER.format(now.toLocalDate())).resolve(fileName);
    }

    private void writeLogRecord(Path file, Map<String, Object> logRecord) throws java.io.IOException {
        Gson gson = properties.isPrettyPrint() ? prettyGson : compactGson;
        Files.writeString(file, gson.toJson(logRecord), StandardCharsets.UTF_8);
    }

    private Object parseBodyForLog(String body) {
        if (body == null || body.isBlank()) {
            return "";
        }
        try {
            JsonElement json = JsonParser.parseString(body);
            return json == null ? body : json;
        } catch (Exception ignored) {
            return body;
        }
    }

    private JsonElement extractUsage(String body) {
        if (body == null || body.isBlank()) {
            return null;
        }
        try {
            JsonElement json = JsonParser.parseString(body);
            if (json != null && json.isJsonObject()) {
                JsonObject object = json.getAsJsonObject();
                if (object.has("usage")) {
                    return object.get("usage");
                }
            }
        } catch (Exception ignored) {
            return null;
        }
        return null;
    }

    private String sanitize(String value) {
        String text = defaultIfBlank(value, "unknown")
                .replaceAll("[\\\\/:*?\"<>|\\s]+", "_")
                .replaceAll("_+", "_");
        if (text.length() > 80) {
            return text.substring(0, 80);
        }
        return text;
    }

    private String defaultIfBlank(String value, String defaultValue) {
        return value == null || value.isBlank() ? defaultValue : value;
    }

    private String defaultString(String value) {
        return value == null ? "" : value;
    }

    public record LogContext(Path file, Map<String, Object> logRecord) {

        private static LogContext noop() {
            return new LogContext(null, null);
        }

        private boolean isEnabled() {
            return file != null && logRecord != null;
        }
    }
}
