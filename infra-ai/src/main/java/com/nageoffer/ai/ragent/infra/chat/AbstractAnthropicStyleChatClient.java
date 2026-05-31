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

import cn.hutool.core.collection.CollUtil;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.infra.chat.log.LLMRequestLogger;
import com.nageoffer.ai.ragent.infra.chat.log.LLMRequestLogger.LogContext;
import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import com.nageoffer.ai.ragent.infra.enums.ModelCapability;
import com.nageoffer.ai.ragent.infra.http.AnthropicStyleSseParser;
import com.nageoffer.ai.ragent.infra.http.HttpMediaTypes;
import com.nageoffer.ai.ragent.infra.http.HttpResponseHelper;
import com.nageoffer.ai.ragent.infra.http.ModelClientErrorType;
import com.nageoffer.ai.ragent.infra.http.ModelClientException;
import com.nageoffer.ai.ragent.infra.http.ModelUrlResolver;
import com.nageoffer.ai.ragent.infra.model.ModelTarget;
import lombok.extern.slf4j.Slf4j;
import okhttp3.Call;
import okhttp3.Headers;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;
import okio.BufferedSource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Anthropic Messages API 协议 ChatClient 抽象基类
 */
@Slf4j
public abstract class AbstractAnthropicStyleChatClient implements ChatClient {

    private static final String ANTHROPIC_VERSION = "2023-06-01";
    private static final int DEFAULT_MAX_TOKENS = 4096;

    protected final OkHttpClient syncHttpClient;
    protected final OkHttpClient streamingHttpClient;
    protected final Executor modelStreamExecutor;
    protected final LLMRequestLogger requestLogger;
    protected final Gson gson = new Gson();

    protected AbstractAnthropicStyleChatClient(OkHttpClient syncHttpClient,
                                               OkHttpClient streamingHttpClient,
                                               Executor modelStreamExecutor,
                                               LLMRequestLogger requestLogger) {
        this.syncHttpClient = syncHttpClient;
        this.streamingHttpClient = streamingHttpClient;
        this.modelStreamExecutor = modelStreamExecutor;
        this.requestLogger = requestLogger;
    }

    protected String doChat(ChatRequest request, ModelTarget target) {
        AIModelProperties.ProviderConfig provider = HttpResponseHelper.requireProvider(target, provider());

        JsonObject reqBody = buildRequestBody(request, target, false);
        Request httpRequest = newAuthorizedRequest(provider, target)
                .post(RequestBody.create(reqBody.toString(), HttpMediaTypes.JSON))
                .build();
        LogContext logContext = requestLogger.logChatRequest(request, target, httpRequest, reqBody, false);

        JsonObject respJson;
        try (Response response = syncHttpClient.newCall(httpRequest).execute()) {
            ResponseBody responseBody = response.body();
            String body = HttpResponseHelper.readBody(responseBody);
            requestLogger.logChatResponse(logContext, response, body, null);
            if (!response.isSuccessful()) {
                log.warn("{} 同步请求失败: status={}, body={}", provider(), response.code(), body);
                throw new ModelClientException(
                        provider() + " 同步请求失败: HTTP " + response.code(),
                        ModelClientErrorType.fromHttpStatus(response.code()),
                        response.code()
                );
            }
            if (responseBody == null) {
                throw new ModelClientException(provider() + " 响应为空", ModelClientErrorType.INVALID_RESPONSE, null);
            }
            respJson = gson.fromJson(body, JsonObject.class);
        } catch (IOException e) {
            requestLogger.logChatResponse(logContext, null, null, null, e);
            throw new ModelClientException(
                    provider() + " 同步请求失败: " + e.getMessage(),
                    ModelClientErrorType.NETWORK_ERROR, null, e);
        }

        return extractChatContent(respJson);
    }

    protected StreamCancellationHandle doStreamChat(ChatRequest request, StreamCallback callback, ModelTarget target) {
        AIModelProperties.ProviderConfig provider = HttpResponseHelper.requireProvider(target, provider());

        JsonObject reqBody = buildRequestBody(request, target, true);
        Request streamRequest = newAuthorizedRequest(provider, target)
                .post(RequestBody.create(reqBody.toString(), HttpMediaTypes.JSON))
                .addHeader("Accept", "text/event-stream")
                .build();
        LogContext logContext = requestLogger.logChatRequest(request, target, streamRequest, reqBody, true);

        Call call = streamingHttpClient.newCall(streamRequest);
        return StreamAsyncExecutor.submit(
                modelStreamExecutor,
                call,
                callback,
                cancelled -> doStream(call, callback, cancelled, Boolean.TRUE.equals(request.getThinking()), logContext)
        );
    }

    private void doStream(Call call,
                          StreamCallback callback,
                          AtomicBoolean cancelled,
                          boolean reasoningEnabled,
                          LogContext logContext) {
        Integer statusCode = null;
        Headers responseHeaders = null;
        String rawBody = null;
        StringBuilder contentBuilder = new StringBuilder();
        StringBuilder reasoningBuilder = new StringBuilder();
        JsonElement usage = null;
        int eventCount = 0;
        boolean completed = false;
        Throwable error = null;
        try (Response response = call.execute()) {
            statusCode = response.code();
            responseHeaders = response.headers();
            if (!response.isSuccessful()) {
                rawBody = HttpResponseHelper.readBody(response.body());
                throw new ModelClientException(
                        provider() + " 流式请求失败: HTTP " + response.code() + " - " + rawBody,
                        ModelClientErrorType.fromHttpStatus(response.code()),
                        response.code()
                );
            }
            ResponseBody body = response.body();
            if (body == null) {
                throw new ModelClientException(provider() + " 流式响应为空", ModelClientErrorType.INVALID_RESPONSE, null);
            }
            BufferedSource source = body.source();
            while (!cancelled.get()) {
                String line = source.readUtf8Line();
                if (line == null) {
                    break;
                }
                if (line.isBlank()) {
                    continue;
                }
                if (log.isDebugEnabled()) {
                    log.debug("{} SSE line: {}", provider(), line);
                }
                eventCount++;
                try {
                    AnthropicStyleSseParser.ParsedEvent event = AnthropicStyleSseParser.parseLine(line, gson);
                    if (event.isError()) {
                        String detail = event.errorMessage() != null ? event.errorMessage() : line;
                        log.warn("{} 流式响应错误事件: {}", provider(), detail);
                        throw new ModelClientException(provider() + " 流式响应收到错误事件: " + detail, ModelClientErrorType.INVALID_RESPONSE, null);
                    }
                    if (reasoningEnabled && event.hasReasoning()) {
                        reasoningBuilder.append(event.reasoning());
                        callback.onThinking(event.reasoning());
                    }
                    if (event.hasContent()) {
                        contentBuilder.append(event.content());
                        callback.onContent(event.content());
                    }
                    if (event.hasUsage()) {
                        usage = event.usage();
                    }
                    if (event.completed()) {
                        callback.onComplete();
                        completed = true;
                        break;
                    }
                } catch (ModelClientException e) {
                    throw e;
                } catch (Exception parseEx) {
                    log.warn("{} 流式响应解析失败: line={}", provider(), line, parseEx);
                }
            }
            if (cancelled.get()) {
                log.info("{} 流式响应已被取消", provider());
                return;
            }
            if (!completed) {
                throw new ModelClientException(provider() + " 流式响应异常结束", ModelClientErrorType.INVALID_RESPONSE, null);
            }
        } catch (Exception e) {
            error = e;
            if (!cancelled.get()) {
                callback.onError(e);
            } else {
                log.info("{} 流式响应取消期间产生异常（可忽略）: {}", provider(), e.getMessage());
            }
        } finally {
            requestLogger.logChatStreamResponse(
                    logContext,
                    statusCode,
                    responseHeaders,
                    contentBuilder.toString(),
                    reasoningBuilder.toString(),
                    usage,
                    eventCount,
                    completed,
                    cancelled.get(),
                    rawBody,
                    error
            );
        }
    }

    protected JsonObject buildRequestBody(ChatRequest request, ModelTarget target, boolean stream) {
        JsonObject body = new JsonObject();
        body.addProperty("model", HttpResponseHelper.requireModel(target, provider()));
        if (stream) {
            body.addProperty("stream", true);
        }

        int maxTokens = request.getMaxTokens() != null ? request.getMaxTokens() : DEFAULT_MAX_TOKENS;
        body.addProperty("max_tokens", maxTokens);

        if (Boolean.TRUE.equals(request.getThinking())) {
            JsonObject thinking = new JsonObject();
            thinking.addProperty("type", "enabled");
            thinking.addProperty("budget_tokens", maxTokens);
            body.add("thinking", thinking);
        }

        JsonArray messagesArr = new JsonArray();
        List<String> systemParts = new ArrayList<>();
        List<ChatMessage> messages = request.getMessages();
        if (CollUtil.isNotEmpty(messages)) {
            for (ChatMessage m : messages) {
                if (m.getRole() == ChatMessage.Role.SYSTEM) {
                    if (m.getContent() != null && !m.getContent().isBlank()) {
                        systemParts.add(m.getContent());
                    }
                } else {
                    JsonObject msg = new JsonObject();
                    msg.addProperty("role", m.getRole() == ChatMessage.Role.ASSISTANT ? "assistant" : "user");
                    msg.addProperty("content", m.getContent());
                    messagesArr.add(msg);
                }
            }
        }
        body.add("messages", messagesArr);

        if (!systemParts.isEmpty()) {
            body.addProperty("system", String.join("\n\n", systemParts));
        }

        if (request.getTemperature() != null) {
            body.addProperty("temperature", request.getTemperature());
        }
        if (request.getTopP() != null) {
            body.addProperty("top_p", request.getTopP());
        }
        if (request.getTopK() != null) {
            body.addProperty("top_k", request.getTopK());
        }

        return body;
    }

    private Request.Builder newAuthorizedRequest(AIModelProperties.ProviderConfig provider, ModelTarget target) {
        Request.Builder builder = new Request.Builder()
                .url(ModelUrlResolver.resolveUrl(provider, target.candidate(), ModelCapability.CHAT));
        builder.addHeader("anthropic-version", ANTHROPIC_VERSION);
        if (provider.getApiKey() != null && !provider.getApiKey().isBlank()) {
            builder.addHeader("x-api-key", provider.getApiKey());
        }
        return builder;
    }

    private String extractChatContent(JsonObject respJson) {
        if (respJson == null || !respJson.has("content")) {
            throw new ModelClientException(provider() + " 响应缺少 content", ModelClientErrorType.INVALID_RESPONSE, null);
        }
        JsonArray content = respJson.getAsJsonArray("content");
        if (content == null || content.isEmpty()) {
            throw new ModelClientException(provider() + " 响应 content 为空", ModelClientErrorType.INVALID_RESPONSE, null);
        }
        StringBuilder sb = new StringBuilder();
        for (JsonElement elem : content) {
            JsonObject block = elem.getAsJsonObject();
            if ("text".equals(optString(block, "type"))) {
                String text = optString(block, "text");
                if (text != null) {
                    sb.append(text);
                }
            }
        }
        return sb.toString();
    }

    private static String optString(JsonObject obj, String key) {
        if (obj == null || !obj.has(key) || obj.get(key).isJsonNull()) {
            return null;
        }
        return obj.get(key).getAsString();
    }
}
