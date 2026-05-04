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

package com.nageoffer.ai.ragent.rag.controller;

import com.nageoffer.ai.ragent.framework.convention.Result;
import com.nageoffer.ai.ragent.framework.exception.ClientException;
import com.nageoffer.ai.ragent.framework.web.Results;
import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import com.nageoffer.ai.ragent.rag.config.MemoryProperties;
import com.nageoffer.ai.ragent.rag.config.RAGConfigProperties;
import com.nageoffer.ai.ragent.rag.config.RAGDefaultProperties;
import com.nageoffer.ai.ragent.rag.config.RAGRateLimitProperties;
import com.nageoffer.ai.ragent.rag.controller.request.AIModelSelectionUpdateRequest;
import com.nageoffer.ai.ragent.rag.controller.request.AIProvidersUpdateRequest;
import com.nageoffer.ai.ragent.rag.controller.vo.SystemSettingsVO;
import com.nageoffer.ai.ragent.rag.controller.vo.SystemSettingsVO.AISettings;
import com.nageoffer.ai.ragent.rag.controller.vo.SystemSettingsVO.DefaultSettings;
import com.nageoffer.ai.ragent.rag.controller.vo.SystemSettingsVO.MemorySettings;
import com.nageoffer.ai.ragent.rag.service.AIModelSelectionConfigService;
import com.nageoffer.ai.ragent.rag.service.AIProviderConfigService;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.util.StringUtils;
import org.springframework.util.unit.DataSize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * RAG 设置控制器，负责系统 RAG、AI 模型等配置信息的查询
 */
@RestController
@RequiredArgsConstructor
public class RAGSettingsController {

    private final RAGDefaultProperties ragDefaultProperties;
    private final RAGConfigProperties ragConfigProperties;
    private final RAGRateLimitProperties ragRateLimitProperties;
    private final MemoryProperties memoryProperties;
    private final AIModelProperties aiModelProperties;
    private final AIProviderConfigService aiProviderConfigService;
    private final AIModelSelectionConfigService aiModelSelectionConfigService;

    @Value("${spring.servlet.multipart.max-file-size:50MB}")
    private DataSize maxFileSize;

    @Value("${spring.servlet.multipart.max-request-size:100MB}")
    private DataSize maxRequestSize;

    /**
     * 获取系统 RAG、AI 模型等配置信息
     */
    @GetMapping("/rag/settings")
    public Result<SystemSettingsVO> settings() {
        SystemSettingsVO response = SystemSettingsVO.builder()
                .upload(SystemSettingsVO.UploadSettings.builder()
                        .maxFileSize(maxFileSize.toBytes())
                        .maxRequestSize(maxRequestSize.toBytes())
                        .build())
                .rag(SystemSettingsVO.RagSettings.builder()
                        .defaultConfig(toDefaultSettings(ragDefaultProperties))
                        .queryRewrite(SystemSettingsVO.QueryRewriteSettings.builder()
                                .enabled(ragConfigProperties.getQueryRewriteEnabled())
                                .build())
                        .rateLimit(SystemSettingsVO.RateLimitSettings.builder()
                                .global(SystemSettingsVO.GlobalRateLimit.builder()
                                        .enabled(ragRateLimitProperties.getGlobalEnabled())
                                        .maxConcurrent(ragRateLimitProperties.getGlobalMaxConcurrent())
                                        .maxWaitSeconds(ragRateLimitProperties.getGlobalMaxWaitSeconds())
                                        .leaseSeconds(ragRateLimitProperties.getGlobalLeaseSeconds())
                                        .pollIntervalMs(ragRateLimitProperties.getGlobalPollIntervalMs())
                                        .build())
                                .build())
                        .memory(toMemorySettings(memoryProperties))
                        .build())
                .ai(toAISettings(aiModelProperties))
                .build();
        return Results.success(response);
    }

    /**
     * 更新 AI 服务提供方配置，立即影响当前进程内的模型路由。
     */
    @PutMapping("/rag/settings/ai/providers")
    public Result<Map<String, AISettings.ProviderConfig>> updateAIProviders(@RequestBody AIProvidersUpdateRequest request) {
        Map<String, AIModelProperties.ProviderConfig> providers = toRuntimeProviderConfig(request);
        aiProviderConfigService.updateProviders(providers);
        return Results.success(toAISettings(aiModelProperties).getProviders());
    }

    /**
     * 更新 AI 默认模型选择，立即影响当前进程内的模型路由。
     */
    @PutMapping("/rag/settings/ai/model-selection")
    public Result<AISettings> updateAIModelSelection(@RequestBody AIModelSelectionUpdateRequest request) {
        aiModelSelectionConfigService.updateSelection(request);
        return Results.success(toAISettings(aiModelProperties));
    }

    private DefaultSettings toDefaultSettings(RAGDefaultProperties props) {
        return DefaultSettings.builder()
                .collectionName(props.getCollectionName())
                .dimension(props.getDimension())
                .metricType(props.getMetricType())
                .build();
    }

    private MemorySettings toMemorySettings(MemoryProperties props) {
        return MemorySettings.builder()
                .historyKeepTurns(props.getHistoryKeepTurns())
                .summaryEnabled(props.getSummaryEnabled())
                .summaryStartTurns(props.getSummaryStartTurns())
                .summaryMaxChars(props.getSummaryMaxChars())
                .titleMaxLength(props.getTitleMaxLength())
                .build();
    }

    private AISettings toAISettings(AIModelProperties props) {
        Map<String, AISettings.ProviderConfig> providers = new HashMap<>();
        if (props.getProviders() != null) {
            props.getProviders().forEach((k, v) -> providers.put(k, AISettings.ProviderConfig.builder()
                    .url(v.getUrl())
                    .apiKey(maskApiKey(v.getApiKey()))
                    .endpoints(v.getEndpoints())
                    .build()));
        }

        return AISettings.builder()
                .providers(providers)
                .chat(toModelGroup(props.getChat()))
                .embedding(toModelGroup(props.getEmbedding()))
                .rerank(toModelGroup(props.getRerank()))
                .selection(props.getSelection() == null
                        ? null
                        : AISettings.Selection.builder()
                          .failureThreshold(props.getSelection().getFailureThreshold())
                          .openDurationMs(props.getSelection().getOpenDurationMs())
                          .build())
                .stream(props.getStream() == null
                        ? null
                        : AISettings.Stream.builder()
                          .messageChunkSize(props.getStream().getMessageChunkSize())
                          .build())
                .build();
    }

    private AISettings.ModelGroup toModelGroup(AIModelProperties.ModelGroup group) {
        if (group == null) {
            return null;
        }
        return AISettings.ModelGroup.builder()
                .defaultModel(group.getDefaultModel())
                .deepThinkingModel(group.getDeepThinkingModel())
                .candidates(group.getCandidates() == null
                        ? null
                        : group.getCandidates().stream()
                          .map(c -> AISettings.ModelCandidate.builder()
                                    .id(c.getId())
                                    .provider(c.getProvider())
                                    .model(c.getModel())
                                    .url(c.getUrl())
                                    .dimension(c.getDimension())
                                    .priority(c.getPriority())
                                    .enabled(c.getEnabled())
                                    .supportsThinking(c.getSupportsThinking())
                                    .build())
                          .collect(Collectors.toList()))
                .build();
    }

    private Map<String, AIModelProperties.ProviderConfig> toRuntimeProviderConfig(AIProvidersUpdateRequest request) {
        if (request == null || request.getProviders() == null) {
            throw new ClientException("providers不能为空");
        }

        Map<String, AIModelProperties.ProviderConfig> result = new HashMap<>();
        request.getProviders().forEach((name, provider) -> {
            if (name == null || name.isBlank()) {
                throw new ClientException("Provider名称不能为空");
            }
            if (provider == null || provider.getUrl() == null || provider.getUrl().isBlank()) {
                throw new ClientException("Provider[" + name + "] URL不能为空");
            }

            AIModelProperties.ProviderConfig config = new AIModelProperties.ProviderConfig();
            config.setUrl(provider.getUrl().trim());
            config.setApiKey(provider.getApiKey());
            config.setEndpoints(copyEndpoints(name, provider.getEndpoints()));
            result.put(name.trim(), config);
        });
        return result;
    }

    private Map<String, String> copyEndpoints(String providerName, Map<String, String> endpoints) {
        Map<String, String> result = new HashMap<>();
        if (endpoints == null) {
            return result;
        }
        endpoints.forEach((name, path) -> {
            if (name == null || name.isBlank()) {
                throw new ClientException("Provider[" + providerName + "] Endpoint名称不能为空");
            }
            if (path == null || path.isBlank()) {
                throw new ClientException("Provider[" + providerName + "] Endpoint[" + name + "]路径不能为空");
            }
            result.put(name.trim(), path.trim());
        });
        return result;
    }

    private String maskApiKey(String apiKey) {
        if (!StringUtils.hasText(apiKey)) {
            return null;
        }
        String trimmed = apiKey.trim();
        if (trimmed.length() <= 10) {
            return "******";
        }
        return trimmed.substring(0, 6) + "***" + trimmed.substring(trimmed.length() - 4);
    }
}
