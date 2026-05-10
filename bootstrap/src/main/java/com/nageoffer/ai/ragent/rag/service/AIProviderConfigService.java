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

package com.nageoffer.ai.ragent.rag.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nageoffer.ai.ragent.framework.errorcode.BaseErrorCode;
import com.nageoffer.ai.ragent.framework.exception.ClientException;
import com.nageoffer.ai.ragent.framework.exception.ServiceException;
import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * AI Provider 运行时配置服务
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AIProviderConfigService {

    private static final String DEFAULT_CONFIG_PATH = "data/ai-providers.json";

    private final AIModelProperties aiModelProperties;
    private final ObjectMapper objectMapper;

    @Value("${rag.settings.ai-provider-config-path:" + DEFAULT_CONFIG_PATH + "}")
    private String configPath;

    /**
     * 启动时加载后台保存的 Provider 配置
     */
    @PostConstruct
    public void loadSavedProviders() {
        Path path = resolveConfigPath();
        if (!Files.exists(path)) {
            return;
        }

        try {
            Map<String, AIModelProperties.ProviderConfig> providers = objectMapper.readValue(
                    path.toFile(),
                    new TypeReference<>() {
                    });
            aiModelProperties.setProviders(copyProviders(providers));
            log.info("已加载 AI Provider 运行时配置: {}", path);
        } catch (Exception ex) {
            log.warn("加载 AI Provider 运行时配置失败，将继续使用 application 配置: {}", path, ex);
        }
    }

    /**
     * 更新并持久化 Provider 配置
     */
    public synchronized Map<String, AIModelProperties.ProviderConfig> updateProviders(
            Map<String, AIModelProperties.ProviderConfig> providers) {
        Map<String, AIModelProperties.ProviderConfig> copiedProviders = copyProviders(providers);
        aiModelProperties.setProviders(copiedProviders);
        persist(copiedProviders);
        return copiedProviders;
    }

    private Map<String, AIModelProperties.ProviderConfig> copyProviders(
            Map<String, AIModelProperties.ProviderConfig> providers) {
        if (providers == null) {
            throw new ClientException("providers不能为空");
        }

        Map<String, AIModelProperties.ProviderConfig> result = new HashMap<>();
        providers.forEach((name, provider) -> {
            if (name == null || name.isBlank()) {
                throw new ClientException("Provider名称不能为空");
            }
            if (provider == null || provider.getUrl() == null || provider.getUrl().isBlank()) {
                throw new ClientException("Provider[" + name + "] URL不能为空");
            }

            AIModelProperties.ProviderConfig config = new AIModelProperties.ProviderConfig();
            config.setUrl(provider.getUrl().trim());
            config.setApiKey(provider.getApiKey());
            if (provider.getFormat() != null && !provider.getFormat().isBlank()) {
                config.setFormat(provider.getFormat().trim());
            }
            if (provider.getThinkingParameter() != null && !provider.getThinkingParameter().isBlank()) {
                config.setThinkingParameter(provider.getThinkingParameter().trim());
            }
            if (provider.getReasoningEffort() != null && !provider.getReasoningEffort().isBlank()) {
                config.setReasoningEffort(provider.getReasoningEffort().trim());
            }
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

    private void persist(Map<String, AIModelProperties.ProviderConfig> providers) {
        Path path = resolveConfigPath();
        try {
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(path.toFile(), providers);
        } catch (IOException ex) {
            throw new ServiceException("保存 AI Provider 配置失败", ex, BaseErrorCode.SERVICE_ERROR);
        }
    }

    private Path resolveConfigPath() {
        Path path = Path.of(configPath);
        if (path.isAbsolute()) {
            return path.normalize();
        }
        return Path.of("").toAbsolutePath().resolve(path).normalize();
    }
}
