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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nageoffer.ai.ragent.framework.errorcode.BaseErrorCode;
import com.nageoffer.ai.ragent.framework.exception.ClientException;
import com.nageoffer.ai.ragent.framework.exception.ServiceException;
import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import com.nageoffer.ai.ragent.rag.controller.request.AIModelSelectionUpdateRequest;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * AI 默认模型选择运行时配置服务
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AIModelSelectionConfigService {

    private static final String DEFAULT_CONFIG_PATH = "data/ai-model-selection.json";

    private final AIModelProperties aiModelProperties;
    private final ObjectMapper objectMapper;

    @Value("${rag.settings.ai-model-selection-config-path:" + DEFAULT_CONFIG_PATH + "}")
    private String configPath;

    /**
     * 启动时加载后台保存的默认模型选择
     */
    @PostConstruct
    public void loadSavedSelection() {
        Path path = resolveConfigPath();
        if (!Files.exists(path)) {
            return;
        }

        try {
            AIModelSelectionUpdateRequest request = objectMapper.readValue(path.toFile(), AIModelSelectionUpdateRequest.class);
            applySelection(request);
            log.info("已加载 AI 默认模型运行时配置: {}", path);
        } catch (Exception ex) {
            log.warn("加载 AI 默认模型运行时配置失败，将继续使用 application 配置: {}", path, ex);
        }
    }

    /**
     * 更新并持久化默认模型选择
     */
    public synchronized void updateSelection(AIModelSelectionUpdateRequest request) {
        applySelection(request);
        persist(request);
    }

    private void applySelection(AIModelSelectionUpdateRequest request) {
        if (request == null) {
            throw new ClientException("模型选择配置不能为空");
        }

        setDefaultModel(aiModelProperties.getChat(), request.getChatDefaultModel(), "Chat默认模型");
        setDeepThinkingModel(aiModelProperties.getChat(), request.getChatDeepThinkingModel());
        setInternalModel(aiModelProperties.getChat(), request.getChatInternalModel(), request.getChatDefaultModel());
        setDefaultModel(aiModelProperties.getEmbedding(), request.getEmbeddingDefaultModel(), "Embedding模型");
        setDefaultModel(aiModelProperties.getRerank(), request.getRerankDefaultModel(), "Rerank模型");
    }

    private void setDefaultModel(AIModelProperties.ModelGroup group, String modelId, String label) {
        requireCandidate(group, modelId, label);
        group.setDefaultModel(modelId);
    }

    private void setDeepThinkingModel(AIModelProperties.ModelGroup group, String modelId) {
        AIModelProperties.ModelCandidate candidate = requireCandidate(group, modelId, "深度思考模型");
        if (!Boolean.TRUE.equals(candidate.getSupportsThinking())) {
            throw new ClientException("深度思考模型必须选择支持Thinking的Chat候选模型");
        }
        group.setDeepThinkingModel(modelId);
    }

    private void setInternalModel(AIModelProperties.ModelGroup group, String modelId, String fallbackModelId) {
        String actualModelId = StringUtils.hasText(modelId) ? modelId : fallbackModelId;
        requireCandidate(group, actualModelId, "内部任务模型");
        group.setInternalModel(actualModelId);
    }

    private AIModelProperties.ModelCandidate requireCandidate(
            AIModelProperties.ModelGroup group, String modelId, String label) {
        if (group == null || group.getCandidates() == null || group.getCandidates().isEmpty()) {
            throw new ClientException(label + "候选列表为空");
        }
        if (modelId == null || modelId.isBlank()) {
            throw new ClientException(label + "不能为空");
        }
        return group.getCandidates().stream()
                .filter(candidate -> Objects.equals(resolveCandidateId(candidate), modelId))
                .findFirst()
                .orElseThrow(() -> new ClientException(label + "不在候选模型列表中: " + modelId));
    }

    private String resolveCandidateId(AIModelProperties.ModelCandidate candidate) {
        if (candidate.getId() != null && !candidate.getId().isBlank()) {
            return candidate.getId();
        }
        return candidate.getProvider() + "::" + candidate.getModel();
    }

    private void persist(AIModelSelectionUpdateRequest request) {
        Path path = resolveConfigPath();
        try {
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(path.toFile(), request);
        } catch (IOException ex) {
            throw new ServiceException("保存 AI 默认模型配置失败", ex, BaseErrorCode.SERVICE_ERROR);
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
