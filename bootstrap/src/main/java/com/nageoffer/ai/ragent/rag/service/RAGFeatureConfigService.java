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
import com.nageoffer.ai.ragent.rag.config.RAGConfigProperties;
import com.nageoffer.ai.ragent.rag.controller.request.RAGCitationSettingsUpdateRequest;
import jakarta.annotation.PostConstruct;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * RAG 功能运行时配置服务
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RAGFeatureConfigService {

    private static final String DEFAULT_CONFIG_PATH = "data/rag-features.json";

    private final RAGConfigProperties ragConfigProperties;
    private final ObjectMapper objectMapper;

    @Value("${rag.settings.feature-config-path:" + DEFAULT_CONFIG_PATH + "}")
    private String configPath;

    @PostConstruct
    public void loadSavedFeatures() {
        Path path = resolveConfigPath();
        if (!Files.exists(path)) {
            return;
        }

        try {
            RuntimeFeatureConfig config = objectMapper.readValue(path.toFile(), RuntimeFeatureConfig.class);
            if (config.getAnswerCitationEnabled() != null) {
                ragConfigProperties.setAnswerCitationEnabled(config.getAnswerCitationEnabled());
            }
            log.info("已加载 RAG 运行时功能配置: {}", path);
        } catch (Exception ex) {
            log.warn("加载 RAG 运行时功能配置失败，将继续使用 application 配置: {}", path, ex);
        }
    }

    public synchronized Boolean updateCitationSettings(RAGCitationSettingsUpdateRequest request) {
        if (request == null || request.getEnabled() == null) {
            throw new ClientException("引用来源开关不能为空");
        }
        ragConfigProperties.setAnswerCitationEnabled(request.getEnabled());
        RuntimeFeatureConfig config = new RuntimeFeatureConfig();
        config.setAnswerCitationEnabled(request.getEnabled());
        persist(config);
        return request.getEnabled();
    }

    private void persist(RuntimeFeatureConfig config) {
        Path path = resolveConfigPath();
        try {
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(path.toFile(), config);
        } catch (IOException ex) {
            throw new ServiceException("保存 RAG 功能配置失败", ex, BaseErrorCode.SERVICE_ERROR);
        }
    }

    private Path resolveConfigPath() {
        Path path = Path.of(configPath);
        if (path.isAbsolute()) {
            return path.normalize();
        }
        return Path.of("").toAbsolutePath().resolve(path).normalize();
    }

    @Data
    public static class RuntimeFeatureConfig {
        private Boolean answerCitationEnabled;
    }
}
