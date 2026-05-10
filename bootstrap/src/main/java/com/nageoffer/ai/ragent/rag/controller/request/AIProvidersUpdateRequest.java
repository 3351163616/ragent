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

package com.nageoffer.ai.ragent.rag.controller.request;

import lombok.Data;

import java.util.Map;

/**
 * AI 服务提供方更新请求
 */
@Data
public class AIProvidersUpdateRequest {

    private Map<String, ProviderConfig> providers;

    @Data
    public static class ProviderConfig {

        private String url;

        private String apiKey;

        private String format;

        private String thinkingParameter;

        private String reasoningEffort;

        private Map<String, String> endpoints;
    }
}
