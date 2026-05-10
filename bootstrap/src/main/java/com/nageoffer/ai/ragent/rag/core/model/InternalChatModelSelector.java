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

package com.nageoffer.ai.ragent.rag.core.model;

import com.nageoffer.ai.ragent.infra.config.AIModelProperties;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

/**
 * RAG 内部 LLM 任务的模型选择器。
 */
@Component
@RequiredArgsConstructor
public class InternalChatModelSelector {

    private final AIModelProperties aiModelProperties;

    public String modelId() {
        AIModelProperties.ModelGroup chat = aiModelProperties.getChat();
        if (chat == null) {
            return null;
        }
        return StringUtils.hasText(chat.getInternalModel())
                ? chat.getInternalModel()
                : chat.getDefaultModel();
    }
}
