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

package com.nageoffer.ai.ragent.infra.embedding;

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Embedding 调用结果，除向量外保留最终路由到的模型信息。
 */
@Data
@Builder
public class EmbeddingResult {

    /**
     * 文本向量。
     */
    private List<Float> embedding;

    /**
     * 实际使用的 embedding 模型 ID。
     */
    private String embeddingModelId;

    /**
     * 实际使用的模型提供商。
     */
    private String embeddingProvider;

    /**
     * 向量维度。
     */
    private int embeddingDimension;
}
