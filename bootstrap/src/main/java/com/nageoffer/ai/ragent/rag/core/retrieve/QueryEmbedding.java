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

package com.nageoffer.ai.ragent.rag.core.retrieve;

import lombok.Builder;
import lombok.Data;

/**
 * 单次检索请求内可复用的 Query Embedding。
 */
@Data
@Builder
public class QueryEmbedding {

    /**
     * 向量对应的查询文本。
     */
    private String query;

    /**
     * 实际使用的 embedding 模型 ID。
     */
    private String embeddingModelId;

    /**
     * 实际使用的 embedding 提供商。
     */
    private String embeddingProvider;

    /**
     * 向量维度。
     */
    private int embeddingDimension;

    /**
     * 归一化后的查询向量，直接用于向量库检索。
     */
    private float[] vector;
}
