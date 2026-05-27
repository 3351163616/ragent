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

import java.util.List;

@Data
public class ConfigBootstrapCreateRequest {

    /**
     * 为空时分析全部未删除知识库。
     */
    private List<String> kbIds;

    /**
     * 最多采样文档数，默认 100。
     */
    private Integer maxDocuments;

    /**
     * 每篇文档最多采样 chunk 数，默认 3。
     */
    private Integer maxChunksPerDocument;

    /**
     * 是否调用 LLM 生成候选，默认 true；false 时仅使用规则兜底。
     */
    private Boolean useLlm;
}
