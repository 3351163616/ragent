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

package com.nageoffer.ai.ragent.rag.core.configbootstrap;

import java.util.List;

public record ConfigBootstrapSuggestion(
        List<TermMappingSuggestion> termMappings,
        List<IntentNodeSuggestion> intentNodes
) {

    public static ConfigBootstrapSuggestion empty() {
        return new ConfigBootstrapSuggestion(List.of(), List.of());
    }

    public record TermMappingSuggestion(
            String sourceTerm,
            String targetTerm,
            double confidence,
            String riskLevel,
            List<String> evidence
    ) {
    }

    public record IntentNodeSuggestion(
            String intentCode,
            String name,
            Integer level,
            String parentCode,
            String description,
            List<String> examples,
            String kbId,
            String collectionName,
            Integer topK,
            Integer kind,
            Integer sortOrder,
            double confidence,
            String riskLevel,
            List<String> evidence
    ) {
    }
}
