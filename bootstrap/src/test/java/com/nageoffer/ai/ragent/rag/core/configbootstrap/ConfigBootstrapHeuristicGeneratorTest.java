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

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ConfigBootstrapHeuristicGeneratorTest {

    private final ConfigBootstrapHeuristicGenerator generator = new ConfigBootstrapHeuristicGenerator();

    @Test
    void shouldGenerateIntentAndTermCandidatesFromDocumentSamples() {
        ConfigBootstrapDocumentSample sample = new ConfigBootstrapDocumentSample(
                "kb-1",
                "财务制度",
                "kb_finance",
                1L,
                1L,
                "doc-1",
                "发票管理制度.md",
                1L,
                1,
                List.of(new ConfigBootstrapDocumentSample.ChunkSample(
                        "chunk-1",
                        1,
                        "公司税号（纳税人识别号、统一社会信用代码）需要填写在发票信息中。发票抬头/开票抬头，保持一致。"
                ))
        );

        ConfigBootstrapSuggestion suggestion = generator.generate(List.of(sample));

        assertTrue(suggestion.intentNodes().stream().anyMatch(item -> "财务制度".equals(item.name())));
        assertTrue(suggestion.intentNodes().stream().anyMatch(item -> "发票管理制度".equals(item.name())));
        assertTrue(suggestion.termMappings().stream()
                .anyMatch(item -> "纳税人识别号".equals(item.sourceTerm()) && "公司税号".equals(item.targetTerm())));
        assertTrue(suggestion.termMappings().stream()
                .anyMatch(item -> "开票抬头".equals(item.sourceTerm()) && "发票抬头".equals(item.targetTerm())));
    }
}
