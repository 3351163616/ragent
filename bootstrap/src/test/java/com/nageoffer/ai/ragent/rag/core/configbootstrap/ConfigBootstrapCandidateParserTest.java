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

import static org.junit.jupiter.api.Assertions.assertEquals;

class ConfigBootstrapCandidateParserTest {

    private final ConfigBootstrapCandidateParser parser = new ConfigBootstrapCandidateParser();

    @Test
    void shouldParseMarkdownWrappedCandidateJson() {
        String raw = """
                ```json
                {
                  "term_mappings": [
                    {
                      "sourceTerm": "开票抬头",
                      "targetTerm": "发票抬头",
                      "confidence": 0.91,
                      "riskLevel": "LOW",
                      "evidence": ["发票管理制度"]
                    }
                  ],
                  "intent_nodes": [
                    {
                      "intentCode": "finance-invoice",
                      "name": "发票管理",
                      "level": 2,
                      "description": "发票抬头、税号、开票地址等问题",
                      "examples": ["公司税号是多少？"],
                      "kbId": "123",
                      "collectionName": "kb_finance",
                      "kind": 0,
                      "confidence": 0.86,
                      "riskLevel": "MEDIUM",
                      "evidence": ["发票管理制度.md"]
                    }
                  ]
                }
                ```
                """;

        ConfigBootstrapSuggestion suggestion = parser.parse(raw);

        assertEquals(1, suggestion.termMappings().size());
        assertEquals("开票抬头", suggestion.termMappings().get(0).sourceTerm());
        assertEquals("发票抬头", suggestion.termMappings().get(0).targetTerm());
        assertEquals("LOW", suggestion.termMappings().get(0).riskLevel());
        assertEquals(1, suggestion.intentNodes().size());
        assertEquals("finance-invoice", suggestion.intentNodes().get(0).intentCode());
        assertEquals("发票管理", suggestion.intentNodes().get(0).name());
    }
}
