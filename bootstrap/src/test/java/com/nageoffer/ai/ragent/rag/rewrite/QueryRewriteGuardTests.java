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

package com.nageoffer.ai.ragent.rag.rewrite;

import com.nageoffer.ai.ragent.infra.embedding.EmbeddingService;
import com.nageoffer.ai.ragent.rag.config.RAGConfigProperties;
import com.nageoffer.ai.ragent.rag.core.rewrite.QueryRewriteGuard;
import com.nageoffer.ai.ragent.rag.core.rewrite.RewriteResult;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class QueryRewriteGuardTests {

    @Test
    void acceptsRewriteWhenSemanticSimilarityPasses() {
        FakeEmbeddingService embeddingService = new FakeEmbeddingService()
                .put("OA系统在移动端的审批流程", List.of(1.0f, 0.0f))
                .put("OA系统移动端审批流程", List.of(0.95f, 0.05f));
        QueryRewriteGuard guard = new QueryRewriteGuard(embeddingService, properties());
        RewriteResult candidate = new RewriteResult("OA系统移动端审批流程", List.of("OA系统移动端审批流程"));

        RewriteResult result = guard.validateOrFallback(
                "OA系统在移动端的审批流程",
                "OA系统在移动端的审批流程",
                candidate
        );

        assertThat(result).isEqualTo(candidate);
    }

    @Test
    void fallsBackWhenSemanticSimilarityIsBelowThreshold() {
        FakeEmbeddingService embeddingService = new FakeEmbeddingService()
                .put("OA系统在移动端的审批流程", List.of(1.0f, 0.0f))
                .put("北京今天天气", List.of(0.0f, 1.0f));
        QueryRewriteGuard guard = new QueryRewriteGuard(embeddingService, properties());

        RewriteResult result = guard.validateOrFallback(
                "OA系统在移动端的审批流程",
                "OA系统在移动端的审批流程",
                new RewriteResult("北京今天天气", List.of("北京今天天气"))
        );

        assertThat(result.rewrittenQuestion()).isEqualTo("OA系统在移动端的审批流程");
        assertThat(result.subQuestions()).containsExactly("OA系统在移动端的审批流程");
    }

    @Test
    void shortQueryFallsBackWhenRefundIsRewrittenToPayment() {
        FakeEmbeddingService embeddingService = new FakeEmbeddingService()
                .put("怎么退费", List.of(1.0f, 0.0f))
                .put("如何申请收费", List.of(0.95f, 0.05f));
        QueryRewriteGuard guard = new QueryRewriteGuard(embeddingService, properties());

        RewriteResult result = guard.validateOrFallback(
                "怎么退费",
                "怎么退费",
                new RewriteResult("如何申请收费", List.of("如何申请收费"))
        );

        assertThat(result.rewrittenQuestion()).isEqualTo("怎么退费");
        assertThat(result.subQuestions()).containsExactly("怎么退费");
    }

    @Test
    void shortQueryFallsBackWhenLogisticsRewriteAddsReturnScenario() {
        FakeEmbeddingService embeddingService = new FakeEmbeddingService()
                .put("查物流", List.of(1.0f, 0.0f))
                .put("查询退货物流流程", List.of(0.95f, 0.05f));
        QueryRewriteGuard guard = new QueryRewriteGuard(embeddingService, properties());

        RewriteResult result = guard.validateOrFallback(
                "查物流",
                "查物流",
                new RewriteResult("查询退货物流流程", List.of("查询退货物流流程"))
        );

        assertThat(result.rewrittenQuestion()).isEqualTo("查物流");
        assertThat(result.subQuestions()).containsExactly("查物流");
    }

    @Test
    void fallsBackWhenSimilarityCannotBeCalculated() {
        FakeEmbeddingService embeddingService = new FakeEmbeddingService()
                .put("怎么退费", List.of(1.0f, 0.0f));
        QueryRewriteGuard guard = new QueryRewriteGuard(embeddingService, properties());

        RewriteResult result = guard.validateOrFallback(
                "怎么退费",
                "怎么退费",
                new RewriteResult("怎么退款", List.of("怎么退款"))
        );

        assertThat(result.rewrittenQuestion()).isEqualTo("怎么退费");
        assertThat(result.subQuestions()).containsExactly("怎么退费");
    }

    private static RAGConfigProperties properties() {
        RAGConfigProperties properties = new RAGConfigProperties();
        properties.setQueryRewriteSemanticValidationEnabled(true);
        properties.setQueryRewriteSemanticSimilarityThreshold(0.8D);
        properties.setQueryRewriteShortQueryMaxLength(8);
        return properties;
    }

    private static final class FakeEmbeddingService implements EmbeddingService {

        private final Map<String, List<Float>> embeddings = new HashMap<>();

        private FakeEmbeddingService put(String text, List<Float> embedding) {
            embeddings.put(text, embedding);
            return this;
        }

        @Override
        public List<Float> embed(String text) {
            List<Float> embedding = embeddings.get(text);
            if (embedding == null) {
                throw new IllegalArgumentException("missing embedding: " + text);
            }
            return embedding;
        }

        @Override
        public List<Float> embed(String text, String modelId) {
            return embed(text);
        }

        @Override
        public List<List<Float>> embedBatch(List<String> texts) {
            return texts.stream().map(this::embed).toList();
        }

        @Override
        public List<List<Float>> embedBatch(List<String> texts, String modelId) {
            return embedBatch(texts);
        }
    }
}
