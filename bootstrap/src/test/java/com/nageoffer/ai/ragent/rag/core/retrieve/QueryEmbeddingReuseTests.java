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

import com.nageoffer.ai.ragent.framework.convention.RetrievedChunk;
import com.nageoffer.ai.ragent.infra.embedding.EmbeddingResult;
import com.nageoffer.ai.ragent.infra.embedding.EmbeddingService;
import com.nageoffer.ai.ragent.knowledge.dao.entity.KnowledgeBaseDO;
import com.nageoffer.ai.ragent.knowledge.dao.mapper.KnowledgeBaseMapper;
import com.nageoffer.ai.ragent.rag.config.SearchChannelProperties;
import com.nageoffer.ai.ragent.rag.core.intent.IntentNode;
import com.nageoffer.ai.ragent.rag.core.intent.NodeScore;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.IntentDirectedSearchChannel;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchContext;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchTarget;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.VectorGlobalSearchChannel;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.strategy.CollectionParallelRetriever;
import com.nageoffer.ai.ragent.rag.dto.SubQuestionIntent;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class QueryEmbeddingReuseTests {

    private static final Executor DIRECT_EXECUTOR = Runnable::run;

    @Test
    void sameEmbeddingModelTargetsShareSingleEmbedding() {
        CountingEmbeddingService embeddingService = new CountingEmbeddingService();
        RecordingRetrieverService retrieverService = new RecordingRetrieverService();
        CollectionParallelRetriever retriever = new CollectionParallelRetriever(retrieverService, DIRECT_EXECUTOR);
        SearchContext context = searchContext("如何办理入职", embeddingService);
        List<SearchTarget> targets = List.of(
                target("collection-a", "model-a"),
                target("collection-b", "model-a"),
                target("collection-c", "model-a")
        );

        List<RetrievedChunk> optimizedChunks = retriever.executeParallelRetrieval(context, targets, 5);
        List<RetrievedChunk> oldChunks = retriever.executeParallelRetrieval("如何办理入职", targets, 5);

        assertThat(embeddingService.totalCalls()).isEqualTo(1);
        assertThat(embeddingService.callsFor("model-a")).isEqualTo(1);
        assertThat(retrieverService.vectorCalls()).isEqualTo(3);
        assertThat(retrieverService.retrieveCalls()).isEqualTo(3);
        assertThat(optimizedChunks).hasSameSizeAs(oldChunks).hasSize(3);
    }

    @Test
    void differentEmbeddingModelTargetsEmbedByModelId() {
        CountingEmbeddingService embeddingService = new CountingEmbeddingService();
        RecordingRetrieverService retrieverService = new RecordingRetrieverService();
        CollectionParallelRetriever retriever = new CollectionParallelRetriever(retrieverService, DIRECT_EXECUTOR);
        SearchContext context = searchContext("如何报销发票", embeddingService);
        List<SearchTarget> targets = List.of(
                target("collection-a", "model-a"),
                target("collection-b", "model-b"),
                target("collection-c", "model-c")
        );

        List<RetrievedChunk> chunks = retriever.executeParallelRetrieval(context, targets, 5);

        assertThat(embeddingService.totalCalls()).isEqualTo(3);
        assertThat(embeddingService.callsFor("model-a")).isEqualTo(1);
        assertThat(embeddingService.callsFor("model-b")).isEqualTo(1);
        assertThat(embeddingService.callsFor("model-c")).isEqualTo(1);
        assertThat(retrieverService.vectorCalls()).isEqualTo(3);
        assertThat(chunks).hasSize(3);
    }

    @Test
    void intentDirectedAndVectorGlobalReuseSameEmbeddingModel() {
        CountingEmbeddingService embeddingService = new CountingEmbeddingService();
        RecordingRetrieverService retrieverService = new RecordingRetrieverService();
        KnowledgeBaseMapper knowledgeBaseMapper = mock(KnowledgeBaseMapper.class);
        when(knowledgeBaseMapper.selectList(any())).thenReturn(List.of(kb("kb-a", "入职知识库", "collection-a", "model-a")));

        SearchChannelProperties properties = new SearchChannelProperties();
        IntentDirectedSearchChannel intentChannel = new IntentDirectedSearchChannel(
                retrieverService,
                properties,
                knowledgeBaseMapper,
                DIRECT_EXECUTOR
        );
        VectorGlobalSearchChannel globalChannel = new VectorGlobalSearchChannel(
                retrieverService,
                properties,
                knowledgeBaseMapper,
                DIRECT_EXECUTOR
        );
        MultiChannelRetrievalEngine engine = new MultiChannelRetrievalEngine(
                List.of(intentChannel, globalChannel),
                List.of(),
                embeddingService,
                DIRECT_EXECUTOR
        );
        SubQuestionIntent intent = new SubQuestionIntent("如何办理入职", List.of(NodeScore.builder()
                .node(IntentNode.builder()
                        .id("intent-a")
                        .name("入职")
                        .collectionName("collection-a")
                        .build())
                .score(0.7)
                .build()));

        List<RetrievedChunk> chunks = engine.retrieveKnowledgeChannels(List.of(intent), 5);

        assertThat(embeddingService.totalCalls()).isEqualTo(1);
        assertThat(embeddingService.callsFor("model-a")).isEqualTo(1);
        assertThat(retrieverService.vectorCalls()).isEqualTo(2);
        assertThat(chunks).hasSize(2);
    }

    @Test
    void missingQueryEmbeddingContextFallsBackToOldRetrieveLogic() {
        RecordingRetrieverService retrieverService = new RecordingRetrieverService();
        CollectionParallelRetriever retriever = new CollectionParallelRetriever(retrieverService, DIRECT_EXECUTOR);
        SearchContext contextWithoutEmbedding = SearchContext.builder()
                .originalQuestion("如何申请 VPN")
                .rewrittenQuestion("如何申请 VPN")
                .build();

        List<RetrievedChunk> chunks = retriever.executeParallelRetrieval(
                contextWithoutEmbedding,
                List.of(target("collection-a", "model-a")),
                5
        );

        assertThat(retrieverService.retrieveCalls()).isEqualTo(1);
        assertThat(retrieverService.vectorCalls()).isZero();
        assertThat(chunks).hasSize(1);
    }

    private static SearchContext searchContext(String query, EmbeddingService embeddingService) {
        return SearchContext.builder()
                .originalQuestion(query)
                .rewrittenQuestion(query)
                .topK(5)
                .queryEmbeddingContext(QueryEmbeddingContext.create(query, query, "request-id", "trace-id", embeddingService))
                .build();
    }

    private static SearchTarget target(String collectionName, String embeddingModelId) {
        return SearchTarget.builder()
                .channelName("test")
                .targetId(collectionName)
                .targetName(collectionName)
                .collectionName(collectionName)
                .embeddingModelId(embeddingModelId)
                .query("如何办理入职")
                .build();
    }

    private static KnowledgeBaseDO kb(String id, String name, String collectionName, String embeddingModel) {
        return KnowledgeBaseDO.builder()
                .id(id)
                .name(name)
                .collectionName(collectionName)
                .embeddingModel(embeddingModel)
                .deleted(0)
                .build();
    }

    private static final class CountingEmbeddingService implements EmbeddingService {

        private static final String DEFAULT_MODEL = "__default__";

        private final Map<String, AtomicInteger> calls = new ConcurrentHashMap<>();

        @Override
        public List<Float> embed(String text) {
            return embedWithMetadata(text).getEmbedding();
        }

        @Override
        public EmbeddingResult embedWithMetadata(String text) {
            return embeddingResult(DEFAULT_MODEL);
        }

        @Override
        public List<Float> embed(String text, String modelId) {
            return embedWithMetadata(text, modelId).getEmbedding();
        }

        @Override
        public EmbeddingResult embedWithMetadata(String text, String modelId) {
            return embeddingResult(modelId);
        }

        @Override
        public List<List<Float>> embedBatch(List<String> texts) {
            throw new UnsupportedOperationException();
        }

        @Override
        public List<List<Float>> embedBatch(List<String> texts, String modelId) {
            throw new UnsupportedOperationException();
        }

        int totalCalls() {
            return calls.values().stream().mapToInt(AtomicInteger::get).sum();
        }

        int callsFor(String modelId) {
            AtomicInteger count = calls.get(modelId);
            return count == null ? 0 : count.get();
        }

        private EmbeddingResult embeddingResult(String modelId) {
            calls.computeIfAbsent(modelId, ignored -> new AtomicInteger()).incrementAndGet();
            return EmbeddingResult.builder()
                    .embedding(List.of(3.0f, 4.0f))
                    .embeddingModelId(modelId)
                    .embeddingProvider("test-provider")
                    .embeddingDimension(2)
                    .build();
        }
    }

    private static final class RecordingRetrieverService implements RetrieverService {

        private final AtomicInteger retrieveCalls = new AtomicInteger();
        private final AtomicInteger vectorCalls = new AtomicInteger();

        @Override
        public List<RetrievedChunk> retrieve(RetrieveRequest retrieveParam) {
            retrieveCalls.incrementAndGet();
            return List.of(chunk("old", retrieveParam.getCollectionName()));
        }

        @Override
        public List<RetrievedChunk> retrieveByVector(float[] vector, RetrieveRequest retrieveParam) {
            vectorCalls.incrementAndGet();
            return List.of(chunk("vector", retrieveParam.getCollectionName()));
        }

        int retrieveCalls() {
            return retrieveCalls.get();
        }

        int vectorCalls() {
            return vectorCalls.get();
        }

        private RetrievedChunk chunk(String prefix, String collectionName) {
            return RetrievedChunk.builder()
                    .id(prefix + "-" + collectionName)
                    .text(prefix + "-" + collectionName)
                    .score(1.0f)
                    .collectionName(collectionName)
                    .build();
        }
    }
}
