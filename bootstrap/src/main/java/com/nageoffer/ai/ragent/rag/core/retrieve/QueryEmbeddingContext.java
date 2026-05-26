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

import cn.hutool.core.collection.CollUtil;
import com.nageoffer.ai.ragent.infra.embedding.EmbeddingResult;
import com.nageoffer.ai.ragent.infra.embedding.EmbeddingService;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchTarget;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.util.StringUtils;

import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * 一次 RAG 检索内的 Query Embedding 复用上下文。
 * <p>
 * 缓存维度为 query + embeddingModelId，避免不同模型或特殊 query rewrite 之间误复用。
 */
@Slf4j
public class QueryEmbeddingContext {

    private static final String DEFAULT_MODEL_KEY = "__default__";
    private static final String KEY_SEPARATOR = "\u001F";

    @Getter
    private final String originalQuery;

    @Getter
    private final String rewrittenQuery;

    @Getter
    private final String requestId;

    @Getter
    private final String traceId;

    private final EmbeddingService embeddingService;
    private final ConcurrentMap<String, QueryEmbedding> embeddings = new ConcurrentHashMap<>();
    private final AtomicInteger generatedCount = new AtomicInteger();

    private QueryEmbeddingContext(String originalQuery,
                                  String rewrittenQuery,
                                  String requestId,
                                  String traceId,
                                  EmbeddingService embeddingService) {
        this.originalQuery = originalQuery;
        this.rewrittenQuery = rewrittenQuery;
        this.requestId = requestId;
        this.traceId = traceId;
        this.embeddingService = embeddingService;
    }

    public static QueryEmbeddingContext create(String originalQuery,
                                               String rewrittenQuery,
                                               String requestId,
                                               String traceId,
                                               EmbeddingService embeddingService) {
        return new QueryEmbeddingContext(originalQuery, rewrittenQuery, requestId, traceId, embeddingService);
    }

    /**
     * 按 target 所需的 query + embeddingModelId 预热 embedding。
     */
    public void preload(Collection<SearchTarget> targets) {
        if (CollUtil.isEmpty(targets)) {
            log.info("本轮检索未发现需要预生成 Query Embedding 的向量目标，requestId={}, traceId={}", requestId, traceId);
            return;
        }

        List<SearchTarget> distinctTargets = targets.stream()
                .filter(Objects::nonNull)
                .collect(Collectors.toMap(
                        target -> cacheKey(resolveQuery(target.getQuery()), target.getEmbeddingModelId()),
                        target -> target,
                        (left, right) -> left
                ))
                .values()
                .stream()
                .toList();

        log.info("开始预生成 Query Embedding，目标数={}, 去重后模型/Query组数={}, requestId={}, traceId={}",
                targets.size(), distinctTargets.size(), requestId, traceId);

        for (SearchTarget target : distinctTargets) {
            getOrCreate(
                    target.getQuery(),
                    target.getEmbeddingModelId(),
                    target.getChannelName(),
                    target.displayName()
            );
        }
    }

    /**
     * 获取或创建 Query Embedding。
     */
    public QueryEmbedding getOrCreate(String query,
                                      String embeddingModelId,
                                      String channelName,
                                      String targetIdentifier) {
        String effectiveQuery = resolveQuery(query);
        String cacheKey = cacheKey(effectiveQuery, embeddingModelId);

        AtomicBoolean created = new AtomicBoolean(false);
        QueryEmbedding embedding = embeddings.computeIfAbsent(cacheKey, ignored -> {
            created.set(true);
            return createEmbedding(effectiveQuery, embeddingModelId, channelName, targetIdentifier);
        });

        if (!created.get()) {
            log.info("复用 Query Embedding，channel={}, target={}, modelId={}, provider={}, dimension={}, requestId={}, traceId={}",
                    channelName,
                    targetIdentifier,
                    embedding.getEmbeddingModelId(),
                    embedding.getEmbeddingProvider(),
                    embedding.getEmbeddingDimension(),
                    requestId,
                    traceId);
        }
        return embedding;
    }

    public int generatedCount() {
        return generatedCount.get();
    }

    public int cachedGroupCount() {
        return embeddings.size();
    }

    public String describeEmbeddingGroups() {
        return embeddings.values().stream()
                .map(embedding -> String.format(
                        "modelId=%s, provider=%s, dimension=%s",
                        embedding.getEmbeddingModelId(),
                        embedding.getEmbeddingProvider(),
                        embedding.getEmbeddingDimension()
                ))
                .collect(Collectors.joining("; "));
    }

    private QueryEmbedding createEmbedding(String query,
                                           String embeddingModelId,
                                           String channelName,
                                           String targetIdentifier) {
        String requestedModel = modelKey(embeddingModelId);
        log.info("生成 Query Embedding，channel={}, target={}, requestedModelId={}, requestId={}, traceId={}",
                channelName, targetIdentifier, requestedModel, requestId, traceId);

        EmbeddingResult result = DEFAULT_MODEL_KEY.equals(requestedModel)
                ? embeddingService.embedWithMetadata(query)
                : embeddingService.embedWithMetadata(query, embeddingModelId);

        float[] vector = normalize(toArray(result.getEmbedding()));
        generatedCount.incrementAndGet();

        QueryEmbedding queryEmbedding = QueryEmbedding.builder()
                .query(query)
                .embeddingModelId(StringUtils.hasText(result.getEmbeddingModelId()) ? result.getEmbeddingModelId() : requestedModel)
                .embeddingProvider(result.getEmbeddingProvider())
                .embeddingDimension(result.getEmbeddingDimension() > 0 ? result.getEmbeddingDimension() : vector.length)
                .vector(vector)
                .build();

        log.info("Query Embedding 生成完成，channel={}, target={}, modelId={}, provider={}, dimension={}, generatedCount={}, requestId={}, traceId={}",
                channelName,
                targetIdentifier,
                queryEmbedding.getEmbeddingModelId(),
                queryEmbedding.getEmbeddingProvider(),
                queryEmbedding.getEmbeddingDimension(),
                generatedCount.get(),
                requestId,
                traceId);
        return queryEmbedding;
    }

    private String resolveQuery(String query) {
        if (StringUtils.hasText(query)) {
            return query;
        }
        if (StringUtils.hasText(rewrittenQuery)) {
            return rewrittenQuery;
        }
        return StringUtils.hasText(originalQuery) ? originalQuery : "";
    }

    private String cacheKey(String query, String embeddingModelId) {
        return modelKey(embeddingModelId) + KEY_SEPARATOR + query;
    }

    private String modelKey(String embeddingModelId) {
        return StringUtils.hasText(embeddingModelId) ? embeddingModelId : DEFAULT_MODEL_KEY;
    }

    private float[] toArray(List<Float> list) {
        if (CollUtil.isEmpty(list)) {
            throw new IllegalStateException("Embedding 结果为空");
        }
        float[] arr = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }

    private float[] normalize(float[] vector) {
        float norm = 0;
        for (float v : vector) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
        return vector;
    }
}
