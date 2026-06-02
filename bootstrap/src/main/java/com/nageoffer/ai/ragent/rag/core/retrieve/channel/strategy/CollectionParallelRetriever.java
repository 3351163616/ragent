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

package com.nageoffer.ai.ragent.rag.core.retrieve.channel.strategy;

import com.nageoffer.ai.ragent.framework.convention.RetrievedChunk;
import com.nageoffer.ai.ragent.rag.core.retrieve.QueryEmbedding;
import com.nageoffer.ai.ragent.rag.core.retrieve.RetrieveRequest;
import com.nageoffer.ai.ragent.rag.core.retrieve.RetrieverService;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.AbstractParallelRetriever;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchContext;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchTarget;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.concurrent.Executor;

/**
 * Collection 并行检索器
 * 继承模板类，实现 Collection 特定的检索逻辑
 */
@Slf4j
public class CollectionParallelRetriever extends AbstractParallelRetriever<SearchTarget> {

    private final RetrieverService retrieverService;

    public CollectionParallelRetriever(RetrieverService retrieverService, Executor executor) {
        super(executor);
        this.retrieverService = retrieverService;
    }

    @Override
    protected List<RetrievedChunk> createRetrievalTask(String question, SearchTarget target, int topK) {
        return createRetrievalTask(null, question, target, topK);
    }

    @Override
    protected List<RetrievedChunk> createRetrievalTask(SearchContext context,
                                                       String question,
                                                       SearchTarget target,
                                                       int topK) {
        String collectionName = target.getCollectionName();
        try {
            return retrieverService.retrieve(
                    RetrieveRequest.builder()
                            .collectionName(collectionName)
                            .query(question)
                            .topK(topK)
                            .filterContext(context == null ? null : context.getFilterContext())
                            .build()
            );
        } catch (Exception e) {
            log.error("在 collection {} 中检索失败，错误: {}", collectionName, e.getMessage(), e);
            return List.of();
        }
    }

    @Override
    protected List<RetrievedChunk> createRetrievalTask(String question,
                                                       SearchTarget target,
                                                       int topK,
                                                       QueryEmbedding queryEmbedding) {
        return createRetrievalTask(null, question, target, topK, queryEmbedding);
    }

    @Override
    protected List<RetrievedChunk> createRetrievalTask(SearchContext context,
                                                       String question,
                                                       SearchTarget target,
                                                       int topK,
                                                       QueryEmbedding queryEmbedding) {
        String collectionName = target.getCollectionName();
        try {
            return retrieverService.retrieveByVector(
                    queryEmbedding.getVector(),
                    RetrieveRequest.builder()
                            .collectionName(collectionName)
                            .query(question)
                            .topK(topK)
                            .filterContext(context == null ? null : context.getFilterContext())
                            .build()
            );
        } catch (Exception e) {
            log.error("在 collection {} 中检索失败，错误: {}", collectionName, e.getMessage(), e);
            return List.of();
        }
    }

    @Override
    protected String getEmbeddingModelId(SearchTarget target) {
        return target.getEmbeddingModelId();
    }

    @Override
    protected String getTargetQuery(String question, SearchTarget target) {
        return target.getQuery() == null ? question : target.getQuery();
    }

    @Override
    protected String getTargetIdentifier(SearchTarget target) {
        return "Collection: " + target.getCollectionName();
    }

    @Override
    protected String getStatisticsName() {
        return "全局检索";
    }
}
