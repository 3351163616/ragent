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

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import cn.hutool.core.util.StrUtil;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.nageoffer.ai.ragent.rag.config.RAGDefaultProperties;
import com.nageoffer.ai.ragent.framework.convention.RetrievedChunk;
import com.nageoffer.ai.ragent.infra.embedding.EmbeddingService;
import com.nageoffer.ai.ragent.knowledge.dao.entity.KnowledgeBaseDO;
import com.nageoffer.ai.ragent.knowledge.dao.entity.KnowledgeChunkDO;
import com.nageoffer.ai.ragent.knowledge.dao.entity.KnowledgeDocumentDO;
import com.nageoffer.ai.ragent.knowledge.dao.mapper.KnowledgeBaseMapper;
import com.nageoffer.ai.ragent.knowledge.dao.mapper.KnowledgeChunkMapper;
import com.nageoffer.ai.ragent.knowledge.dao.mapper.KnowledgeDocumentMapper;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.data.BaseVector;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.response.SearchResp;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
@ConditionalOnProperty(name = "rag.vector.type", havingValue = "milvus", matchIfMissing = true)
public class MilvusRetrieverService implements RetrieverService {

    private final EmbeddingService embeddingService;
    private final MilvusClientV2 milvusClient;
    private final RAGDefaultProperties ragDefaultProperties;
    private final KnowledgeDocumentMapper knowledgeDocumentMapper;
    private final KnowledgeBaseMapper knowledgeBaseMapper;
    private final KnowledgeChunkMapper knowledgeChunkMapper;

    @Override
    public List<RetrievedChunk> retrieve(RetrieveRequest retrieveParam) {
        List<Float> emb = embeddingService.embed(retrieveParam.getQuery());
        float[] vec = toArray(emb);

        float[] norm = normalize(vec);

        return retrieveByVector(norm, retrieveParam);
    }

    @Override
    public List<RetrievedChunk> retrieveByVector(float[] vector, RetrieveRequest retrieveParam) {
        List<BaseVector> vectors = List.of(new FloatVec(vector));

        Map<String, Object> params = new HashMap<>();
        params.put("metric_type", ragDefaultProperties.getMetricType());
        params.put("ef", 128);

        SearchReq req = SearchReq.builder()
                .collectionName(
                        StrUtil.isBlank(retrieveParam.getCollectionName()) ? ragDefaultProperties.getCollectionName() : retrieveParam.getCollectionName()
                )
                .annsField("embedding")
                .data(vectors)
                .topK(retrieveParam.getTopK())
                .searchParams(params)
                .outputFields(List.of("id", "content", "metadata"))
                .build();

        SearchResp resp = milvusClient.search(req);
        List<List<SearchResp.SearchResult>> results = resp.getSearchResults();

        if (results == null || results.isEmpty()) {
            return List.of();
        }

        // TODO 需确认后续是否对分数较低数据进行限制，限制多少合适？0.65？
        // TODO 如果本次查询分数都较高，是否应该扩大查询范围？1.5倍？
        List<RetrievedChunk> chunks = results.get(0).stream()
                .map(this::toRetrievedChunk)
                .collect(Collectors.toList());
        enrichSources(chunks, retrieveParam.getCollectionName());
        return chunks;
    }

    private RetrievedChunk toRetrievedChunk(SearchResp.SearchResult result) {
        Map<String, Object> entity = result.getEntity() == null ? Map.of() : result.getEntity();
        Map<String, Object> metadata = parseMetadata(entity.get("metadata"));
        return RetrievedChunk.builder()
                .id(Objects.toString(entity.get("id"), ""))
                .text(Objects.toString(entity.get("content"), ""))
                .score(result.getScore())
                .metadata(metadata)
                .collectionName(asString(metadata.get("collection_name")))
                .docId(asString(metadata.get("doc_id")))
                .chunkIndex(asInteger(metadata.get("chunk_index")))
                .build();
    }

    private Map<String, Object> parseMetadata(Object metadata) {
        if (metadata instanceof Map<?, ?> map) {
            Map<String, Object> result = new HashMap<>();
            map.forEach((key, value) -> result.put(Objects.toString(key, ""), value));
            return result;
        }
        if (metadata instanceof JsonObject jsonObject) {
            Map<String, Object> result = new HashMap<>();
            for (Map.Entry<String, JsonElement> entry : jsonObject.entrySet()) {
                JsonElement value = entry.getValue();
                if (value == null || value.isJsonNull()) {
                    result.put(entry.getKey(), null);
                } else if (value.isJsonPrimitive()) {
                    var primitive = value.getAsJsonPrimitive();
                    if (primitive.isBoolean()) {
                        result.put(entry.getKey(), primitive.getAsBoolean());
                    } else if (primitive.isNumber()) {
                        result.put(entry.getKey(), primitive.getAsNumber());
                    } else {
                        result.put(entry.getKey(), primitive.getAsString());
                    }
                } else {
                    result.put(entry.getKey(), value.toString());
                }
            }
            return result;
        }
        return Map.of();
    }

    private void enrichSources(List<RetrievedChunk> chunks, String fallbackCollectionName) {
        if (chunks == null || chunks.isEmpty()) {
            return;
        }
        Set<String> docIds = chunks.stream()
                .map(RetrievedChunk::getDocId)
                .filter(StrUtil::isNotBlank)
                .collect(Collectors.toSet());
        Set<String> chunkIds = chunks.stream()
                .map(RetrievedChunk::getId)
                .filter(StrUtil::isNotBlank)
                .collect(Collectors.toSet());

        Map<String, KnowledgeDocumentDO> docsById = docIds.isEmpty()
                ? Map.of()
                : knowledgeDocumentMapper.selectList(Wrappers.lambdaQuery(KnowledgeDocumentDO.class)
                        .in(KnowledgeDocumentDO::getId, docIds)
                        .eq(KnowledgeDocumentDO::getDeleted, 0))
                .stream()
                .collect(Collectors.toMap(KnowledgeDocumentDO::getId, doc -> doc));

        Map<String, KnowledgeChunkDO> chunksById = chunkIds.isEmpty()
                ? Map.of()
                : knowledgeChunkMapper.selectList(Wrappers.lambdaQuery(KnowledgeChunkDO.class)
                        .in(KnowledgeChunkDO::getId, chunkIds)
                        .eq(KnowledgeChunkDO::getDeleted, 0))
                .stream()
                .collect(Collectors.toMap(KnowledgeChunkDO::getId, chunk -> chunk));

        Set<String> kbIds = docsById.values().stream()
                .map(KnowledgeDocumentDO::getKbId)
                .filter(StrUtil::isNotBlank)
                .collect(Collectors.toSet());
        String collectionName = StrUtil.isBlank(fallbackCollectionName)
                ? chunks.stream().map(RetrievedChunk::getCollectionName).filter(StrUtil::isNotBlank).findFirst().orElse(null)
                : fallbackCollectionName;

        Map<String, KnowledgeBaseDO> kbById = kbIds.isEmpty()
                ? Map.of()
                : knowledgeBaseMapper.selectList(Wrappers.lambdaQuery(KnowledgeBaseDO.class)
                        .in(KnowledgeBaseDO::getId, kbIds)
                        .eq(KnowledgeBaseDO::getDeleted, 0))
                .stream()
                .collect(Collectors.toMap(KnowledgeBaseDO::getId, kb -> kb));
        KnowledgeBaseDO kbByCollection = StrUtil.isBlank(collectionName)
                ? null
                : knowledgeBaseMapper.selectOne(Wrappers.lambdaQuery(KnowledgeBaseDO.class)
                        .eq(KnowledgeBaseDO::getCollectionName, collectionName)
                        .eq(KnowledgeBaseDO::getDeleted, 0)
                        .last("limit 1"));

        for (RetrievedChunk chunk : chunks) {
            KnowledgeDocumentDO doc = docsById.get(chunk.getDocId());
            if (doc != null) {
                chunk.setDocName(doc.getDocName());
                chunk.setKbId(doc.getKbId());
                chunk.setSourceUrl(resolveSourceUrl(doc.getSourceLocation(), doc.getFileUrl()));
            }
            KnowledgeChunkDO knowledgeChunk = chunksById.get(chunk.getId());
            if (knowledgeChunk != null && chunk.getChunkIndex() == null) {
                chunk.setChunkIndex(knowledgeChunk.getChunkIndex());
            }
            KnowledgeBaseDO kb = StrUtil.isNotBlank(chunk.getKbId()) ? kbById.get(chunk.getKbId()) : kbByCollection;
            if (kb != null) {
                chunk.setKbId(kb.getId());
                chunk.setKbName(kb.getName());
                chunk.setCollectionName(kb.getCollectionName());
            }
        }
    }

    private String resolveSourceUrl(String sourceLocation, String fileUrl) {
        if (StrUtil.isNotBlank(sourceLocation)) {
            return sourceLocation;
        }
        return fileUrl;
    }

    private String asString(Object value) {
        return value == null ? null : Objects.toString(value, null);
    }

    private Integer asInteger(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value == null) {
            return null;
        }
        try {
            return Integer.parseInt(value.toString());
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private static float[] toArray(List<Float> list) {
        float[] arr = new float[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return arr;
    }

    private static float[] normalize(float[] v) {
        double sum = 0.0;
        for (float x : v) sum += x * x;
        double len = Math.sqrt(sum);
        float[] nv = new float[v.length];
        for (int i = 0; i < v.length; i++) nv[i] = (float) (v[i] / len);
        return nv;
    }
}
