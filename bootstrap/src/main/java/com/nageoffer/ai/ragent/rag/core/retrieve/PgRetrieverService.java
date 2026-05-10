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

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nageoffer.ai.ragent.framework.convention.RetrievedChunk;
import com.nageoffer.ai.ragent.infra.embedding.EmbeddingService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
@ConditionalOnProperty(name = "rag.vector.type", havingValue = "pg")
public class PgRetrieverService implements RetrieverService {

    private final JdbcTemplate jdbcTemplate;
    private final EmbeddingService embeddingService;
    private final ObjectMapper objectMapper;

    @Override
    public List<RetrievedChunk> retrieve(RetrieveRequest request) {
        List<Float> embedding = embeddingService.embed(request.getQuery());
        float[] vector = normalize(toArray(embedding));
        return retrieveByVector(vector, request);
    }

    @Override
    public List<RetrievedChunk> retrieveByVector(float[] vector, RetrieveRequest request) {
        // 设置ef_search提升召回率
        // noinspection SqlDialectInspection,SqlNoDataSourceInspection
        jdbcTemplate.execute("SET hnsw.ef_search = 200");

        String vectorLiteral = toVectorLiteral(vector);
        // noinspection SqlDialectInspection,SqlNoDataSourceInspection
        return jdbcTemplate.query("""
                        SELECT
                            v.id,
                            v.content,
                            v.metadata::text AS metadata,
                            1 - (v.embedding <=> ?::vector) AS score,
                            v.metadata->>'collection_name' AS collection_name,
                            d.id AS doc_id,
                            d.doc_name,
                            d.file_url,
                            d.source_location,
                            kb.id AS kb_id,
                            kb.name AS kb_name,
                            COALESCE(c.chunk_index, NULLIF(v.metadata->>'chunk_index', '')::int) AS chunk_index
                        FROM t_knowledge_vector v
                        LEFT JOIN t_knowledge_document d
                            ON d.id = v.metadata->>'doc_id'
                            AND d.deleted = 0
                        LEFT JOIN t_knowledge_base kb
                            ON kb.collection_name = v.metadata->>'collection_name'
                            AND kb.deleted = 0
                        LEFT JOIN t_knowledge_chunk c
                            ON c.id = v.id
                            AND c.deleted = 0
                        WHERE v.metadata->>'collection_name' = ?
                        ORDER BY v.embedding <=> ?::vector
                        LIMIT ?
                        """,
                (rs, rowNum) -> RetrievedChunk.builder()
                        .id(rs.getString("id"))
                        .text(rs.getString("content"))
                        .score(rs.getFloat("score"))
                        .metadata(parseMetadata(rs.getString("metadata")))
                        .collectionName(rs.getString("collection_name"))
                        .docId(rs.getString("doc_id"))
                        .docName(rs.getString("doc_name"))
                        .kbId(rs.getString("kb_id"))
                        .kbName(rs.getString("kb_name"))
                        .sourceUrl(resolveSourceUrl(rs.getString("source_location"), rs.getString("file_url")))
                        .chunkIndex((Integer) rs.getObject("chunk_index"))
                        .build(),
                vectorLiteral, request.getCollectionName(), vectorLiteral, request.getTopK()
        );
    }

    private Map<String, Object> parseMetadata(String metadata) {
        if (!StringUtils.hasText(metadata)) {
            return Map.of();
        }
        try {
            return objectMapper.readValue(metadata, new TypeReference<Map<String, Object>>() {
            });
        } catch (Exception e) {
            log.warn("解析向量元数据失败: {}", metadata, e);
            return Map.of();
        }
    }

    private String resolveSourceUrl(String sourceLocation, String fileUrl) {
        if (StringUtils.hasText(sourceLocation)) {
            return sourceLocation;
        }
        return fileUrl;
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

    private float[] toArray(List<Float> list) {
        float[] arr = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }

    private String toVectorLiteral(float[] embedding) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < embedding.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(embedding[i]);
        }
        return sb.append("]").toString();
    }
}
