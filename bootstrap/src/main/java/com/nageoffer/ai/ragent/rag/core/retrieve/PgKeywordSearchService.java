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
import com.nageoffer.ai.ragent.rag.config.SearchChannelProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * PostgreSQL 关键词检索实现。
 * <p>
 * 组合 full text rank 与精确词/短语 boost，覆盖错误码、接口名、条款号、专有名词等向量容易漏召的场景。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class PgKeywordSearchService implements KeywordSearchService {

    private static final Pattern TOKEN_PATTERN = Pattern.compile("[\\p{IsAlphabetic}\\p{IsDigit}_./:#\\-]{2,}");

    private final JdbcTemplate jdbcTemplate;
    private final ObjectMapper objectMapper;
    private final SearchChannelProperties properties;

    @Override
    public List<RetrievedChunk> search(KeywordSearchRequest request) {
        if (request == null || !StringUtils.hasText(request.getQuery())) {
            return List.of();
        }

        SearchChannelProperties.KeywordEs keywordProperties = properties.getChannels().getKeywordEs();
        RetrievalSqlFilterBuilder.SqlFilter filter = RetrievalSqlFilterBuilder.build(request.getFilterContext());
        List<String> terms = extractTerms(request.getQuery(), keywordProperties.getMaxQueryTerms());
        List<Object> args = new ArrayList<>();
        args.add(keywordProperties.getTextSearchConfig());
        args.add(request.getQuery());
        args.add(keywordProperties.getTextSearchConfig());
        args.add(keywordProperties.getTextSearchConfig());
        args.add("%" + escapeLike(request.getQuery()) + "%");
        args.add(keywordProperties.getExactPhraseBoost());
        args.add("%" + escapeLike(request.getQuery()) + "%");
        args.add(keywordProperties.getTitleBoost());
        for (String term : terms) {
            args.add("%" + escapeLike(term) + "%");
            args.add("%" + escapeLike(term) + "%");
            args.add(keywordProperties.getTermBoost());
        }

        String collectionClause = "";
        if (StringUtils.hasText(request.getCollectionName())) {
            collectionClause = " AND kb.collection_name = ?";
            args.add(request.getCollectionName());
        }
        args.addAll(filter.args());
        args.add(keywordProperties.getMinScore());
        args.add(Math.max(1, request.getTopK()));

        // noinspection SqlDialectInspection,SqlNoDataSourceInspection
        String sql = """
                WITH q AS (
                    SELECT websearch_to_tsquery(?::regconfig, ?) AS tsq
                ),
                ranked AS (
                    SELECT
                        c.id,
                        c.content,
                        jsonb_build_object(
                            'doc_id', d.id,
                            'kb_id', kb.id,
                            'collection_name', kb.collection_name,
                            'chunk_index', c.chunk_index,
                            'search_type', 'keyword'
                        )::text AS metadata,
                        kb.collection_name,
                        d.id AS doc_id,
                        d.doc_name,
                        d.file_url,
                        d.source_location,
                        kb.id AS kb_id,
                        kb.name AS kb_name,
                        c.chunk_index,
                        (
                            ts_rank_cd(
                                setweight(to_tsvector(?::regconfig, coalesce(d.doc_name, '')), 'A') ||
                                setweight(to_tsvector(?::regconfig, coalesce(c.content, '')), 'B'),
                                q.tsq
                            )
                            + CASE WHEN lower(c.content) LIKE lower(?) ESCAPE '\\' THEN ? ELSE 0 END
                            + CASE WHEN lower(d.doc_name) LIKE lower(?) ESCAPE '\\' THEN ? ELSE 0 END
                            %s
                        )::real AS score
                    FROM t_knowledge_chunk c
                    JOIN t_knowledge_document d
                        ON d.id = c.doc_id
                        AND d.deleted = 0
                    JOIN t_knowledge_base kb
                        ON kb.id = c.kb_id
                        AND kb.deleted = 0
                    CROSS JOIN q
                    WHERE c.deleted = 0
                    %s
                    %s
                )
                SELECT *
                FROM ranked
                WHERE score >= ?
                ORDER BY score DESC, chunk_index ASC
                LIMIT ?
                """.formatted(termBoostExpression(terms), collectionClause, filter.whereClause());

        return jdbcTemplate.query(sql,
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
                args.toArray());
    }

    private List<String> extractTerms(String query, int maxTerms) {
        if (!StringUtils.hasText(query) || maxTerms <= 0) {
            return List.of();
        }
        List<String> terms = TOKEN_PATTERN.matcher(query)
                .results()
                .map(match -> match.group().trim())
                .filter(StringUtils::hasText)
                .distinct()
                .limit(maxTerms)
                .toList();
        return terms.isEmpty() ? List.of(query.trim()) : terms;
    }

    private String escapeLike(String value) {
        return value
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_");
    }

    private String termBoostExpression(List<String> terms) {
        if (terms == null || terms.isEmpty()) {
            return "";
        }
        return terms.stream()
                .map(ignored -> """
                        + CASE
                            WHEN lower(c.content) LIKE lower(?) ESCAPE '\\'
                              OR lower(d.doc_name) LIKE lower(?) ESCAPE '\\'
                            THEN ? ELSE 0
                          END
                        """)
                .collect(java.util.stream.Collectors.joining("\n"));
    }

    private Map<String, Object> parseMetadata(String metadata) {
        if (!StringUtils.hasText(metadata)) {
            return Map.of();
        }
        try {
            return objectMapper.readValue(metadata, new TypeReference<Map<String, Object>>() {
            });
        } catch (Exception e) {
            log.warn("解析关键词检索元数据失败: {}", metadata, e);
            return Map.of();
        }
    }

    private String resolveSourceUrl(String sourceLocation, String fileUrl) {
        if (StringUtils.hasText(sourceLocation)) {
            return sourceLocation;
        }
        return fileUrl;
    }
}
