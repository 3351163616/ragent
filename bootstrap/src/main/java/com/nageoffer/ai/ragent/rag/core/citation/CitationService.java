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

package com.nageoffer.ai.ragent.rag.core.citation;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.framework.convention.RetrievedChunk;
import com.nageoffer.ai.ragent.rag.config.RAGConfigProperties;
import com.nageoffer.ai.ragent.rag.dto.Citation;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 回答引用来源编排服务
 */
@Service
@RequiredArgsConstructor
public class CitationService {

    private static final int SNIPPET_MAX_LENGTH = 160;

    private final RAGConfigProperties ragConfigProperties;

    public boolean isEnabled() {
        return Boolean.TRUE.equals(ragConfigProperties.getAnswerCitationEnabled());
    }

    public CitationIndexAllocator newAllocator() {
        return new CitationIndexAllocator();
    }

    public void assignCitationIndexes(List<RetrievedChunk> chunks, CitationIndexAllocator allocator) {
        if (!isEnabled() || CollUtil.isEmpty(chunks) || allocator == null) {
            return;
        }
        for (RetrievedChunk chunk : chunks) {
            if (chunk == null) {
                continue;
            }
            chunk.setCitationIndex(allocator.indexFor(chunk));
        }
    }

    public String formatChunkForPrompt(RetrievedChunk chunk) {
        if (chunk == null) {
            return "";
        }
        if (!isEnabled() || chunk.getCitationIndex() == null) {
            return StrUtil.emptyIfNull(chunk.getText());
        }
        String source = resolveSourceLabel(chunk);
        return "[" + chunk.getCitationIndex() + "] " + source + "\n" + StrUtil.emptyIfNull(chunk.getText());
    }

    public String buildPromptInstruction() {
        if (!isEnabled()) {
            return "";
        }
        return """

                引用要求：
                - 知识库证据中若包含 [n] 编号，回答中涉及对应事实时必须在句末标注 [n]。
                - 可以组合多个来源，如 [1][2]。
                - 不要编造未出现在证据中的引用编号。
                """;
    }

    public List<Citation> buildCitations(Map<String, List<RetrievedChunk>> intentChunks) {
        if (!isEnabled() || intentChunks == null || intentChunks.isEmpty()) {
            return List.of();
        }
        Map<Integer, Citation> result = new LinkedHashMap<>();
        intentChunks.values().stream()
                .filter(Objects::nonNull)
                .flatMap(List::stream)
                .filter(Objects::nonNull)
                .filter(chunk -> chunk.getCitationIndex() != null)
                .sorted(Comparator.comparing(RetrievedChunk::getCitationIndex))
                .forEach(chunk -> result.putIfAbsent(chunk.getCitationIndex(), toCitation(chunk)));
        return List.copyOf(result.values());
    }

    private Citation toCitation(RetrievedChunk chunk) {
        return Citation.builder()
                .index(chunk.getCitationIndex())
                .chunkId(chunk.getId())
                .docId(chunk.getDocId())
                .docName(chunk.getDocName())
                .kbId(chunk.getKbId())
                .kbName(chunk.getKbName())
                .chunkIndex(chunk.getChunkIndex())
                .sourceUrl(chunk.getSourceUrl())
                .score(chunk.getScore())
                .snippet(abbreviate(chunk.getText()))
                .build();
    }

    private String resolveSourceLabel(RetrievedChunk chunk) {
        String docName = StrUtil.blankToDefault(chunk.getDocName(), "未知文档");
        StringBuilder label = new StringBuilder(docName);
        if (chunk.getChunkIndex() != null) {
            label.append("#chunk-").append(chunk.getChunkIndex());
        }
        if (StrUtil.isNotBlank(chunk.getKbName())) {
            label.append("（").append(chunk.getKbName()).append("）");
        }
        return label.toString();
    }

    private String abbreviate(String text) {
        if (StrUtil.isBlank(text)) {
            return "";
        }
        String compact = text.replaceAll("\\s+", " ").trim();
        if (compact.length() <= SNIPPET_MAX_LENGTH) {
            return compact;
        }
        return compact.substring(0, SNIPPET_MAX_LENGTH) + "...";
    }

    public static class CitationIndexAllocator {
        private final AtomicInteger counter = new AtomicInteger(1);
        private final Map<String, Integer> indexes = new ConcurrentHashMap<>();

        private int indexFor(RetrievedChunk chunk) {
            return indexes.computeIfAbsent(resolveKey(chunk), ignored -> counter.getAndIncrement());
        }

        private String resolveKey(RetrievedChunk chunk) {
            if (StrUtil.isNotBlank(chunk.getId())) {
                return "chunk:" + chunk.getId();
            }
            if (StrUtil.isNotBlank(chunk.getDocId()) && chunk.getChunkIndex() != null) {
                return "doc:" + chunk.getDocId() + "#" + chunk.getChunkIndex();
            }
            return "text:" + StrUtil.emptyIfNull(chunk.getText());
        }
    }
}
