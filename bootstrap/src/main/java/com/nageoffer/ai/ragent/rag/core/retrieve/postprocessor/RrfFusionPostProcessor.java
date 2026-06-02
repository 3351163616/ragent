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

package com.nageoffer.ai.ragent.rag.core.retrieve.postprocessor;

import cn.hutool.core.collection.CollUtil;
import com.nageoffer.ai.ragent.framework.convention.RetrievedChunk;
import com.nageoffer.ai.ragent.rag.config.SearchChannelProperties;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchChannelResult;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchChannelType;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchContext;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Reciprocal Rank Fusion 后置处理器。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class RrfFusionPostProcessor implements SearchResultPostProcessor {

    private final SearchChannelProperties properties;

    @Override
    public String getName() {
        return "RrfFusion";
    }

    @Override
    public int getOrder() {
        return 5;
    }

    @Override
    public boolean isEnabled(SearchContext context) {
        return properties.getFusion().isEnabled();
    }

    @Override
    public List<RetrievedChunk> process(List<RetrievedChunk> chunks,
                                        List<SearchChannelResult> results,
                                        SearchContext context) {
        if (CollUtil.isEmpty(chunks) || CollUtil.isEmpty(results)) {
            return chunks;
        }

        int rrfK = Math.max(1, properties.getFusion().getRrfK());
        Map<String, FusionAccumulator> accumulatorMap = new LinkedHashMap<>();
        Map<String, RetrievedChunk> chunkByKey = new HashMap<>();

        for (SearchChannelResult result : results) {
            if (result == null || CollUtil.isEmpty(result.getChunks())) {
                continue;
            }
            double weight = channelWeight(result);
            List<RetrievedChunk> channelChunks = result.getChunks();
            for (int i = 0; i < channelChunks.size(); i++) {
                RetrievedChunk chunk = channelChunks.get(i);
                String key = chunkKey(chunk);
                double contribution = weight / (rrfK + i + 1.0d);
                FusionAccumulator accumulator = accumulatorMap.computeIfAbsent(key, ignored -> new FusionAccumulator());
                accumulator.fusionScore += contribution;
                accumulator.channelHits++;
                accumulator.bestRawScore = Math.max(accumulator.bestRawScore, safeScore(chunk));
                Map<String, Object> source = new LinkedHashMap<>();
                source.put("channelType", result.getChannelType() == null ? null : result.getChannelType().name());
                source.put("rank", i + 1);
                source.put("rawScore", safeScore(chunk));
                source.put("weight", weight);
                source.put("rrfContribution", contribution);
                accumulator.sources.put(result.getChannelName(), source);
                chunkByKey.compute(key, (ignored, existing) -> chooseRepresentative(existing, chunk));
            }
        }

        int maxCandidates = properties.getFusion().getMaxCandidatesBeforeRerank();
        List<RetrievedChunk> fused = accumulatorMap.entrySet().stream()
                .map(entry -> applyFusion(chunkByKey.get(entry.getKey()), entry.getValue()))
                .filter(chunk -> chunk != null && containsChunk(chunks, chunkKey(chunk)))
                .sorted(Comparator.comparingDouble((RetrievedChunk chunk) -> safeScore(chunk)).reversed())
                .limit(maxCandidates > 0 ? maxCandidates : Long.MAX_VALUE)
                .toList();

        log.info("RRF 融合完成，输入候选: {}, 输出候选: {}, rrfK: {}",
                chunks.size(), fused.size(), rrfK);
        return fused;
    }

    private RetrievedChunk applyFusion(RetrievedChunk chunk, FusionAccumulator accumulator) {
        if (chunk == null) {
            return null;
        }
        Map<String, Object> metadata = chunk.getMetadata() == null ? new HashMap<>() : new HashMap<>(chunk.getMetadata());
        metadata.put("fusionStrategy", "RRF");
        metadata.put("fusionScore", accumulator.fusionScore);
        metadata.put("fusionChannelHits", accumulator.channelHits);
        metadata.put("fusionSources", accumulator.sources);
        metadata.put("bestRawScore", accumulator.bestRawScore);

        return chunk.toBuilder()
                .score((float) accumulator.fusionScore)
                .metadata(metadata)
                .build();
    }

    private RetrievedChunk chooseRepresentative(RetrievedChunk existing, RetrievedChunk candidate) {
        if (existing == null) {
            return candidate;
        }
        return safeScore(candidate) > safeScore(existing) ? candidate : existing;
    }

    private boolean containsChunk(List<RetrievedChunk> chunks, String key) {
        return chunks.stream().anyMatch(chunk -> chunkKey(chunk).equals(key));
    }

    private String chunkKey(RetrievedChunk chunk) {
        if (chunk.getId() != null && !chunk.getId().isBlank()) {
            return chunk.getId();
        }
        return String.valueOf(chunk.getText() == null ? 0 : chunk.getText().hashCode());
    }

    private double channelWeight(SearchChannelResult result) {
        Map<String, Double> weights = properties.getFusion().getChannelWeights();
        if (weights == null || weights.isEmpty()) {
            return defaultWeight(result.getChannelType());
        }
        String channelName = result.getChannelName();
        if (channelName != null) {
            Double byName = weights.get(channelName);
            if (byName == null) {
                byName = weights.get(channelName.toLowerCase(Locale.ROOT));
            }
            if (byName != null) {
                return byName;
            }
        }
        String typeKey = toConfigKey(result.getChannelType());
        Double byType = typeKey == null ? null : weights.get(typeKey);
        return byType == null ? defaultWeight(result.getChannelType()) : byType;
    }

    private double defaultWeight(SearchChannelType type) {
        if (type == SearchChannelType.INTENT_DIRECTED) {
            return 1.2d;
        }
        if (type == SearchChannelType.KEYWORD_ES) {
            return 1.1d;
        }
        return 1.0d;
    }

    private String toConfigKey(SearchChannelType type) {
        if (type == null) {
            return null;
        }
        return switch (type) {
            case INTENT_DIRECTED -> "intent-directed";
            case KEYWORD_ES -> "keyword-es";
            case VECTOR_GLOBAL -> "vector-global";
            case HYBRID -> "hybrid";
        };
    }

    private double safeScore(RetrievedChunk chunk) {
        return chunk == null || chunk.getScore() == null ? 0d : chunk.getScore();
    }

    private static final class FusionAccumulator {

        private double fusionScore;

        private int channelHits;

        private double bestRawScore = Double.NEGATIVE_INFINITY;

        private final Map<String, Object> sources = new LinkedHashMap<>();
    }
}
