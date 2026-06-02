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

package com.nageoffer.ai.ragent.rag.eval;

import cn.hutool.core.collection.CollUtil;
import com.nageoffer.ai.ragent.rag.dao.entity.EvalCaseDO;
import org.springframework.stereotype.Component;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 按通道统计参考文档召回贡献，用于定位漏召来自哪一路。
 */
@Component
public class RetrievalChannelRecallScorer implements EvalScorer {

    @Override
    public List<EvalMetricResult> score(EvalCaseDO evalCase, EvalGroundTruth groundTruth, EvalSnapshot snapshot) {
        if (groundTruth == null
                || CollUtil.isEmpty(groundTruth.getReferenceDocIds())
                || snapshot == null
                || CollUtil.isEmpty(snapshot.getRetrievedChunkDetails())) {
            return List.of();
        }

        Set<String> expected = new LinkedHashSet<>(groundTruth.getReferenceDocIds());
        Map<String, Set<String>> docsByChannel = new java.util.LinkedHashMap<>();
        for (EvalSnapshot.RetrievalChunkSnapshot detailSnapshot : snapshot.getRetrievedChunkDetails()) {
            if (detailSnapshot.getDocId() == null || detailSnapshot.getDocId().isBlank()) {
                continue;
            }
            Set<String> channelTypes = channelTypes(detailSnapshot);
            if (channelTypes.isEmpty()) {
                channelTypes.add("UNKNOWN");
            }
            for (String channelType : channelTypes) {
                docsByChannel
                        .computeIfAbsent(channelType, ignored -> new LinkedHashSet<>())
                        .add(detailSnapshot.getDocId());
            }
        }

        Map<String, Object> detail = new java.util.LinkedHashMap<>();
        detail.put("expected", expected);
        detail.put("channelDocs", docsByChannel);
        detail.put("channelCounts", snapshot.getRetrievalChannelCounts());

        long bestHits = docsByChannel.values().stream()
                .mapToLong(docs -> expected.stream().filter(docs::contains).count())
                .max()
                .orElse(0L);
        double score = expected.isEmpty() ? 0d : bestHits * 1.0d / expected.size();

        return List.of(EvalMetricResult.builder()
                .metricName("retrieval_channel_recall")
                .score(score)
                .passed(score >= 1.0d)
                .reason(score >= 1.0d ? "至少一路检索通道召回全部参考文档" : "没有单一路检索通道召回全部参考文档")
                .detail(detail)
                .build());
    }

    @SuppressWarnings("unchecked")
    private Set<String> channelTypes(EvalSnapshot.RetrievalChunkSnapshot detail) {
        Set<String> channelTypes = new LinkedHashSet<>();
        if (detail.getChannelType() != null && !detail.getChannelType().isBlank()) {
            channelTypes.add(detail.getChannelType());
        }
        if (detail.getFusionSources() instanceof Map<?, ?> sources) {
            for (Object value : sources.values()) {
                if (value instanceof Map<?, ?> source) {
                    Object channelType = source.get("channelType");
                    if (channelType != null && !channelType.toString().isBlank()) {
                        channelTypes.add(channelType.toString());
                    }
                }
            }
        }
        return channelTypes;
    }
}
