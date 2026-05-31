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
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.rag.dao.entity.EvalCaseDO;
import org.springframework.stereotype.Component;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * chunk 上下文文档来源精度阅卷器。
 */
@Component
public class RetrievalContextPrecisionScorer implements EvalScorer {

    @Override
    public List<EvalMetricResult> score(EvalCaseDO evalCase, EvalGroundTruth groundTruth, EvalSnapshot snapshot) {
        if (groundTruth == null || CollUtil.isEmpty(groundTruth.getReferenceDocIds())) {
            return List.of();
        }
        List<String> contextDocIds = snapshot == null ? List.of() : snapshot.getRetrievedContextDocIds();
        if (CollUtil.isEmpty(contextDocIds)) {
            return List.of(EvalMetricResult.builder()
                    .metricName("retrieval_context_precision")
                    .score(0)
                    .passed(false)
                    .reason("未召回可计算来源的上下文")
                    .detail(Map.of("expected", groundTruth.getReferenceDocIds(), "actual", List.of()))
                    .build());
        }

        Set<String> expected = new LinkedHashSet<>(groundTruth.getReferenceDocIds());
        List<String> nonBlankActual = contextDocIds.stream()
                .filter(StrUtil::isNotBlank)
                .toList();
        long relevant = nonBlankActual.stream()
                .filter(Objects::nonNull)
                .filter(expected::contains)
                .count();
        double score = nonBlankActual.isEmpty() ? 0 : relevant * 1.0d / nonBlankActual.size();

        return List.of(EvalMetricResult.builder()
                .metricName("retrieval_context_precision")
                .score(score)
                .passed(score >= 0.5d)
                .reason(score >= 0.5d ? "召回上下文多数来自参考文档" : "召回上下文参考文档占比偏低")
                .detail(Map.of("expected", expected, "actual", nonBlankActual, "relevant", relevant))
                .build());
    }
}
