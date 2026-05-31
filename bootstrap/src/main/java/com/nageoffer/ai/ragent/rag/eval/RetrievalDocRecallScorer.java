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
 * 文档级召回率阅卷器。
 */
@Component
public class RetrievalDocRecallScorer implements EvalScorer {

    @Override
    public List<EvalMetricResult> score(EvalCaseDO evalCase, EvalGroundTruth groundTruth, EvalSnapshot snapshot) {
        if (groundTruth == null || CollUtil.isEmpty(groundTruth.getReferenceDocIds())) {
            return List.of();
        }
        Set<String> expected = new LinkedHashSet<>(groundTruth.getReferenceDocIds());
        Set<String> actual = snapshot == null || CollUtil.isEmpty(snapshot.getRetrievedDocIds())
                ? Set.of()
                : new LinkedHashSet<>(snapshot.getRetrievedDocIds());
        long hits = expected.stream().filter(actual::contains).count();
        double score = expected.isEmpty() ? 0 : hits * 1.0d / expected.size();

        return List.of(EvalMetricResult.builder()
                .metricName("retrieval_doc_recall")
                .score(score)
                .passed(score >= 1.0d)
                .reason(score >= 1.0d ? "参考文档全部召回" : "参考文档未全部召回")
                .detail(Map.of("expected", expected, "actual", actual, "hits", hits))
                .build());
    }
}
