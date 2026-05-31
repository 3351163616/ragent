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

import java.util.List;
import java.util.Map;

/**
 * 基于 requiredFacts / forbiddenClaims 的答案规则阅卷器。
 */
@Component
public class AnswerRuleScorer implements EvalScorer {

    @Override
    public List<EvalMetricResult> score(EvalCaseDO evalCase, EvalGroundTruth groundTruth, EvalSnapshot snapshot) {
        if (groundTruth == null
                || (CollUtil.isEmpty(groundTruth.getRequiredFacts()) && CollUtil.isEmpty(groundTruth.getForbiddenClaims()))) {
            return List.of();
        }
        String answer = snapshot == null ? null : snapshot.getAnswer();
        if (StrUtil.isBlank(answer)) {
            return List.of(EvalMetricResult.builder()
                    .metricName("answer_rule")
                    .score(0)
                    .passed(false)
                    .reason("当前评测快照尚未生成最终答案")
                    .detail(Map.of(
                            "requiredFacts", safeList(groundTruth.getRequiredFacts()),
                            "forbiddenClaims", safeList(groundTruth.getForbiddenClaims())))
                    .build());
        }

        long requiredHits = safeList(groundTruth.getRequiredFacts()).stream()
                .filter(answer::contains)
                .count();
        long forbiddenHits = safeList(groundTruth.getForbiddenClaims()).stream()
                .filter(answer::contains)
                .count();
        int requiredSize = safeList(groundTruth.getRequiredFacts()).size();
        double requiredScore = requiredSize == 0 ? 1.0d : requiredHits * 1.0d / requiredSize;
        boolean passed = requiredScore >= 1.0d && forbiddenHits == 0;

        return List.of(EvalMetricResult.builder()
                .metricName("answer_rule")
                .score(forbiddenHits == 0 ? requiredScore : 0d)
                .passed(passed)
                .reason(passed ? "答案规则全部满足" : "答案缺少必要事实或包含禁用断言")
                .detail(Map.of(
                        "requiredFacts", safeList(groundTruth.getRequiredFacts()),
                        "requiredHits", requiredHits,
                        "forbiddenClaims", safeList(groundTruth.getForbiddenClaims()),
                        "forbiddenHits", forbiddenHits))
                .build());
    }

    private List<String> safeList(List<String> input) {
        return input == null ? List.of() : input;
    }
}
