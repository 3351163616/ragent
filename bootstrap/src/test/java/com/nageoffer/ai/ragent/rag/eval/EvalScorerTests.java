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

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EvalScorerTests {

    @Test
    void intentAccuracyRequiresAllExpectedLeafIds() {
        EvalGroundTruth groundTruth = new EvalGroundTruth();
        groundTruth.setExpectedIntentLeafIds(List.of("intent-a", "intent-b"));
        EvalSnapshot snapshot = EvalSnapshot.builder()
                .intentLeafIds(List.of("intent-a", "intent-c"))
                .build();

        EvalMetricResult result = new IntentAccuracyScorer().score(null, groundTruth, snapshot).get(0);

        assertEquals("intent_accuracy", result.getMetricName());
        assertEquals(0.5d, result.getScore());
        assertFalse(result.isPassed());
    }

    @Test
    void docRecallPassesWhenAllReferenceDocsAreRetrieved() {
        EvalGroundTruth groundTruth = new EvalGroundTruth();
        groundTruth.setReferenceDocIds(List.of("FAQ_A", "FAQ_B"));
        EvalSnapshot snapshot = EvalSnapshot.builder()
                .retrievedDocIds(List.of("FAQ_B", "FAQ_A", "FAQ_C"))
                .build();

        EvalMetricResult result = new RetrievalDocRecallScorer().score(null, groundTruth, snapshot).get(0);

        assertEquals("retrieval_doc_recall", result.getMetricName());
        assertEquals(1.0d, result.getScore());
        assertTrue(result.isPassed());
    }

    @Test
    void contextPrecisionUsesChunkLevelDocIds() {
        EvalGroundTruth groundTruth = new EvalGroundTruth();
        groundTruth.setReferenceDocIds(List.of("FAQ_A"));
        EvalSnapshot snapshot = EvalSnapshot.builder()
                .retrievedContextDocIds(Arrays.asList("FAQ_A", "FAQ_A", "FAQ_B", null))
                .build();

        EvalMetricResult result = new RetrievalContextPrecisionScorer().score(null, groundTruth, snapshot).get(0);

        assertEquals("retrieval_context_precision", result.getMetricName());
        assertEquals(2.0d / 3.0d, result.getScore());
        assertTrue(result.isPassed());
    }

    @Test
    void channelRecallReportsBestSingleChannelCoverage() {
        EvalGroundTruth groundTruth = new EvalGroundTruth();
        groundTruth.setReferenceDocIds(List.of("FAQ_A", "FAQ_B"));
        EvalSnapshot snapshot = EvalSnapshot.builder()
                .retrievedChunkDetails(List.of(
                        EvalSnapshot.RetrievalChunkSnapshot.builder()
                                .docId("FAQ_A")
                                .channelType("KEYWORD_ES")
                                .build(),
                        EvalSnapshot.RetrievalChunkSnapshot.builder()
                                .docId("FAQ_B")
                                .channelType("VECTOR_GLOBAL")
                                .build()
                ))
                .build();

        EvalMetricResult result = new RetrievalChannelRecallScorer().score(null, groundTruth, snapshot).get(0);

        assertEquals("retrieval_channel_recall", result.getMetricName());
        assertEquals(0.5d, result.getScore());
        assertFalse(result.isPassed());
    }

    @Test
    void channelRecallUsesFusionSourcesWhenPresent() {
        EvalGroundTruth groundTruth = new EvalGroundTruth();
        groundTruth.setReferenceDocIds(List.of("FAQ_A"));
        EvalSnapshot snapshot = EvalSnapshot.builder()
                .retrievedChunkDetails(List.of(
                        EvalSnapshot.RetrievalChunkSnapshot.builder()
                                .docId("FAQ_A")
                                .channelType("VECTOR_GLOBAL")
                                .fusionSources(Map.of(
                                        "KeywordESSearch", Map.of("channelType", "KEYWORD_ES"),
                                        "VectorGlobalSearch", Map.of("channelType", "VECTOR_GLOBAL")
                                ))
                                .build()
                ))
                .build();

        EvalMetricResult result = new RetrievalChannelRecallScorer().score(null, groundTruth, snapshot).get(0);

        assertEquals(1.0d, result.getScore());
        assertTrue(result.isPassed());
    }

    @Test
    void mcpToolScorerComparesExpectedToolName() {
        EvalGroundTruth.ExpectedTool expectedTool = new EvalGroundTruth.ExpectedTool();
        expectedTool.setName("sales_query");
        EvalGroundTruth groundTruth = new EvalGroundTruth();
        groundTruth.setExpectedTool(expectedTool);
        EvalSnapshot snapshot = EvalSnapshot.builder()
                .mcpToolIds(List.of("weather_query", "sales_query"))
                .build();

        EvalMetricResult result = new McpToolScorer().score(null, groundTruth, snapshot).get(0);

        assertEquals("mcp_tool_accuracy", result.getMetricName());
        assertEquals(1.0d, result.getScore());
        assertTrue(result.isPassed());
    }

    @Test
    void answerRuleIgnoresFormattingWhitespaceAndPunctuation() {
        EvalGroundTruth groundTruth = new EvalGroundTruth();
        groundTruth.setRequiredFacts(List.of("3-4月", "员工 培训"));
        EvalSnapshot snapshot = EvalSnapshot.builder()
                .answer("年度计划一般在 **3～4月** 制定，并覆盖员工培训安排。")
                .build();

        EvalMetricResult result = new AnswerRuleScorer().score(null, groundTruth, snapshot).get(0);

        assertEquals("answer_rule", result.getMetricName());
        assertEquals(1.0d, result.getScore());
        assertTrue(result.isPassed());
    }

    @Test
    void answerRuleAppliesNormalizedForbiddenClaims() {
        EvalGroundTruth groundTruth = new EvalGroundTruth();
        groundTruth.setForbiddenClaims(List.of("公网直透内网服务"));
        EvalSnapshot snapshot = EvalSnapshot.builder()
                .answer("第三方回调可以**公网直透内网服务**。")
                .build();

        EvalMetricResult result = new AnswerRuleScorer().score(null, groundTruth, snapshot).get(0);

        assertEquals("answer_rule", result.getMetricName());
        assertEquals(0d, result.getScore());
        assertFalse(result.isPassed());
    }
}
