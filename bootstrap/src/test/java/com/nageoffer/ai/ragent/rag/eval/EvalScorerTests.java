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

import java.util.List;

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
                .retrievedContextDocIds(List.of("FAQ_A", "FAQ_A", "FAQ_B", null))
                .build();

        EvalMetricResult result = new RetrievalContextPrecisionScorer().score(null, groundTruth, snapshot).get(0);

        assertEquals("retrieval_context_precision", result.getMetricName());
        assertEquals(2.0d / 3.0d, result.getScore());
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
}
