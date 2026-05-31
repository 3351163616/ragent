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
 * MCP 工具选择阅卷器。
 */
@Component
public class McpToolScorer implements EvalScorer {

    @Override
    public List<EvalMetricResult> score(EvalCaseDO evalCase, EvalGroundTruth groundTruth, EvalSnapshot snapshot) {
        String expectedToolName = groundTruth == null || groundTruth.getExpectedTool() == null
                ? null
                : StrUtil.trimToNull(groundTruth.getExpectedTool().getName());
        if (StrUtil.isBlank(expectedToolName)) {
            return List.of();
        }
        List<String> actualToolIds = snapshot == null || CollUtil.isEmpty(snapshot.getMcpToolIds())
                ? List.of()
                : snapshot.getMcpToolIds();
        boolean passed = actualToolIds.contains(expectedToolName);

        return List.of(EvalMetricResult.builder()
                .metricName("mcp_tool_accuracy")
                .score(passed ? 1.0d : 0d)
                .passed(passed)
                .reason(passed ? "MCP 工具命中" : "MCP 工具未命中")
                .detail(Map.of("expected", expectedToolName, "actual", actualToolIds))
                .build());
    }
}
