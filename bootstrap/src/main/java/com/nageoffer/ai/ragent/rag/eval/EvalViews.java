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

import lombok.Builder;
import lombok.Data;

import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 * 自动化评测 API 视图模型集合。
 */
public final class EvalViews {

    private EvalViews() {
    }

    @Data
    @Builder
    public static class SuiteVO {

        private String id;

        private String name;

        private String description;

        private String version;

        private Integer enabled;

        private Date createTime;

        private Date updateTime;
    }

    @Data
    @Builder
    public static class CaseVO {

        private String id;

        private String suiteId;

        private String question;

        private String category;

        private String groundTruth;

        private Integer enabled;

        private Date createTime;

        private Date updateTime;
    }

    @Data
    @Builder
    public static class RunVO {

        private String id;

        private String suiteId;

        private Integer trialCount;

        private String status;

        private Integer totalCases;

        private Integer totalTrials;

        private Integer completedTrials;

        private Integer passedTrials;

        private Double successRate;

        private Double passAtK;

        private Date startedAt;

        private Date finishedAt;

        private Long durationMs;

        private String errorMessage;
    }

    @Data
    @Builder
    public static class TrialVO {

        private String id;

        private String runId;

        private String suiteId;

        private String caseId;

        private String caseQuestion;

        private String caseCategory;

        private Integer trialIndex;

        private String status;

        private Integer success;

        private String traceId;

        private Long latencyMs;

        private Date startedAt;

        private Date finishedAt;

        private Long durationMs;

        private String snapshotJson;

        private String errorMessage;

        private List<ScoreVO> scores;
    }

    @Data
    @Builder
    public static class ScoreVO {

        private String id;

        private String runId;

        private String trialId;

        private String suiteId;

        private String caseId;

        private String metricName;

        private Double scoreValue;

        private Integer passed;

        private String reason;

        private String detailJson;
    }

    @Data
    @Builder
    public static class RunDetailVO {

        private RunVO run;

        private List<TrialVO> trials;

        private Map<String, Object> report;
    }
}
