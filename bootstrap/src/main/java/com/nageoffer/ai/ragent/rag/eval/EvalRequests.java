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

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import lombok.Data;

import java.util.List;

/**
 * 自动化评测 API 请求模型集合。
 */
public final class EvalRequests {

    private EvalRequests() {
    }

    @Data
    public static class SuitePageRequest extends Page {

        private String keyword;
    }

    @Data
    public static class SuiteCreateRequest {

        private String name;

        private String description;

        private String version;
    }

    @Data
    public static class SuiteUpdateRequest {

        private String name;

        private String description;

        private String version;

        private Integer enabled;
    }

    @Data
    public static class CasePageRequest extends Page {

        private String suiteId;

        private String keyword;

        private String category;
    }

    @Data
    public static class CaseCreateRequest {

        private String suiteId;

        private String question;

        private String category;

        private Object groundTruth;

        private Integer enabled;
    }

    @Data
    public static class CaseUpdateRequest {

        private String question;

        private String category;

        private Object groundTruth;

        private Integer enabled;
    }

    @Data
    public static class RunCreateRequest {

        private String suiteId;

        private Integer trialCount;

        private List<String> caseIds;
    }

    @Data
    public static class RunPageRequest extends Page {

        private String suiteId;

        private String status;
    }
}
