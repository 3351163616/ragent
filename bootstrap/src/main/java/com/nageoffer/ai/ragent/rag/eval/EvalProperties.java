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

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * 评测模式配置
 * <p>
 * 用途：控制评测专用接口（/rag/eval/sync）和 AOP 切面是否启用
 * 生产环境默认 false，评测环境通过 -Dragent.eval.enabled=true 或独立 profile 开启
 */
@Data
@Component
@ConfigurationProperties(prefix = "app.eval")
public class EvalProperties {

    /**
     * 是否启用评测模式
     * <p>
     * false（默认）：EvalController 和 EvalRetrievalCaptureAspect 不注册，零运行时开销
     * true：注册评测接口和切面，用于评测项目调用
     */
    private boolean enabled = false;

    /**
     * 批量评测执行配置。
     */
    private Execution execution = new Execution();

    @Data
    public static class Execution {

        /**
         * 单个评测任务内最多同时执行多少个 trial，避免压垮模型、向量库和 MCP。
         */
        private int maxConcurrency = 1;

        /**
         * 未指定 trialCount 时的默认执行次数。
         */
        private int defaultTrialCount = 5;

        /**
         * 单次 trial 的建议超时时间；第一版用于运行记录和后续接入取消能力。
         */
        private int timeoutSeconds = 120;
    }
}
