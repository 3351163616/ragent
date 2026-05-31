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

package com.nageoffer.ai.ragent.rag.config;

import com.nageoffer.ai.ragent.rag.config.validation.ValidMemoryConfig;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import org.springframework.validation.annotation.Validated;

/**
 * 记忆配置属性类
 * 用于配置 RAG 系统中的对话记忆管理相关参数
 * 包括历史轮数保留、缓存时间、摘要压缩等功能的配置
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "rag.memory")
@Validated
@ValidMemoryConfig
public class MemoryProperties {

    /**
     * 保留原文的最近轮数（user+assistant 视为一轮）
     */
    @Min(1)
    @Max(1000)
    private Integer historyKeepTurns = 100;

    /**
     * 最近历史原文的 Token 预算。大上下文模型下优先保留原文，超过预算才从更早的消息开始裁剪。
     */
    @Min(1024)
    @Max(1000000)
    private Integer historyTokenBudget = 80000;

    /**
     * 是否启用对话记忆压缩
     */
    private Boolean summaryEnabled = false;

    /**
     * 开始考虑摘要的最小用户轮数。该配置仅作为低轮次保护，真正触发由 Token 阈值决定。
     */
    @Min(1)
    @Max(10000)
    private Integer summaryStartTurns = 20;

    /**
     * 首次摘要触发的绝对 Token 下限。
     */
    @Min(1024)
    @Max(1000000)
    private Integer summaryTriggerTokenThreshold = 200000;

    /**
     * 首次摘要触发的模型上下文比例阈值。
     */
    @DecimalMin("0.1")
    @DecimalMax("0.95")
    private Double summaryTriggerContextRatio = 0.7D;

    /**
     * 已存在摘要后，新增待压缩历史达到该 Token 数才增量更新摘要。
     */
    @Min(1024)
    @Max(1000000)
    private Integer summaryMinTokenThreshold = 20000;

    /**
     * 摘要更新的最短间隔，避免用户连续提问时频繁重写摘要破坏 Prompt Cache。
     */
    @Min(0)
    @Max(86400)
    private Integer summaryDebounceSeconds = 60;

    /**
     * 摘要最大字数
     */
    @Min(200)
    @Max(1000)
    private Integer summaryMaxChars = 200;

    /**
     * 会话标题最大长度（用于提示词约束）
     */
    @Min(10)
    @Max(100)
    private Integer titleMaxLength = 30;
}
