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

package com.nageoffer.ai.ragent.infra.chat.log;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * LLM 请求日志配置
 */
@Data
@Component
@ConfigurationProperties(prefix = "ai.llm-request-log")
public class LLMRequestLogProperties {

    /**
     * 是否记录所有 Chat LLM 请求
     */
    private boolean enabled = false;

    /**
     * 请求日志输出目录，相对路径会基于应用工作目录解析
     */
    private String directory = "logs/llm-requests";

    /**
     * 是否记录请求头
     */
    private boolean includeHeaders = true;

    /**
     * 是否格式化 JSON 文件
     */
    private boolean prettyPrint = true;
}
