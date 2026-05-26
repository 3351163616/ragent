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

package com.nageoffer.ai.ragent.rag.core.retrieve.channel;

import lombok.Builder;
import lombok.Data;
import org.springframework.util.StringUtils;

/**
 * 单个向量检索目标描述。
 */
@Data
@Builder
public class SearchTarget {

    /**
     * 来源检索通道。
     */
    private String channelName;

    /**
     * 业务目标 ID，如意图节点 ID。
     */
    private String targetId;

    /**
     * 业务目标名称。
     */
    private String targetName;

    /**
     * 向量集合名称。
     */
    private String collectionName;

    /**
     * 目标集合使用的 embedding 模型 ID。
     */
    private String embeddingModelId;

    /**
     * 目标专属 query。为空时使用 SearchContext mainQuestion。
     */
    private String query;

    public String displayName() {
        if (StringUtils.hasText(targetName)) {
            return targetName;
        }
        if (StringUtils.hasText(targetId)) {
            return targetId;
        }
        return StringUtils.hasText(collectionName) ? collectionName : "unknown";
    }
}
