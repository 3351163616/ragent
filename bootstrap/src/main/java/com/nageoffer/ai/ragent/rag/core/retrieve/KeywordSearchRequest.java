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

package com.nageoffer.ai.ragent.rag.core.retrieve;

import com.nageoffer.ai.ragent.rag.core.retrieve.channel.RetrievalFilterContext;
import lombok.Builder;
import lombok.Data;

/**
 * 关键词检索请求。
 */
@Data
@Builder
public class KeywordSearchRequest {

    private String query;

    @Builder.Default
    private int topK = 10;

    /**
     * 限定单个 collection。为空时全局检索。
     */
    private String collectionName;

    private RetrievalFilterContext filterContext;
}
