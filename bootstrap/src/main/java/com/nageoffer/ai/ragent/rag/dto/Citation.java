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

package com.nageoffer.ai.ragent.rag.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 回答引用来源
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Citation {

    /**
     * 回答中的引用编号，对应模型输出里的 [n]
     */
    private Integer index;

    /**
     * 命中的 chunk ID
     */
    private String chunkId;

    /**
     * 文档 ID
     */
    private String docId;

    /**
     * 文档名称
     */
    private String docName;

    /**
     * 知识库 ID
     */
    private String kbId;

    /**
     * 知识库名称
     */
    private String kbName;

    /**
     * 分块序号
     */
    private Integer chunkIndex;

    /**
     * 来源地址
     */
    private String sourceUrl;

    /**
     * 检索得分
     */
    private Float score;

    /**
     * 展示用文本片段
     */
    private String snippet;
}
