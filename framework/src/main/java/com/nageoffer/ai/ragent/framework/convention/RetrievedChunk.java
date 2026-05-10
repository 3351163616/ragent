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

package com.nageoffer.ai.ragent.framework.convention;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.HashMap;
import java.util.Map;

/**
 * RAG 检索命中结果
 * <p>
 * 表示一次向量检索或相关性搜索命中的单条记录
 * 包含原始文档片段、来源信息、主键以及相关性得分
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder(toBuilder = true)
public class RetrievedChunk {

    /**
     * 命中记录的唯一标识
     * 比如向量库中的 primary key 或文档 id
     */
    private String id;

    /**
     * 命中的文本内容
     * 一般是被切分后的文档片段或段落
     */
    private String text;

    /**
     * 命中得分
     * 数值越大表示与查询的相关性越高
     */
    private Float score;

    /**
     * 所属文档 ID
     */
    private String docId;

    /**
     * 所属文档名称
     */
    private String docName;

    /**
     * 所属知识库 ID
     */
    private String kbId;

    /**
     * 所属知识库名称
     */
    private String kbName;

    /**
     * 所属向量集合名称
     */
    private String collectionName;

    /**
     * 文档来源地址或存储地址
     */
    private String sourceUrl;

    /**
     * 文档分块序号
     */
    private Integer chunkIndex;

    /**
     * 本轮回答中的引用编号
     */
    private Integer citationIndex;

    /**
     * 向量库返回的原始元数据
     */
    @Builder.Default
    private Map<String, Object> metadata = new HashMap<>();

    public RetrievedChunk(String id, String text, Float score) {
        this.id = id;
        this.text = text;
        this.score = score;
        this.metadata = new HashMap<>();
    }

    public RetrievedChunk withScore(Float nextScore) {
        return this.toBuilder()
                .score(nextScore)
                .metadata(this.metadata == null ? new HashMap<>() : new HashMap<>(this.metadata))
                .build();
    }
}
