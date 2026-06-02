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

import java.util.List;
import java.util.Map;

/**
 * 单次评测执行快照，保存改写、意图、检索、MCP 与 Trace 结果。
 */
@Data
@Builder
public class EvalSnapshot {

    private String originalQuestion;

    private String rewrittenQuestion;

    private List<String> subQuestions;

    private List<String> intentLeafIds;

    private List<String> retrievedDocIds;

    private List<String> retrievedChunkIds;

    private List<String> retrievedContexts;

    private List<String> retrievedContextDocIds;

    private List<RetrievalChunkSnapshot> retrievedChunkDetails;

    private Map<String, Integer> retrievalChannelCounts;

    private String mcpContext;

    private List<String> mcpToolIds;

    private boolean hasMcp;

    private boolean hasKb;

    private String answer;

    private String traceId;

    private long latencyMs;

    @Data
    @Builder
    public static class RetrievalChunkSnapshot {

        private String chunkId;

        private String docId;

        private String docName;

        private String kbId;

        private String kbName;

        private String collectionName;

        private Integer chunkIndex;

        private Float score;

        private String channelType;

        private String channelName;

        private Double fusionScore;

        private Integer fusionChannelHits;

        private Object fusionSources;
    }
}
