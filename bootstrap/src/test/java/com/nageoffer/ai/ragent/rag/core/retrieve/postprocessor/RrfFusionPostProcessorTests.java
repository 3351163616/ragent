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

package com.nageoffer.ai.ragent.rag.core.retrieve.postprocessor;

import com.nageoffer.ai.ragent.framework.convention.RetrievedChunk;
import com.nageoffer.ai.ragent.rag.config.SearchChannelProperties;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchChannelResult;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchChannelType;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.SearchContext;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class RrfFusionPostProcessorTests {

    @Test
    void duplicatedChunkGetsFusionMetadataAndScore() {
        SearchChannelProperties properties = new SearchChannelProperties();
        RrfFusionPostProcessor processor = new RrfFusionPostProcessor(properties);
        SearchChannelResult keyword = result(SearchChannelType.KEYWORD_ES, "KeywordESSearch",
                List.of(chunk("chunk-a", 12f), chunk("chunk-b", 9f)));
        SearchChannelResult vector = result(SearchChannelType.VECTOR_GLOBAL, "VectorGlobalSearch",
                List.of(chunk("chunk-a", 0.8f), chunk("chunk-c", 0.7f)));

        List<RetrievedChunk> fused = processor.process(
                List.of(chunk("chunk-a", 12f), chunk("chunk-b", 9f), chunk("chunk-c", 0.7f)),
                List.of(keyword, vector),
                SearchContext.builder().topK(5).build()
        );

        assertThat(fused).hasSize(3);
        RetrievedChunk top = fused.get(0);
        assertThat(top.getId()).isEqualTo("chunk-a");
        assertThat(top.getMetadata()).containsEntry("fusionStrategy", "RRF");
        assertThat(top.getMetadata()).containsEntry("fusionChannelHits", 2);
        assertThat(top.getScore()).isLessThan(1f);
    }

    private SearchChannelResult result(SearchChannelType type, String name, List<RetrievedChunk> chunks) {
        return SearchChannelResult.builder()
                .channelType(type)
                .channelName(name)
                .chunks(chunks)
                .build();
    }

    private RetrievedChunk chunk(String id, Float score) {
        return RetrievedChunk.builder()
                .id(id)
                .text(id)
                .score(score)
                .build();
    }
}
