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

import cn.hutool.core.collection.CollUtil;
import com.nageoffer.ai.ragent.framework.convention.RetrievedChunk;
import com.nageoffer.ai.ragent.rag.config.SearchChannelProperties;
import com.nageoffer.ai.ragent.rag.core.retrieve.KeywordSearchRequest;
import com.nageoffer.ai.ragent.rag.core.retrieve.KeywordSearchService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.util.List;
import java.util.Map;

/**
 * 关键词/BM25 检索通道。
 * <p>
 * 类型沿用 KEYWORD_ES，当前默认由 PostgreSQL full text 实现，后续可替换为 ES/OpenSearch。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class KeywordESSearchChannel implements SearchChannel {

    private final KeywordSearchService keywordSearchService;
    private final SearchChannelProperties properties;

    @Override
    public String getName() {
        return "KeywordESSearch";
    }

    @Override
    public int getPriority() {
        return 5;
    }

    @Override
    public boolean isEnabled(SearchContext context) {
        return properties.getChannels().getKeywordEs().isEnabled()
                && context != null
                && StringUtils.hasText(context.getMainQuestion());
    }

    @Override
    public SearchChannelResult search(SearchContext context) {
        long startTime = System.currentTimeMillis();
        try {
            int topKMultiplier = properties.getChannels().getKeywordEs().getTopKMultiplier();
            int topK = context.getTopK() * Math.max(1, topKMultiplier);
            List<RetrievedChunk> chunks = keywordSearchService.search(KeywordSearchRequest.builder()
                    .query(context.getMainQuestion())
                    .topK(topK)
                    .filterContext(context.getFilterContext())
                    .build());
            annotate(chunks);

            long latency = System.currentTimeMillis() - startTime;
            log.info("关键词检索完成，检索到 {} 个 Chunk，耗时 {}ms", chunks.size(), latency);
            return SearchChannelResult.builder()
                    .channelType(SearchChannelType.KEYWORD_ES)
                    .channelName(getName())
                    .chunks(chunks)
                    .latencyMs(latency)
                    .metadata(Map.of("topK", topK))
                    .build();
        } catch (Exception e) {
            log.error("关键词检索失败", e);
            return SearchChannelResult.builder()
                    .channelType(SearchChannelType.KEYWORD_ES)
                    .channelName(getName())
                    .chunks(List.of())
                    .latencyMs(System.currentTimeMillis() - startTime)
                    .build();
        }
    }

    @Override
    public SearchChannelType getType() {
        return SearchChannelType.KEYWORD_ES;
    }

    private void annotate(List<RetrievedChunk> chunks) {
        if (CollUtil.isEmpty(chunks)) {
            return;
        }
        for (RetrievedChunk chunk : chunks) {
            if (chunk.getMetadata() != null) {
                chunk.getMetadata().put("channelType", SearchChannelType.KEYWORD_ES.name());
                chunk.getMetadata().put("channelName", getName());
            }
        }
    }
}
