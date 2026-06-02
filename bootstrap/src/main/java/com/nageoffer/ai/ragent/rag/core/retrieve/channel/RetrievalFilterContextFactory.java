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
import com.nageoffer.ai.ragent.framework.context.UserContext;
import com.nageoffer.ai.ragent.rag.config.SearchChannelProperties;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;

/**
 * 从登录态与检索配置构造全局过滤上下文。
 */
@Component
@RequiredArgsConstructor
public class RetrievalFilterContextFactory {

    private final SearchChannelProperties properties;

    public RetrievalFilterContext create() {
        SearchChannelProperties.Filters filters = properties.getFilters();
        if (!filters.isEnabled()) {
            return RetrievalFilterContext.builder()
                    .enabledOnly(false)
                    .successStatusOnly(false)
                    .build();
        }

        return RetrievalFilterContext.builder()
                .userId(UserContext.getUserId())
                .username(UserContext.getUsername())
                .role(UserContext.getRole())
                .allowedKbIds(copy(filters.getAllowedKbIds()))
                .allowedDocIds(copy(filters.getAllowedDocIds()))
                .updatedAfter(resolveUpdatedAfter(filters.getRecentDays()))
                .enabledOnly(filters.isEnabledOnly())
                .successStatusOnly(filters.isSuccessStatusOnly())
                .userOwnedOnly(filters.isUserOwnedOnly())
                .adminBypassUserOwned(filters.isAdminBypassUserOwned())
                .build();
    }

    private Instant resolveUpdatedAfter(int recentDays) {
        if (recentDays <= 0) {
            return null;
        }
        return Instant.now().minus(recentDays, ChronoUnit.DAYS);
    }

    private List<String> copy(List<String> input) {
        if (CollUtil.isEmpty(input)) {
            return List.of();
        }
        return input.stream()
                .filter(value -> value != null && !value.isBlank())
                .distinct()
                .toList();
    }
}
