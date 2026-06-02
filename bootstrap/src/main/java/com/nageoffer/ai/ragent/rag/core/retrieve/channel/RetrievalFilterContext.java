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
import lombok.Builder;
import lombok.Data;

import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 检索全局过滤上下文。
 * <p>
 * 这里承载权限、租户、版本、时间等硬约束。当前表结构只直接落地 user/kb/doc/time/status，
 * 其他字段作为扩展元数据保留，便于后续下沉到 ES、Milvus 或业务系统通道。
 */
@Data
@Builder
public class RetrievalFilterContext {

    private String userId;

    private String username;

    private String role;

    @Builder.Default
    private List<String> allowedKbIds = List.of();

    @Builder.Default
    private List<String> allowedDocIds = List.of();

    private Instant updatedAfter;

    private Instant updatedBefore;

    @Builder.Default
    private boolean enabledOnly = true;

    @Builder.Default
    private boolean successStatusOnly = true;

    @Builder.Default
    private boolean userOwnedOnly = false;

    @Builder.Default
    private boolean adminBypassUserOwned = true;

    private String tenantId;

    private String departmentId;

    private String locale;

    private String productLine;

    private String docVersion;

    @Builder.Default
    private Map<String, Object> attributes = new HashMap<>();

    public boolean hasKbScope() {
        return CollUtil.isNotEmpty(allowedKbIds);
    }

    public boolean hasDocScope() {
        return CollUtil.isNotEmpty(allowedDocIds);
    }

    public boolean shouldApplyUserOwnedFilter() {
        if (!userOwnedOnly || userId == null || userId.isBlank()) {
            return false;
        }
        return !(adminBypassUserOwned && "admin".equalsIgnoreCase(role));
    }
}
