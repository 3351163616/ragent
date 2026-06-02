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

import cn.hutool.core.collection.CollUtil;
import com.nageoffer.ai.ragent.rag.core.retrieve.channel.RetrievalFilterContext;

import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;

/**
 * PostgreSQL 检索过滤 SQL 片段构造器。
 */
public final class RetrievalSqlFilterBuilder {

    private RetrievalSqlFilterBuilder() {
    }

    public static SqlFilter build(RetrievalFilterContext filterContext) {
        if (filterContext == null) {
            return new SqlFilter("", List.of());
        }

        StringBuilder where = new StringBuilder();
        List<Object> args = new ArrayList<>();

        if (filterContext.isEnabledOnly()) {
            where.append(" AND COALESCE(d.enabled, 1) = 1 AND COALESCE(c.enabled, 1) = 1");
        }
        if (filterContext.isSuccessStatusOnly()) {
            where.append(" AND (d.status IS NULL OR d.status = 'success')");
        }
        if (filterContext.hasKbScope()) {
            appendInClause(where, args, "kb.id", filterContext.getAllowedKbIds());
        }
        if (filterContext.hasDocScope()) {
            appendInClause(where, args, "d.id", filterContext.getAllowedDocIds());
        }
        if (filterContext.shouldApplyUserOwnedFilter()) {
            where.append(" AND (kb.created_by = ? OR d.created_by = ?)");
            args.add(resolveOwner(filterContext));
            args.add(resolveOwner(filterContext));
        }
        if (filterContext.getUpdatedAfter() != null) {
            where.append(" AND d.update_time >= ?");
            args.add(Timestamp.from(filterContext.getUpdatedAfter()));
        }
        if (filterContext.getUpdatedBefore() != null) {
            where.append(" AND d.update_time <= ?");
            args.add(Timestamp.from(filterContext.getUpdatedBefore()));
        }

        return new SqlFilter(where.toString(), args);
    }

    public static boolean hasCollectionScope(RetrievalFilterContext filterContext) {
        return filterContext != null && CollUtil.isNotEmpty(filterContext.getAllowedKbIds());
    }

    private static String resolveOwner(RetrievalFilterContext filterContext) {
        if (filterContext.getUsername() != null && !filterContext.getUsername().isBlank()) {
            return filterContext.getUsername();
        }
        return filterContext.getUserId();
    }

    private static void appendInClause(StringBuilder where, List<Object> args, String column, List<String> values) {
        where.append(" AND ").append(column).append(" IN (");
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                where.append(", ");
            }
            where.append("?");
            args.add(values.get(i));
        }
        where.append(")");
    }

    public record SqlFilter(String whereClause, List<Object> args) {
    }
}
