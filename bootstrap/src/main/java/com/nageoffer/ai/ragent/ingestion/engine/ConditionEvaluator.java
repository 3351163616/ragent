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

package com.nageoffer.ai.ragent.ingestion.engine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nageoffer.ai.ragent.ingestion.domain.context.IngestionContext;
import com.nageoffer.ai.ragent.ingestion.domain.enums.SourceType;
import org.springframework.beans.BeanWrapperImpl;
import org.springframework.expression.ExpressionParser;
import org.springframework.expression.spel.standard.SpelExpressionParser;
import org.springframework.expression.spel.support.StandardEvaluationContext;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.util.List;
import java.util.Objects;

/**
 * 条件评估器
 * 用于根据给定的 IngestionContext 上下文和 JsonNode 格式的条件配置来评估条件是否满足
 */
@Component
public class ConditionEvaluator {

    private final ObjectMapper objectMapper;
    private final ExpressionParser parser = new SpelExpressionParser();

    public ConditionEvaluator(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public boolean evaluate(IngestionContext context, JsonNode condition) {
        if (condition == null || condition.isNull()) {
            return true;
        }
        if (condition.isBoolean()) {
            return condition.asBoolean();
        }
        if (condition.isTextual()) {
            return evalSpel(context, condition.asText());
        }
        if (condition.isObject()) {
            if (condition.has("all")) {
                return evalAll(context, condition.get("all"));
            }
            if (condition.has("any")) {
                return evalAny(context, condition.get("any"));
            }
            if (condition.has("not")) {
                return !evaluate(context, condition.get("not"));
            }
            if (condition.has("field")) {
                return evalRule(context, condition);
            }
        }
        return true;
    }

    private boolean evalAll(IngestionContext context, JsonNode node) {
        if (node == null || !node.isArray()) {
            return true;
        }
        for (JsonNode item : node) {
            if (!evaluate(context, item)) {
                return false;
            }
        }
        return true;
    }

    private boolean evalAny(IngestionContext context, JsonNode node) {
        if (node == null || !node.isArray()) {
            return true;
        }
        for (JsonNode item : node) {
            if (evaluate(context, item)) {
                return true;
            }
        }
        return false;
    }

    private boolean evalRule(IngestionContext context, JsonNode node) {
        String field = node.path("field").asText(null);
        if (!StringUtils.hasText(field)) {
            return true;
        }
        String operator = node.path("operator").asText(null);
        if (!StringUtils.hasText(operator)) {
            operator = node.path("op").asText("eq");
        }
        JsonNode valueNode = node.get("value");
        Object left = readField(context, field);
        Object right = valueNode == null ? null : objectMapper.convertValue(valueNode, Object.class);
        return compare(left, right, operator);
    }

    private Object readField(IngestionContext context, String path) {
        String normalizedPath = path.trim();
        Object aliasValue = readAliasField(context, normalizedPath);
        if (aliasValue != AliasField.UNMATCHED) {
            return aliasValue;
        }
        try {
            BeanWrapperImpl wrapper = new BeanWrapperImpl(context);
            return normalizeFieldValue(wrapper.getPropertyValue(normalizedPath));
        } catch (Exception e) {
            return null;
        }
    }

    private Object readAliasField(IngestionContext context, String path) {
        return switch (path) {
            case "source_type" -> context.getSource() == null || context.getSource().getType() == null
                    ? null
                    : context.getSource().getType().getValue();
            case "source_location" -> context.getSource() == null ? null : context.getSource().getLocation();
            case "file_name", "source_file_name" -> context.getSource() == null ? null : context.getSource().getFileName();
            case "mime_type" -> context.getMimeType();
            case "file_size", "raw_bytes_length" -> context.getRawBytes() == null ? null : context.getRawBytes().length;
            default -> AliasField.UNMATCHED;
        };
    }

    private Object normalizeFieldValue(Object value) {
        if (value instanceof SourceType sourceType) {
            return sourceType.getValue();
        }
        if (value instanceof Enum<?> enumValue) {
            return enumValue.name().toLowerCase();
        }
        return value;
    }

    private boolean compare(Object left, Object right, String operator) {
        return switch (operator.toLowerCase()) {
            case "ne" -> !Objects.equals(normalize(left), normalize(right));
            case "in" -> in(left, right);
            case "contains" -> contains(left, right);
            case "regex" -> regex(left, right);
            case "gt" -> compareNumber(left, right) > 0;
            case "gte" -> compareNumber(left, right) >= 0;
            case "lt" -> compareNumber(left, right) < 0;
            case "lte" -> compareNumber(left, right) <= 0;
            case "exists" -> left != null;
            case "not_exists" -> left == null;
            default -> Objects.equals(normalize(left), normalize(right));
        };
    }

    private boolean in(Object left, Object right) {
        Object normalizedLeft = normalize(left);
        Object normalizedRight = normalize(right);
        if (right instanceof List<?> list) {
            return list.stream().map(this::normalize).anyMatch(item -> Objects.equals(item, normalizedLeft));
        }
        if (left instanceof List<?> list) {
            return list.stream().map(this::normalize).anyMatch(item -> Objects.equals(item, normalizedRight));
        }
        return Objects.equals(normalizedLeft, normalizedRight);
    }

    private boolean contains(Object left, Object right) {
        if (left == null || right == null) {
            return false;
        }
        if (left instanceof String ls) {
            return ls.contains(String.valueOf(right));
        }
        if (left instanceof List<?> list) {
            Object normalizedRight = normalize(right);
            return list.stream().map(this::normalize).anyMatch(item -> Objects.equals(item, normalizedRight));
        }
        return false;
    }

    private boolean regex(Object left, Object right) {
        if (left == null || right == null) {
            return false;
        }
        return String.valueOf(left).matches(String.valueOf(right));
    }

    private int compareNumber(Object left, Object right) {
        if (left == null || right == null) {
            return 0;
        }
        Double l = toDouble(left);
        Double r = toDouble(right);
        if (l == null || r == null) {
            return 0;
        }
        return Double.compare(l, r);
    }

    private Double toDouble(Object value) {
        if (value instanceof Number n) {
            return n.doubleValue();
        }
        try {
            return Double.parseDouble(String.valueOf(value));
        } catch (Exception e) {
            return null;
        }
    }

    private Object normalize(Object value) {
        if (value instanceof String s) {
            return s.trim();
        }
        return normalizeFieldValue(value);
    }

    private enum AliasField {
        UNMATCHED
    }

    private boolean evalSpel(IngestionContext context, String expression) {
        try {
            StandardEvaluationContext ctx = new StandardEvaluationContext(context);
            ctx.setVariable("ctx", context);
            Boolean result = parser.parseExpression(expression).getValue(ctx, Boolean.class);
            return Boolean.TRUE.equals(result);
        } catch (Exception e) {
            return false;
        }
    }
}
