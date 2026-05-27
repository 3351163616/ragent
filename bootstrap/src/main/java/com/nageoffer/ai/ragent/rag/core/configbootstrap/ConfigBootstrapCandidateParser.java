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

package com.nageoffer.ai.ragent.rag.core.configbootstrap;

import cn.hutool.core.util.StrUtil;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.nageoffer.ai.ragent.infra.util.LLMResponseCleaner;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapSuggestion.IntentNodeSuggestion;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapSuggestion.TermMappingSuggestion;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

@Component
public class ConfigBootstrapCandidateParser {

    public ConfigBootstrapSuggestion parse(String raw) {
        if (StrUtil.isBlank(raw)) {
            return ConfigBootstrapSuggestion.empty();
        }
        String cleaned = LLMResponseCleaner.stripMarkdownCodeFence(raw);
        JsonElement root = JsonParser.parseString(cleaned);
        if (!root.isJsonObject()) {
            return ConfigBootstrapSuggestion.empty();
        }
        JsonObject object = root.getAsJsonObject();
        return new ConfigBootstrapSuggestion(
                parseTermMappings(object.getAsJsonArray("term_mappings")),
                parseIntentNodes(object.getAsJsonArray("intent_nodes"))
        );
    }

    private List<TermMappingSuggestion> parseTermMappings(JsonArray array) {
        if (array == null || array.isEmpty()) {
            return List.of();
        }
        List<TermMappingSuggestion> result = new ArrayList<>();
        for (JsonElement element : array) {
            if (!element.isJsonObject()) {
                continue;
            }
            JsonObject object = element.getAsJsonObject();
            String sourceTerm = readString(object, "sourceTerm", "source_term");
            String targetTerm = readString(object, "targetTerm", "target_term");
            if (StrUtil.isBlank(sourceTerm) || StrUtil.isBlank(targetTerm)
                    || sourceTerm.trim().equals(targetTerm.trim())) {
                continue;
            }
            result.add(new TermMappingSuggestion(
                    sourceTerm.trim(),
                    targetTerm.trim(),
                    readConfidence(object),
                    readRiskLevel(object),
                    readStringList(object, "evidence")
            ));
        }
        return result;
    }

    private List<IntentNodeSuggestion> parseIntentNodes(JsonArray array) {
        if (array == null || array.isEmpty()) {
            return List.of();
        }
        List<IntentNodeSuggestion> result = new ArrayList<>();
        for (JsonElement element : array) {
            if (!element.isJsonObject()) {
                continue;
            }
            JsonObject object = element.getAsJsonObject();
            String name = readString(object, "name");
            if (StrUtil.isBlank(name)) {
                continue;
            }
            result.add(new IntentNodeSuggestion(
                    readString(object, "intentCode", "intent_code"),
                    name.trim(),
                    readInteger(object, "level"),
                    readString(object, "parentCode", "parent_code"),
                    readString(object, "description"),
                    readStringList(object, "examples", "example_questions"),
                    readString(object, "kbId", "kb_id"),
                    readString(object, "collectionName", "collection_name"),
                    readInteger(object, "topK", "top_k"),
                    readInteger(object, "kind"),
                    readInteger(object, "sortOrder", "sort_order"),
                    readConfidence(object),
                    readRiskLevel(object),
                    readStringList(object, "evidence")
            ));
        }
        return result;
    }

    private String readString(JsonObject object, String... names) {
        for (String name : names) {
            JsonElement element = object.get(name);
            if (element != null && element.isJsonPrimitive()) {
                return element.getAsString();
            }
        }
        return null;
    }

    private Integer readInteger(JsonObject object, String... names) {
        for (String name : names) {
            JsonElement element = object.get(name);
            if (element != null && element.isJsonPrimitive() && element.getAsJsonPrimitive().isNumber()) {
                return element.getAsInt();
            }
        }
        return null;
    }

    private double readConfidence(JsonObject object) {
        JsonElement element = object.get("confidence");
        if (element == null || !element.isJsonPrimitive() || !element.getAsJsonPrimitive().isNumber()) {
            return 0.6D;
        }
        double confidence = element.getAsDouble();
        if (confidence < 0D) {
            return 0D;
        }
        if (confidence > 1D) {
            return 1D;
        }
        return confidence;
    }

    private String readRiskLevel(JsonObject object) {
        String riskLevel = readString(object, "riskLevel", "risk_level");
        if (StrUtil.isBlank(riskLevel)) {
            return "MEDIUM";
        }
        String normalized = riskLevel.trim().toUpperCase(Locale.ROOT);
        return switch (normalized) {
            case "LOW", "MEDIUM", "HIGH" -> normalized;
            default -> "MEDIUM";
        };
    }

    private List<String> readStringList(JsonObject object, String... names) {
        for (String name : names) {
            JsonElement element = object.get(name);
            if (element == null || !element.isJsonArray()) {
                continue;
            }
            List<String> result = new ArrayList<>();
            for (JsonElement item : element.getAsJsonArray()) {
                if (item.isJsonPrimitive()) {
                    String value = item.getAsString();
                    if (StrUtil.isNotBlank(value)) {
                        result.add(value.trim());
                    }
                }
            }
            return result;
        }
        return List.of();
    }
}
