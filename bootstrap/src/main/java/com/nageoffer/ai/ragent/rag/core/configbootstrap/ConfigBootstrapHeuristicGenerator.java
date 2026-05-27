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
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapDocumentSample.ChunkSample;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapSuggestion.IntentNodeSuggestion;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapSuggestion.TermMappingSuggestion;
import com.nageoffer.ai.ragent.rag.enums.IntentKind;
import com.nageoffer.ai.ragent.rag.enums.IntentLevel;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Component
@Slf4j
public class ConfigBootstrapHeuristicGenerator {

    private static final Pattern PAREN_ALIAS = Pattern.compile("([\\p{IsHan}A-Za-z0-9_\\-]{2,24})[（(]([^）)]{2,80})[）)]");
    private static final Pattern SLASH_ALIAS = Pattern.compile("([\\p{IsHan}A-Za-z0-9_\\-]{2,24})[/／]([\\p{IsHan}A-Za-z0-9_\\-]{2,24})");
    private static final int MAX_INTENT_CANDIDATES = 60;
    private static final int MAX_TERM_CANDIDATES = 500;

    public ConfigBootstrapSuggestion generate(List<ConfigBootstrapDocumentSample> samples) {
        if (samples == null || samples.isEmpty()) {
            return ConfigBootstrapSuggestion.empty();
        }
        return new ConfigBootstrapSuggestion(
                generateTermMappings(samples),
                generateIntentNodes(samples)
        );
    }

    private List<IntentNodeSuggestion> generateIntentNodes(List<ConfigBootstrapDocumentSample> samples) {
        Map<String, ConfigBootstrapDocumentSample> byKb = new LinkedHashMap<>();
        Map<String, ConfigBootstrapDocumentSample> byDoc = new LinkedHashMap<>();
        for (ConfigBootstrapDocumentSample sample : samples) {
            if (StrUtil.isNotBlank(sample.kbId())) {
                byKb.putIfAbsent(sample.kbId(), sample);
            }
            if (StrUtil.isNotBlank(sample.docId())) {
                byDoc.putIfAbsent(sample.docId(), sample);
            }
        }

        List<IntentNodeSuggestion> result = new ArrayList<>();
        int sort = 0;
        for (ConfigBootstrapDocumentSample sample : byKb.values()) {
            result.add(new IntentNodeSuggestion(
                    buildIntentCode("kb", sample.kbName()),
                    sample.kbName(),
                    IntentLevel.TOPIC.getCode(),
                    null,
                    "基于知识库《" + sample.kbName() + "》自动生成的初始意图节点，请审核后发布。",
                    buildExamples(sample.kbName()),
                    sample.kbId(),
                    sample.collectionName(),
                    null,
                    IntentKind.KB.getCode(),
                    sort++,
                    0.72D,
                    "MEDIUM",
                    List.of("知识库名称：" + sample.kbName()),
                    ConfigBootstrapSuggestion.SOURCE_RULE
            ));
            if (result.size() >= MAX_INTENT_CANDIDATES) {
                return result;
            }
        }

        for (ConfigBootstrapDocumentSample sample : byDoc.values()) {
            String docTopic = stripExtension(sample.docName());
            if (StrUtil.isBlank(docTopic) || sameAsKbName(docTopic, sample.kbName())) {
                continue;
            }
            result.add(new IntentNodeSuggestion(
                    buildIntentCode("doc", sample.kbName() + "-" + docTopic),
                    docTopic,
                    IntentLevel.TOPIC.getCode(),
                    null,
                    "基于文档《" + sample.docName() + "》自动生成的初始意图节点，请审核其业务边界。",
                    buildExamples(docTopic),
                    sample.kbId(),
                    sample.collectionName(),
                    null,
                    IntentKind.KB.getCode(),
                    sort++,
                    0.64D,
                    "MEDIUM",
                    List.of("文档名称：" + sample.docName()),
                    ConfigBootstrapSuggestion.SOURCE_RULE
            ));
            if (result.size() >= MAX_INTENT_CANDIDATES) {
                break;
            }
        }
        return result;
    }

    private List<TermMappingSuggestion> generateTermMappings(List<ConfigBootstrapDocumentSample> samples) {
        Map<String, TermMappingSuggestion> dedup = new LinkedHashMap<>();
        for (ConfigBootstrapDocumentSample sample : samples) {
            List<String> texts = new ArrayList<>();
            texts.add(sample.docName());
            texts.addAll(sample.chunks() == null
                    ? List.of()
                    : sample.chunks().stream().map(ChunkSample::content).toList());
            for (String text : texts) {
                collectParenthesizedAliases(text, dedup, sample.docName());
                collectSlashAliases(text, dedup, sample.docName());
                if (dedup.size() >= MAX_TERM_CANDIDATES) {
                    log.info(
                            "AI 初始化配置规则术语候选达到上限: limit={}, current={}, docId={}, docName={}",
                            MAX_TERM_CANDIDATES,
                            dedup.size(),
                            sample.docId(),
                            sample.docName()
                    );
                    return new ArrayList<>(dedup.values());
                }
            }
        }
        return new ArrayList<>(dedup.values());
    }

    private void collectParenthesizedAliases(String text,
                                             Map<String, TermMappingSuggestion> dedup,
                                             String docName) {
        if (StrUtil.isBlank(text)) {
            return;
        }
        Matcher matcher = PAREN_ALIAS.matcher(text);
        while (matcher.find()) {
            String target = normalizeTerm(matcher.group(1));
            if (StrUtil.isBlank(target)) {
                continue;
            }
            for (String alias : splitAliases(matcher.group(2))) {
                addTermMapping(alias, target, dedup, docName, 0.76D);
            }
        }
    }

    private void collectSlashAliases(String text,
                                     Map<String, TermMappingSuggestion> dedup,
                                     String docName) {
        if (StrUtil.isBlank(text)) {
            return;
        }
        Matcher matcher = SLASH_ALIAS.matcher(text);
        while (matcher.find()) {
            String target = normalizeTerm(matcher.group(1));
            String alias = normalizeTerm(matcher.group(2));
            addTermMapping(alias, target, dedup, docName, 0.68D);
        }
    }

    private List<String> splitAliases(String rawAliases) {
        if (StrUtil.isBlank(rawAliases)) {
            return List.of();
        }
        Set<String> aliases = new LinkedHashSet<>();
        for (String item : rawAliases.split("[、,，/／;；]")) {
            String alias = normalizeTerm(item);
            if (StrUtil.isNotBlank(alias)) {
                aliases.add(alias);
            }
        }
        return new ArrayList<>(aliases);
    }

    private void addTermMapping(String source,
                                String target,
                                Map<String, TermMappingSuggestion> dedup,
                                String docName,
                                double confidence) {
        if (StrUtil.isBlank(source) || StrUtil.isBlank(target) || source.equals(target)) {
            return;
        }
        String key = source + "->" + target;
        dedup.putIfAbsent(key, new TermMappingSuggestion(
                source,
                target,
                confidence,
                "MEDIUM",
                List.of("文档《" + docName + "》中出现同义/别名表达"),
                ConfigBootstrapSuggestion.SOURCE_RULE
        ));
    }

    private String normalizeTerm(String value) {
        if (value == null) {
            return null;
        }
        String term = value.trim()
                .replaceAll("^#+", "")
                .replaceAll("[`*_\\s]+", "");
        if (term.length() < 2 || term.length() > 32) {
            return null;
        }
        return term;
    }

    private List<String> buildExamples(String topic) {
        if (StrUtil.isBlank(topic)) {
            return List.of();
        }
        return List.of(
                topic + "是什么？",
                topic + "怎么处理？",
                topic + "有哪些要求？"
        );
    }

    private String stripExtension(String docName) {
        if (StrUtil.isBlank(docName)) {
            return docName;
        }
        return docName.replaceFirst("\\.[A-Za-z0-9]+$", "").trim();
    }

    private boolean sameAsKbName(String docTopic, String kbName) {
        return StrUtil.isNotBlank(docTopic)
                && StrUtil.isNotBlank(kbName)
                && docTopic.equalsIgnoreCase(kbName);
    }

    private String buildIntentCode(String prefix, String text) {
        String normalized = StrUtil.blankToDefault(text, "unknown")
                .toLowerCase(Locale.ROOT)
                .replaceAll("[^\\p{IsHan}a-z0-9]+", "-")
                .replaceAll("^-+|-+$", "")
                .replaceAll("-{2,}", "-");
        if (normalized.length() > 48) {
            normalized = normalized.substring(0, 48);
        }
        return prefix + "-" + normalized;
    }
}
