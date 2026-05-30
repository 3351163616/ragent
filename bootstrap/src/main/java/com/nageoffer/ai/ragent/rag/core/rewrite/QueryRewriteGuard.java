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

package com.nageoffer.ai.ragent.rag.core.rewrite;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.infra.embedding.EmbeddingService;
import com.nageoffer.ai.ragent.rag.config.RAGConfigProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * 查询改写质量守卫。
 * <p>
 * LLM 改写结果只有通过语义相似度校验和短 Query 防噪校验后，才允许进入后续意图识别与检索。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class QueryRewriteGuard {

    private static final Set<String> QUERY_STOP_WORDS = Set.of(
            "怎么", "怎样", "如何", "咋", "查", "查询", "一下", "一下子", "帮我", "请问", "请",
            "能否", "可以", "能不能", "是什么", "怎么办", "哪里", "哪个", "哪些", "吗", "呢", "啊",
            "的", "了", "呀", "吧"
    );

    private static final Set<String> REFUND_TERMS = Set.of("退费", "退款", "退钱", "退还", "返款", "返费");
    private static final Set<String> PAYMENT_TERMS = Set.of("收费", "付款", "缴费", "支付", "充值", "扣费", "付费");
    private static final Set<String> LOGISTICS_TERMS = Set.of("物流", "快递", "运单", "包裹", "配送");
    private static final Set<String> RETURN_EXCHANGE_TERMS = Set.of("退货", "换货", "售后", "退换货");
    private static final Set<String> INVOICE_TERMS = Set.of("发票", "开票", "票据");

    private static final List<Set<String>> SYNONYM_GROUPS = List.of(
            REFUND_TERMS,
            PAYMENT_TERMS,
            LOGISTICS_TERMS,
            RETURN_EXCHANGE_TERMS,
            INVOICE_TERMS,
            Set.of("税号", "纳税人识别号", "统一社会信用代码"),
            Set.of("发票抬头", "开票抬头", "开票名称"),
            Set.of("社保", "社会保险"),
            Set.of("医保", "医疗保险")
    );

    private static final List<Set<String>> SHORT_QUERY_RISK_GROUPS = List.of(
            REFUND_TERMS,
            PAYMENT_TERMS,
            LOGISTICS_TERMS,
            RETURN_EXCHANGE_TERMS,
            INVOICE_TERMS
    );

    private final EmbeddingService embeddingService;
    private final RAGConfigProperties ragConfigProperties;

    public RewriteResult validateOrFallback(String originalQuestion,
                                            String normalizedQuestion,
                                            RewriteResult candidate) {
        String fallbackQuestion = fallbackQuestion(originalQuestion, normalizedQuestion);
        RewriteResult fallback = new RewriteResult(fallbackQuestion, List.of(fallbackQuestion));
        RewriteResult normalizedCandidate = normalizeCandidate(candidate);
        if (normalizedCandidate == null) {
            return fallback;
        }

        String beforeRewrite = fallbackQuestion;
        String rewrittenQuestion = normalizedCandidate.rewrittenQuestion();
        if (!passSemanticSimilarity(beforeRewrite, rewrittenQuestion)) {
            return fallback;
        }
        if (!passShortQueryGuard(beforeRewrite, normalizedCandidate)) {
            return fallback;
        }
        return normalizedCandidate;
    }

    private RewriteResult normalizeCandidate(RewriteResult candidate) {
        if (candidate == null || StrUtil.isBlank(candidate.rewrittenQuestion())) {
            return null;
        }
        String rewrite = candidate.rewrittenQuestion().trim();
        List<String> subQuestions = CollUtil.isEmpty(candidate.subQuestions())
                ? List.of(rewrite)
                : candidate.subQuestions().stream()
                        .filter(StrUtil::isNotBlank)
                        .map(String::trim)
                        .distinct()
                        .toList();
        if (CollUtil.isEmpty(subQuestions)) {
            subQuestions = List.of(rewrite);
        }
        return new RewriteResult(rewrite, subQuestions);
    }

    private boolean passSemanticSimilarity(String beforeRewrite, String rewrittenQuestion) {
        if (!Boolean.TRUE.equals(ragConfigProperties.getQueryRewriteSemanticValidationEnabled())) {
            return true;
        }
        if (sameText(beforeRewrite, rewrittenQuestion)) {
            return true;
        }

        double threshold = resolveSimilarityThreshold();
        try {
            double similarity = cosineSimilarity(beforeRewrite, rewrittenQuestion);
            if (similarity >= threshold) {
                log.info("查询改写语义相似度校验通过，similarity={}, threshold={}, before={}, after={}",
                        similarity, threshold, beforeRewrite, rewrittenQuestion);
                return true;
            }
            log.warn("查询改写语义相似度低于阈值，丢弃改写结果，similarity={}, threshold={}, before={}, after={}",
                    similarity, threshold, beforeRewrite, rewrittenQuestion);
            return false;
        } catch (Exception ex) {
            log.warn("查询改写语义相似度计算失败，丢弃改写结果，before={}, after={}",
                    beforeRewrite, rewrittenQuestion, ex);
            return false;
        }
    }

    private boolean passShortQueryGuard(String beforeRewrite, RewriteResult candidate) {
        if (!isShortQuery(beforeRewrite)) {
            return true;
        }
        if (candidate.subQuestions().size() != 1) {
            log.warn("短 Query 改写产生多个子问题，丢弃改写结果，before={}, subQuestions={}",
                    beforeRewrite, candidate.subQuestions());
            return false;
        }
        String afterRewrite = candidate.rewrittenQuestion();
        if (!coreTermsPreserved(beforeRewrite, afterRewrite)) {
            log.warn("短 Query 改写未保留核心词，丢弃改写结果，before={}, after={}", beforeRewrite, afterRewrite);
            return false;
        }
        if (hasRiskyMeaningShift(beforeRewrite, afterRewrite)) {
            log.warn("短 Query 改写引入高风险业务词，丢弃改写结果，before={}, after={}", beforeRewrite, afterRewrite);
            return false;
        }
        return true;
    }

    private double cosineSimilarity(String beforeRewrite, String rewrittenQuestion) {
        List<List<Float>> embeddings = embedPair(beforeRewrite, rewrittenQuestion);
        if (embeddings.size() < 2) {
            throw new IllegalStateException("Embedding 批量结果数量不足");
        }
        return cosine(embeddings.get(0), embeddings.get(1));
    }

    private List<List<Float>> embedPair(String beforeRewrite, String rewrittenQuestion) {
        List<String> texts = List.of(beforeRewrite, rewrittenQuestion);
        try {
            List<List<Float>> embeddings = embeddingService.embedBatch(texts);
            if (embeddings != null && embeddings.size() == texts.size()) {
                return embeddings;
            }
            log.warn("Embedding 批量结果异常，fallback 到单条向量化，size={}",
                    embeddings == null ? null : embeddings.size());
        } catch (UnsupportedOperationException ex) {
            log.debug("Embedding 服务不支持批量调用，fallback 到单条向量化");
        }
        return List.of(embeddingService.embed(beforeRewrite), embeddingService.embed(rewrittenQuestion));
    }

    private double cosine(List<Float> left, List<Float> right) {
        if (CollUtil.isEmpty(left) || CollUtil.isEmpty(right) || left.size() != right.size()) {
            throw new IllegalStateException("Embedding 为空或维度不一致");
        }

        double dot = 0;
        double leftNorm = 0;
        double rightNorm = 0;
        for (int i = 0; i < left.size(); i++) {
            double l = left.get(i);
            double r = right.get(i);
            dot += l * r;
            leftNorm += l * l;
            rightNorm += r * r;
        }
        if (leftNorm <= 0 || rightNorm <= 0) {
            return 0D;
        }
        return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
    }

    private boolean coreTermsPreserved(String beforeRewrite, String afterRewrite) {
        List<String> coreTerms = extractCoreTerms(beforeRewrite);
        if (coreTerms.isEmpty()) {
            return true;
        }
        String normalizedAfter = normalizeText(afterRewrite);
        for (String coreTerm : coreTerms) {
            if (normalizedAfter.contains(coreTerm)) {
                continue;
            }
            Set<String> synonyms = findSynonymGroup(coreTerm);
            if (synonyms != null && containsAny(normalizedAfter, synonyms)) {
                continue;
            }
            return false;
        }
        return true;
    }

    private boolean hasRiskyMeaningShift(String beforeRewrite, String afterRewrite) {
        String normalizedBefore = normalizeText(beforeRewrite);
        String normalizedAfter = normalizeText(afterRewrite);

        if (containsAny(normalizedBefore, REFUND_TERMS) && containsAny(normalizedAfter, PAYMENT_TERMS)) {
            return true;
        }
        if (containsAny(normalizedBefore, PAYMENT_TERMS) && containsAny(normalizedAfter, REFUND_TERMS)) {
            return true;
        }

        for (Set<String> riskGroup : SHORT_QUERY_RISK_GROUPS) {
            if (!containsAny(normalizedBefore, riskGroup) && containsAny(normalizedAfter, riskGroup)) {
                return true;
            }
        }
        return false;
    }

    private List<String> extractCoreTerms(String query) {
        String normalized = normalizeText(query);
        if (StrUtil.isBlank(normalized)) {
            return List.of();
        }
        String withoutStopWords = normalized;
        List<String> stopWords = QUERY_STOP_WORDS.stream()
                .sorted((left, right) -> Integer.compare(right.length(), left.length()))
                .toList();
        for (String stopWord : stopWords) {
            withoutStopWords = withoutStopWords.replace(stopWord, "");
        }
        if (StrUtil.isBlank(withoutStopWords) || withoutStopWords.length() < 2) {
            return List.of();
        }

        List<String> coreTerms = new ArrayList<>();
        for (Set<String> group : SYNONYM_GROUPS) {
            for (String term : group) {
                String normalizedTerm = normalizeText(term);
                if (withoutStopWords.contains(normalizedTerm)) {
                    coreTerms.add(normalizedTerm);
                    return coreTerms;
                }
            }
        }
        coreTerms.add(withoutStopWords);
        return coreTerms;
    }

    private Set<String> findSynonymGroup(String term) {
        for (Set<String> group : SYNONYM_GROUPS) {
            if (group.stream().map(this::normalizeText).anyMatch(term::equals)) {
                return group;
            }
        }
        return null;
    }

    private boolean isShortQuery(String query) {
        Integer maxLength = ragConfigProperties.getQueryRewriteShortQueryMaxLength();
        return normalizeText(query).length() <= (maxLength == null || maxLength <= 0 ? 8 : maxLength);
    }

    private double resolveSimilarityThreshold() {
        Double threshold = ragConfigProperties.getQueryRewriteSemanticSimilarityThreshold();
        return threshold == null || threshold <= 0 ? 0.8D : threshold;
    }

    private boolean containsAny(String text, Set<String> terms) {
        return terms.stream()
                .map(this::normalizeText)
                .anyMatch(text::contains);
    }

    private String normalizeText(String text) {
        if (text == null) {
            return "";
        }
        String lower = text.toLowerCase(Locale.ROOT);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < lower.length(); i++) {
            char c = lower.charAt(i);
            if (Character.isLetterOrDigit(c)) {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private boolean sameText(String left, String right) {
        return normalizeText(left).equals(normalizeText(right));
    }

    private String fallbackQuestion(String originalQuestion, String normalizedQuestion) {
        if (StrUtil.isNotBlank(normalizedQuestion)) {
            return normalizedQuestion.trim();
        }
        return StrUtil.blankToDefault(originalQuestion, "").trim();
    }
}
