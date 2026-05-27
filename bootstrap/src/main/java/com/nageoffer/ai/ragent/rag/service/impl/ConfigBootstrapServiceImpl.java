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

package com.nageoffer.ai.ragent.rag.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.lang.Assert;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.nageoffer.ai.ragent.framework.context.UserContext;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.framework.exception.ClientException;
import com.nageoffer.ai.ragent.infra.chat.LLMService;
import com.nageoffer.ai.ragent.infra.token.TokenCounterService;
import com.nageoffer.ai.ragent.ingestion.service.IntentTreeService;
import com.nageoffer.ai.ragent.knowledge.dao.entity.KnowledgeBaseDO;
import com.nageoffer.ai.ragent.knowledge.dao.entity.KnowledgeChunkDO;
import com.nageoffer.ai.ragent.knowledge.dao.entity.KnowledgeDocumentDO;
import com.nageoffer.ai.ragent.knowledge.dao.mapper.KnowledgeBaseMapper;
import com.nageoffer.ai.ragent.knowledge.dao.mapper.KnowledgeChunkMapper;
import com.nageoffer.ai.ragent.knowledge.dao.mapper.KnowledgeDocumentMapper;
import com.nageoffer.ai.ragent.rag.controller.request.ConfigBootstrapCreateRequest;
import com.nageoffer.ai.ragent.rag.controller.request.ConfigBootstrapPublishRequest;
import com.nageoffer.ai.ragent.rag.controller.request.ConfigBootstrapReviewRequest;
import com.nageoffer.ai.ragent.rag.controller.request.ConfigBootstrapRunPageRequest;
import com.nageoffer.ai.ragent.rag.controller.request.IntentNodeCreateRequest;
import com.nageoffer.ai.ragent.rag.controller.vo.ConfigBootstrapPublishVO;
import com.nageoffer.ai.ragent.rag.controller.vo.ConfigBootstrapRunVO;
import com.nageoffer.ai.ragent.rag.controller.vo.IntentNodeCandidateVO;
import com.nageoffer.ai.ragent.rag.controller.vo.TermMappingCandidateVO;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapCandidateParser;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapDocumentSample;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapDocumentSample.ChunkSample;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapHeuristicGenerator;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapSuggestion;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapSuggestion.IntentNodeSuggestion;
import com.nageoffer.ai.ragent.rag.core.configbootstrap.ConfigBootstrapSuggestion.TermMappingSuggestion;
import com.nageoffer.ai.ragent.rag.core.intent.IntentTreeCacheManager;
import com.nageoffer.ai.ragent.rag.core.model.InternalChatModelSelector;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptTemplateLoader;
import com.nageoffer.ai.ragent.rag.core.rewrite.QueryTermMappingCacheManager;
import com.nageoffer.ai.ragent.rag.dao.entity.ConfigBootstrapRunDO;
import com.nageoffer.ai.ragent.rag.dao.entity.IntentNodeCandidateDO;
import com.nageoffer.ai.ragent.rag.dao.entity.IntentNodeDO;
import com.nageoffer.ai.ragent.rag.dao.entity.QueryTermMappingDO;
import com.nageoffer.ai.ragent.rag.dao.entity.TermMappingCandidateDO;
import com.nageoffer.ai.ragent.rag.dao.mapper.ConfigBootstrapRunMapper;
import com.nageoffer.ai.ragent.rag.dao.mapper.IntentNodeCandidateMapper;
import com.nageoffer.ai.ragent.rag.dao.mapper.IntentNodeMapper;
import com.nageoffer.ai.ragent.rag.dao.mapper.QueryTermMappingMapper;
import com.nageoffer.ai.ragent.rag.dao.mapper.TermMappingCandidateMapper;
import com.nageoffer.ai.ragent.rag.enums.IntentKind;
import com.nageoffer.ai.ragent.rag.enums.IntentLevel;
import com.nageoffer.ai.ragent.rag.service.ConfigBootstrapService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class ConfigBootstrapServiceImpl implements ConfigBootstrapService {

    private static final String PROMPT_PATH = "prompt/config-bootstrap.st";
    private static final String STATUS_RUNNING = "RUNNING";
    private static final String STATUS_COMPLETED = "COMPLETED";
    private static final String STATUS_FAILED = "FAILED";
    private static final String STATUS_APPROVED = "APPROVED";
    private static final String STATUS_REJECTED = "REJECTED";
    private static final String STATUS_PUBLISHED = "PUBLISHED";
    private static final String STATUS_ROLLED_BACK = "ROLLED_BACK";
    private static final String STATUS_PENDING = "PENDING";
    private static final int DEFAULT_MAX_DOCUMENTS = 100;
    private static final int DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 3;
    private static final int MAX_SAMPLE_CONTENT_LENGTH = 1200;
    private static final int LLM_OUTPUT_TOKEN_RESERVE = 4096;
    private static final double LLM_INPUT_BUDGET_RATIO = 0.9D;
    private static final Type STRING_LIST_TYPE = new TypeToken<List<String>>() {
    }.getType();

    private final ConfigBootstrapRunMapper runMapper;
    private final TermMappingCandidateMapper termCandidateMapper;
    private final IntentNodeCandidateMapper intentCandidateMapper;
    private final KnowledgeBaseMapper knowledgeBaseMapper;
    private final KnowledgeDocumentMapper documentMapper;
    private final KnowledgeChunkMapper chunkMapper;
    private final QueryTermMappingMapper queryTermMappingMapper;
    private final IntentNodeMapper intentNodeMapper;
    private final LLMService llmService;
    private final TokenCounterService tokenCounterService;
    private final InternalChatModelSelector internalChatModelSelector;
    private final PromptTemplateLoader promptTemplateLoader;
    private final ConfigBootstrapCandidateParser candidateParser;
    private final ConfigBootstrapHeuristicGenerator heuristicGenerator;
    private final IntentTreeService intentTreeService;
    private final QueryTermMappingCacheManager queryTermMappingCacheManager;
    private final IntentTreeCacheManager intentTreeCacheManager;
    private final Gson gson = new Gson();

    @Override
    public ConfigBootstrapRunVO createRun(ConfigBootstrapCreateRequest requestParam) {
        long startedAt = System.currentTimeMillis();
        ConfigBootstrapCreateRequest request = requestParam == null ? new ConfigBootstrapCreateRequest() : requestParam;
        List<String> kbIds = normalizeIds(request.getKbIds());
        boolean useLlm = request.getUseLlm() == null || request.getUseLlm();
        int maxDocuments = normalizeLimit(request.getMaxDocuments(), DEFAULT_MAX_DOCUMENTS, 500);
        int maxChunksPerDocument = normalizeLimit(
                request.getMaxChunksPerDocument(),
                DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
                50
        );

        ConfigBootstrapRunDO run = ConfigBootstrapRunDO.builder()
                .status(STATUS_RUNNING)
                .kbIds(gson.toJson(kbIds))
                .useLlm(useLlm ? 1 : 0)
                .documentCount(0)
                .chunkCount(0)
                .termCandidateCount(0)
                .intentCandidateCount(0)
                .createBy(UserContext.getUsername())
                .updateBy(UserContext.getUsername())
                .build();
        runMapper.insert(run);
        log.info(
                "AI 初始化配置任务开始: runId={}, kbScope={}, maxDocuments={}, maxChunksPerDocument={}, useLlm={}",
                run.getId(),
                kbIds.isEmpty() ? "ALL" : kbIds.size(),
                maxDocuments,
                maxChunksPerDocument,
                useLlm
        );

        try {
            List<ConfigBootstrapDocumentSample> samples = collectSamples(kbIds, maxDocuments, maxChunksPerDocument);
            if (samples.isEmpty()) {
                throw new ClientException("没有可用于初始化配置的知识库文档或分块");
            }
            int sampledDocumentCount = (int) samples.stream()
                    .map(ConfigBootstrapDocumentSample::docId)
                    .distinct()
                    .count();
            int sampledChunkCount = samples.stream().mapToInt(sample -> sample.chunks().size()).sum();
            log.info(
                    "AI 初始化配置采样完成: runId={}, documents={}, chunks={}, elapsedMs={}",
                    run.getId(),
                    sampledDocumentCount,
                    sampledChunkCount,
                    System.currentTimeMillis() - startedAt
            );
            run.setSampleJson(gson.toJson(samples));

            ConfigBootstrapSuggestion suggestion = generateSuggestions(samples, useLlm);
            List<TermMappingCandidateDO> termCandidates = toTermCandidates(run.getId(), suggestion.termMappings());
            List<IntentNodeCandidateDO> intentCandidates = toIntentCandidates(run.getId(), suggestion.intentNodes());
            termCandidates.forEach(termCandidateMapper::insert);
            intentCandidates.forEach(intentCandidateMapper::insert);

            run.setStatus(STATUS_COMPLETED);
            run.setDocumentCount(sampledDocumentCount);
            run.setChunkCount(sampledChunkCount);
            run.setTermCandidateCount(termCandidates.size());
            run.setIntentCandidateCount(intentCandidates.size());
            run.setSummary(String.format(
                    "采样 %d 篇文档、%d 个 chunk，生成术语候选 %d 条、意图候选 %d 条",
                    run.getDocumentCount(),
                    run.getChunkCount(),
                    termCandidates.size(),
                    intentCandidates.size()
            ));
            run.setUpdateBy(UserContext.getUsername());
            runMapper.updateById(run);
            log.info(
                    "AI 初始化配置任务完成: runId={}, documents={}, chunks={}, termCandidates={}, intentCandidates={}, elapsedMs={}",
                    run.getId(),
                    run.getDocumentCount(),
                    run.getChunkCount(),
                    termCandidates.size(),
                    intentCandidates.size(),
                    System.currentTimeMillis() - startedAt
            );
        } catch (Exception ex) {
            log.error("AI 初始化配置候选生成失败, runId={}", run.getId(), ex);
            run.setStatus(STATUS_FAILED);
            run.setErrorMessage(ex.getMessage());
            run.setUpdateBy(UserContext.getUsername());
            runMapper.updateById(run);
            log.info(
                    "AI 初始化配置任务失败: runId={}, elapsedMs={}, error={}",
                    run.getId(),
                    System.currentTimeMillis() - startedAt,
                    ex.getMessage()
            );
        }

        return queryRun(run.getId());
    }

    @Override
    public IPage<ConfigBootstrapRunVO> pageRuns(ConfigBootstrapRunPageRequest requestParam) {
        ConfigBootstrapRunPageRequest request = requestParam == null
                ? new ConfigBootstrapRunPageRequest()
                : requestParam;
        long current = request.getCurrent() <= 0 ? 1 : request.getCurrent();
        long size = request.getSize() <= 0 ? 10 : Math.min(request.getSize(), 100);
        Page<ConfigBootstrapRunDO> pageParam = new Page<>(current, size);
        IPage<ConfigBootstrapRunDO> result = runMapper.selectPage(
                pageParam,
                Wrappers.lambdaQuery(ConfigBootstrapRunDO.class)
                        .eq(StrUtil.isNotBlank(request.getStatus()),
                                ConfigBootstrapRunDO::getStatus,
                                StrUtil.blankToDefault(request.getStatus(), "").toUpperCase(Locale.ROOT))
                        .orderByDesc(ConfigBootstrapRunDO::getCreateTime)
        );

        Page<ConfigBootstrapRunVO> voPage = new Page<>(result.getCurrent(), result.getSize(), result.getTotal());
        voPage.setRecords(result.getRecords().stream()
                .map(run -> toRunVO(run, List.of(), List.of()))
                .toList());
        return voPage;
    }

    @Override
    public ConfigBootstrapRunVO queryRun(String runId) {
        ConfigBootstrapRunDO run = loadRun(runId);
        List<TermMappingCandidateDO> termCandidates = termCandidateMapper.selectList(
                Wrappers.lambdaQuery(TermMappingCandidateDO.class)
                        .eq(TermMappingCandidateDO::getRunId, runId)
                        .orderByDesc(TermMappingCandidateDO::getConfidence)
                        .orderByAsc(TermMappingCandidateDO::getId)
        );
        List<IntentNodeCandidateDO> intentCandidates = intentCandidateMapper.selectList(
                Wrappers.lambdaQuery(IntentNodeCandidateDO.class)
                        .eq(IntentNodeCandidateDO::getRunId, runId)
                        .orderByAsc(IntentNodeCandidateDO::getSortOrder, IntentNodeCandidateDO::getId)
        );
        return toRunVO(run, termCandidates, intentCandidates);
    }

    @Override
    public void reviewTermCandidate(String candidateId, ConfigBootstrapReviewRequest requestParam) {
        String nextStatus = normalizeReviewStatus(requestParam == null ? null : requestParam.getStatus());
        TermMappingCandidateDO candidate = termCandidateMapper.selectById(candidateId);
        Assert.notNull(candidate, () -> new ClientException("术语映射候选不存在"));
        candidate.setStatus(nextStatus);
        candidate.setReviewComment(requestParam == null ? null : requestParam.getReviewComment());
        candidate.setReviewBy(UserContext.getUsername());
        candidate.setReviewTime(new Date());
        termCandidateMapper.updateById(candidate);
    }

    @Override
    public void reviewIntentCandidate(String candidateId, ConfigBootstrapReviewRequest requestParam) {
        String nextStatus = normalizeReviewStatus(requestParam == null ? null : requestParam.getStatus());
        IntentNodeCandidateDO candidate = intentCandidateMapper.selectById(candidateId);
        Assert.notNull(candidate, () -> new ClientException("意图节点候选不存在"));
        candidate.setStatus(nextStatus);
        candidate.setReviewComment(requestParam == null ? null : requestParam.getReviewComment());
        candidate.setReviewBy(UserContext.getUsername());
        candidate.setReviewTime(new Date());
        intentCandidateMapper.updateById(candidate);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ConfigBootstrapPublishVO publishRun(String runId, ConfigBootstrapPublishRequest requestParam) {
        loadRun(runId);
        boolean enabled = requestParam != null && Boolean.TRUE.equals(requestParam.getEnablePublishedCandidates());

        int publishedTerms = 0;
        int skippedTerms = 0;
        List<TermMappingCandidateDO> terms = approvedTerms(runId);
        for (TermMappingCandidateDO candidate : terms) {
            if (StrUtil.isNotBlank(candidate.getPublishedId()) || existsTermMapping(candidate)) {
                skippedTerms++;
                continue;
            }
            QueryTermMappingDO mapping = new QueryTermMappingDO();
            mapping.setSourceTerm(candidate.getSourceTerm());
            mapping.setTargetTerm(candidate.getTargetTerm());
            mapping.setMatchType(1);
            mapping.setPriority(0);
            mapping.setEnabled(enabled ? 1 : 0);
            mapping.setRemark("AI 初始化配置候选发布，runId=" + runId);
            mapping.setCreateBy(UserContext.getUsername());
            mapping.setUpdateBy(UserContext.getUsername());
            queryTermMappingMapper.insert(mapping);
            markTermPublished(candidate, mapping.getId());
            publishedTerms++;
        }

        int publishedIntents = 0;
        int skippedIntents = 0;
        List<IntentNodeCandidateDO> intents = approvedIntents(runId);
        for (IntentNodeCandidateDO candidate : intents) {
            if (StrUtil.isNotBlank(candidate.getPublishedId())
                    || existsIntentCode(candidate.getIntentCode())
                    || missingKbBinding(candidate)) {
                skippedIntents++;
                continue;
            }
            IntentNodeCreateRequest createRequest = IntentNodeCreateRequest.builder()
                    .intentCode(candidate.getIntentCode())
                    .name(candidate.getName())
                    .level(candidate.getLevel())
                    .parentCode(candidate.getParentCode())
                    .description(candidate.getDescription())
                    .examples(parseStringList(candidate.getExamplesJson()))
                    .kbId(candidate.getKbId())
                    .topK(candidate.getTopK())
                    .kind(candidate.getKind())
                    .sortOrder(candidate.getSortOrder())
                    .enabled(enabled ? 1 : 0)
                    .build();
            String intentNodeId = intentTreeService.createNode(createRequest);
            markIntentPublished(candidate, intentNodeId);
            publishedIntents++;
        }

        queryTermMappingCacheManager.clearCache();
        intentTreeCacheManager.clearIntentTreeCache();
        updateRunStatus(runId, STATUS_PUBLISHED);

        return ConfigBootstrapPublishVO.builder()
                .runId(runId)
                .publishedTermMappings(publishedTerms)
                .publishedIntentNodes(publishedIntents)
                .skippedTermMappings(skippedTerms)
                .skippedIntentNodes(skippedIntents)
                .rolledBackTermMappings(0)
                .rolledBackIntentNodes(0)
                .build();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ConfigBootstrapPublishVO rollbackRun(String runId) {
        loadRun(runId);
        int termCount = 0;
        for (TermMappingCandidateDO candidate : publishedTerms(runId)) {
            QueryTermMappingDO mapping = queryTermMappingMapper.selectById(candidate.getPublishedId());
            if (mapping == null) {
                continue;
            }
            mapping.setEnabled(0);
            mapping.setUpdateBy(UserContext.getUsername());
            queryTermMappingMapper.updateById(mapping);
            candidate.setStatus(STATUS_ROLLED_BACK);
            termCandidateMapper.updateById(candidate);
            termCount++;
        }

        int intentCount = 0;
        for (IntentNodeCandidateDO candidate : publishedIntents(runId)) {
            IntentNodeDO node = intentNodeMapper.selectById(candidate.getPublishedId());
            if (node == null) {
                continue;
            }
            node.setEnabled(0);
            node.setUpdateBy(UserContext.getUsername());
            intentNodeMapper.updateById(node);
            candidate.setStatus(STATUS_ROLLED_BACK);
            intentCandidateMapper.updateById(candidate);
            intentCount++;
        }

        queryTermMappingCacheManager.clearCache();
        intentTreeCacheManager.clearIntentTreeCache();
        updateRunStatus(runId, STATUS_ROLLED_BACK);

        return ConfigBootstrapPublishVO.builder()
                .runId(runId)
                .publishedTermMappings(0)
                .publishedIntentNodes(0)
                .skippedTermMappings(0)
                .skippedIntentNodes(0)
                .rolledBackTermMappings(termCount)
                .rolledBackIntentNodes(intentCount)
                .build();
    }

    private ConfigBootstrapSuggestion generateSuggestions(List<ConfigBootstrapDocumentSample> samples, boolean useLlm) {
        long startedAt = System.currentTimeMillis();
        ConfigBootstrapSuggestion heuristic = heuristicGenerator.generate(samples);
        log.info(
                "AI 初始化配置规则候选完成: documents={}, chunks={}, termCandidates={}, intentCandidates={}, useLlm={}",
                samples.size(),
                samples.stream().mapToInt(sample -> sample.chunks().size()).sum(),
                safeTerms(heuristic).size(),
                safeIntents(heuristic).size(),
                useLlm
        );
        if (!useLlm) {
            return heuristic;
        }
        try {
            String modelId = internalChatModelSelector.modelId();
            int maxContextTokens = internalChatModelSelector.maxContextTokens(modelId);
            int maxInputTokens = maxInputTokens(maxContextTokens);
            List<List<ConfigBootstrapDocumentSample>> batches = buildPromptBatches(samples, maxInputTokens);
            ConfigBootstrapSuggestion llmSuggestion = ConfigBootstrapSuggestion.empty();
            log.info(
                    "AI 初始化配置 LLM 候选生成开始: modelId={}, maxContextTokens={}, maxInputTokens={}, batchCount={}, documents={}, chunks={}",
                    modelId,
                    maxContextTokens,
                    maxInputTokens,
                    batches.size(),
                    samples.size(),
                    countChunks(samples)
            );
            for (int i = 0; i < batches.size(); i++) {
                List<ConfigBootstrapDocumentSample> batch = batches.get(i);
                String prompt = renderPrompt(batch);
                int promptTokens = estimateTokens(prompt);
                log.info(
                        "AI 初始化配置 LLM 批次发送: modelId={}, batch={}/{}, documents={}, chunks={}, promptChars={}, promptTokens={}, maxInputTokens={}",
                        modelId,
                        i + 1,
                        batches.size(),
                        batch.size(),
                        countChunks(batch),
                        prompt.length(),
                        promptTokens,
                        maxInputTokens
                );
                ChatRequest request = ChatRequest.builder()
                        .scene("config-bootstrap")
                        .messages(List.of(ChatMessage.user(prompt)))
                        .temperature(0.1D)
                        .topP(0.3D)
                        .thinking(false)
                        .build();
                String raw = llmService.chat(request, modelId);
                ConfigBootstrapSuggestion batchSuggestion = candidateParser.parse(raw);
                llmSuggestion = mergeSuggestions(batchSuggestion, llmSuggestion);
                log.info(
                        "AI 初始化配置 LLM 批次完成: batch={}/{}, llmTerms={}, llmIntents={}, mergedTerms={}, mergedIntents={}",
                        i + 1,
                        batches.size(),
                        safeTerms(batchSuggestion).size(),
                        safeIntents(batchSuggestion).size(),
                        safeTerms(llmSuggestion).size(),
                        safeIntents(llmSuggestion).size()
                );
            }
            ConfigBootstrapSuggestion merged = mergeSuggestions(llmSuggestion, heuristic);
            log.info(
                "AI 初始化配置 LLM 候选生成完成: llmTerms={}, llmIntents={}, mergedTerms={}, mergedIntents={}, elapsedMs={}",
                    safeTerms(llmSuggestion).size(),
                    safeIntents(llmSuggestion).size(),
                    safeTerms(merged).size(),
                    safeIntents(merged).size(),
                    System.currentTimeMillis() - startedAt
            );
            return merged;
        } catch (Exception ex) {
            log.warn("AI 初始化配置 LLM 候选生成失败，使用启发式候选兜底", ex);
            log.info(
                    "AI 初始化配置使用规则候选兜底: termCandidates={}, intentCandidates={}, elapsedMs={}",
                    safeTerms(heuristic).size(),
                    safeIntents(heuristic).size(),
                    System.currentTimeMillis() - startedAt
            );
            return heuristic;
        }
    }

    private List<List<ConfigBootstrapDocumentSample>> buildPromptBatches(List<ConfigBootstrapDocumentSample> samples,
                                                                         int maxInputTokens) {
        List<List<ConfigBootstrapDocumentSample>> batches = new ArrayList<>();
        List<ConfigBootstrapDocumentSample> current = new ArrayList<>();
        for (ConfigBootstrapDocumentSample sample : samples) {
            List<ConfigBootstrapDocumentSample> probe = new ArrayList<>(current);
            probe.add(sample);
            if (!current.isEmpty() && estimateTokens(renderPrompt(probe)) > maxInputTokens) {
                batches.add(current);
                current = new ArrayList<>();
            }
            current.add(sample);
            if (current.size() == 1) {
                int singleTokens = estimateTokens(renderPrompt(current));
                if (singleTokens > maxInputTokens) {
                    log.warn(
                            "AI 初始化配置单文档样本超过模型输入预算，仍保留该文档发送: docId={}, docName={}, promptTokens={}, maxInputTokens={}",
                            sample.docId(),
                            sample.docName(),
                            singleTokens,
                            maxInputTokens
                    );
                    batches.add(current);
                    current = new ArrayList<>();
                }
            }
        }
        if (!current.isEmpty()) {
            batches.add(current);
        }
        return batches.isEmpty() ? List.of(samples) : batches;
    }

    private String renderPrompt(List<ConfigBootstrapDocumentSample> samples) {
        return promptTemplateLoader.render(PROMPT_PATH, Map.of("documents", gson.toJson(samples)));
    }

    private int maxInputTokens(int maxContextTokens) {
        int budget = (int) Math.floor(maxContextTokens * LLM_INPUT_BUDGET_RATIO) - LLM_OUTPUT_TOKEN_RESERVE;
        return Math.max(1000, budget);
    }

    private int estimateTokens(String prompt) {
        Integer tokens = tokenCounterService.countTokens(prompt);
        return tokens == null ? 0 : tokens;
    }

    private int countChunks(List<ConfigBootstrapDocumentSample> samples) {
        return samples.stream().mapToInt(sample -> sample.chunks().size()).sum();
    }

    private ConfigBootstrapSuggestion mergeSuggestions(ConfigBootstrapSuggestion primary,
                                                       ConfigBootstrapSuggestion fallback) {
        Map<String, TermMappingSuggestion> terms = new LinkedHashMap<>();
        for (TermMappingSuggestion suggestion : safeTerms(primary)) {
            terms.put(termKey(suggestion), suggestion);
        }
        for (TermMappingSuggestion suggestion : safeTerms(fallback)) {
            terms.putIfAbsent(termKey(suggestion), suggestion);
        }

        Map<String, IntentNodeSuggestion> intents = new LinkedHashMap<>();
        for (IntentNodeSuggestion suggestion : safeIntents(primary)) {
            intents.put(intentKey(suggestion), suggestion);
        }
        for (IntentNodeSuggestion suggestion : safeIntents(fallback)) {
            intents.putIfAbsent(intentKey(suggestion), suggestion);
        }
        return new ConfigBootstrapSuggestion(new ArrayList<>(terms.values()), new ArrayList<>(intents.values()));
    }

    private List<TermMappingSuggestion> safeTerms(ConfigBootstrapSuggestion suggestion) {
        return suggestion == null || suggestion.termMappings() == null ? List.of() : suggestion.termMappings();
    }

    private List<IntentNodeSuggestion> safeIntents(ConfigBootstrapSuggestion suggestion) {
        return suggestion == null || suggestion.intentNodes() == null ? List.of() : suggestion.intentNodes();
    }

    private String termKey(TermMappingSuggestion suggestion) {
        return suggestion.sourceTerm() + "->" + suggestion.targetTerm();
    }

    private String intentKey(IntentNodeSuggestion suggestion) {
        return StrUtil.blankToDefault(suggestion.intentCode(), suggestion.name());
    }

    private List<ConfigBootstrapDocumentSample> collectSamples(List<String> kbIds,
                                                               int maxDocuments,
                                                               int maxChunksPerDocument) {
        log.info(
                "AI 初始化配置采样开始: kbScope={}, maxDocuments={}, maxChunksPerDocument={}",
                kbIds.isEmpty() ? "ALL" : kbIds.size(),
                maxDocuments,
                maxChunksPerDocument
        );
        List<KnowledgeBaseDO> knowledgeBases = knowledgeBaseMapper.selectList(
                Wrappers.lambdaQuery(KnowledgeBaseDO.class)
                        .in(CollUtil.isNotEmpty(kbIds), KnowledgeBaseDO::getId, kbIds)
                        .orderByAsc(KnowledgeBaseDO::getName)
        );
        if (knowledgeBases.isEmpty()) {
            return List.of();
        }
        Map<String, KnowledgeBaseDO> kbById = knowledgeBases.stream()
                .collect(Collectors.toMap(KnowledgeBaseDO::getId, kb -> kb, (left, right) -> left));
        Map<String, Long> documentCountByKbId = knowledgeBases.stream()
                .collect(Collectors.toMap(
                        KnowledgeBaseDO::getId,
                        kb -> documentMapper.selectCount(
                                Wrappers.lambdaQuery(KnowledgeDocumentDO.class)
                                        .eq(KnowledgeDocumentDO::getKbId, kb.getId())
                                        .eq(KnowledgeDocumentDO::getEnabled, 1)),
                        (left, right) -> left));
        Map<String, Long> chunkCountByKbId = knowledgeBases.stream()
                .collect(Collectors.toMap(
                        KnowledgeBaseDO::getId,
                        kb -> chunkMapper.selectCount(
                                Wrappers.lambdaQuery(KnowledgeChunkDO.class)
                                        .eq(KnowledgeChunkDO::getKbId, kb.getId())
                                        .eq(KnowledgeChunkDO::getEnabled, 1)
                                        .isNotNull(KnowledgeChunkDO::getContent)
                                        .ne(KnowledgeChunkDO::getContent, "")),
                        (left, right) -> left));

        List<KnowledgeDocumentDO> documents = documentMapper.selectList(
                Wrappers.lambdaQuery(KnowledgeDocumentDO.class)
                        .in(KnowledgeDocumentDO::getKbId, kbById.keySet())
                        .eq(KnowledgeDocumentDO::getEnabled, 1)
                        .orderByDesc(KnowledgeDocumentDO::getUpdateTime)
        ).stream().limit(maxDocuments).toList();
        log.info(
                "AI 初始化配置采样文档命中: kbCount={}, documentCount={}",
                knowledgeBases.size(),
                documents.size()
        );
        if (documents.isEmpty()) {
            return List.of();
        }

        List<ConfigBootstrapDocumentSample> result = new ArrayList<>();
        for (KnowledgeDocumentDO document : documents) {
            KnowledgeBaseDO kb = kbById.get(document.getKbId());
            if (kb == null) {
                continue;
            }
            List<KnowledgeChunkDO> enabledChunks = chunkMapper.selectList(
                    Wrappers.lambdaQuery(KnowledgeChunkDO.class)
                            .eq(KnowledgeChunkDO::getDocId, document.getId())
                            .eq(KnowledgeChunkDO::getEnabled, 1)
                            .orderByAsc(KnowledgeChunkDO::getChunkIndex)
            ).stream()
                    .filter(chunk -> StrUtil.isNotBlank(chunk.getContent()))
                    .toList();
            List<ChunkSample> chunks = enabledChunks.stream()
                    .limit(maxChunksPerDocument)
                    .map(chunk -> new ChunkSample(
                            chunk.getId(),
                            chunk.getChunkIndex(),
                            truncateSample(chunk.getContent())))
                    .toList();
            result.add(new ConfigBootstrapDocumentSample(
                    kb.getId(),
                    kb.getName(),
                    kb.getCollectionName(),
                    documentCountByKbId.get(kb.getId()),
                    chunkCountByKbId.get(kb.getId()),
                    document.getId(),
                    document.getDocName(),
                    (long) enabledChunks.size(),
                    maxChunksPerDocument,
                    chunks
            ));
        }
        log.info(
                "AI 初始化配置采样明细完成: documentSamples={}, chunkSamples={}",
                result.size(),
                result.stream().mapToInt(sample -> sample.chunks().size()).sum()
        );
        return result;
    }

    private List<TermMappingCandidateDO> toTermCandidates(String runId, List<TermMappingSuggestion> suggestions) {
        if (CollUtil.isEmpty(suggestions)) {
            return List.of();
        }
        Map<String, TermMappingCandidateDO> dedup = new LinkedHashMap<>();
        for (TermMappingSuggestion suggestion : suggestions) {
            if (StrUtil.isBlank(suggestion.sourceTerm()) || StrUtil.isBlank(suggestion.targetTerm())
                    || suggestion.sourceTerm().equals(suggestion.targetTerm())) {
                continue;
            }
            TermMappingCandidateDO candidate = TermMappingCandidateDO.builder()
                    .runId(runId)
                    .sourceTerm(suggestion.sourceTerm())
                    .targetTerm(suggestion.targetTerm())
                    .confidence(suggestion.confidence())
                    .riskLevel(normalizeRiskLevel(suggestion.riskLevel()))
                    .generationSource(normalizeGenerationSource(suggestion.generationSource()))
                    .evidenceJson(gson.toJson(suggestion.evidence()))
                    .status(STATUS_PENDING)
                    .createBy(UserContext.getUsername())
                    .build();
            dedup.putIfAbsent(candidate.getSourceTerm() + "->" + candidate.getTargetTerm(), candidate);
        }
        return new ArrayList<>(dedup.values());
    }

    private List<IntentNodeCandidateDO> toIntentCandidates(String runId, List<IntentNodeSuggestion> suggestions) {
        if (CollUtil.isEmpty(suggestions)) {
            return List.of();
        }
        Map<String, IntentNodeCandidateDO> dedup = new LinkedHashMap<>();
        int sort = 0;
        for (IntentNodeSuggestion suggestion : suggestions) {
            if (StrUtil.isBlank(suggestion.name())) {
                continue;
            }
            String intentCode = StrUtil.blankToDefault(suggestion.intentCode(), buildIntentCode(suggestion.name()));
            IntentNodeCandidateDO candidate = IntentNodeCandidateDO.builder()
                    .runId(runId)
                    .intentCode(intentCode)
                    .name(suggestion.name())
                    .level(suggestion.level() == null ? IntentLevel.TOPIC.getCode() : suggestion.level())
                    .parentCode(StrUtil.trimToNull(suggestion.parentCode()))
                    .description(suggestion.description())
                    .examplesJson(gson.toJson(suggestion.examples()))
                    .kbId(suggestion.kbId())
                    .collectionName(suggestion.collectionName())
                    .topK(suggestion.topK())
                    .kind(suggestion.kind() == null ? IntentKind.KB.getCode() : suggestion.kind())
                    .sortOrder(suggestion.sortOrder() == null ? sort : suggestion.sortOrder())
                    .confidence(suggestion.confidence())
                    .riskLevel(normalizeRiskLevel(suggestion.riskLevel()))
                    .generationSource(normalizeGenerationSource(suggestion.generationSource()))
                    .evidenceJson(gson.toJson(suggestion.evidence()))
                    .status(STATUS_PENDING)
                    .createBy(UserContext.getUsername())
                    .build();
            dedup.putIfAbsent(candidate.getIntentCode(), candidate);
            sort++;
        }
        return new ArrayList<>(dedup.values());
    }

    private ConfigBootstrapRunDO loadRun(String runId) {
        ConfigBootstrapRunDO run = runMapper.selectById(runId);
        Assert.notNull(run, () -> new ClientException("初始化配置任务不存在"));
        return run;
    }

    private List<TermMappingCandidateDO> approvedTerms(String runId) {
        return termCandidateMapper.selectList(
                Wrappers.lambdaQuery(TermMappingCandidateDO.class)
                        .eq(TermMappingCandidateDO::getRunId, runId)
                        .eq(TermMappingCandidateDO::getStatus, STATUS_APPROVED)
        );
    }

    private List<IntentNodeCandidateDO> approvedIntents(String runId) {
        return intentCandidateMapper.selectList(
                Wrappers.lambdaQuery(IntentNodeCandidateDO.class)
                        .eq(IntentNodeCandidateDO::getRunId, runId)
                        .eq(IntentNodeCandidateDO::getStatus, STATUS_APPROVED)
        );
    }

    private List<TermMappingCandidateDO> publishedTerms(String runId) {
        return termCandidateMapper.selectList(
                Wrappers.lambdaQuery(TermMappingCandidateDO.class)
                        .eq(TermMappingCandidateDO::getRunId, runId)
                        .eq(TermMappingCandidateDO::getStatus, STATUS_PUBLISHED)
                        .isNotNull(TermMappingCandidateDO::getPublishedId)
        );
    }

    private List<IntentNodeCandidateDO> publishedIntents(String runId) {
        return intentCandidateMapper.selectList(
                Wrappers.lambdaQuery(IntentNodeCandidateDO.class)
                        .eq(IntentNodeCandidateDO::getRunId, runId)
                        .eq(IntentNodeCandidateDO::getStatus, STATUS_PUBLISHED)
                        .isNotNull(IntentNodeCandidateDO::getPublishedId)
        );
    }

    private void markTermPublished(TermMappingCandidateDO candidate, String publishedId) {
        candidate.setStatus(STATUS_PUBLISHED);
        candidate.setPublishedId(publishedId);
        candidate.setPublishBy(UserContext.getUsername());
        candidate.setPublishTime(new Date());
        termCandidateMapper.updateById(candidate);
    }

    private void markIntentPublished(IntentNodeCandidateDO candidate, String publishedId) {
        candidate.setStatus(STATUS_PUBLISHED);
        candidate.setPublishedId(publishedId);
        candidate.setPublishBy(UserContext.getUsername());
        candidate.setPublishTime(new Date());
        intentCandidateMapper.updateById(candidate);
    }

    private boolean existsTermMapping(TermMappingCandidateDO candidate) {
        return queryTermMappingMapper.selectCount(
                Wrappers.lambdaQuery(QueryTermMappingDO.class)
                        .eq(QueryTermMappingDO::getSourceTerm, candidate.getSourceTerm())
                        .eq(QueryTermMappingDO::getTargetTerm, candidate.getTargetTerm())
        ) > 0;
    }

    private boolean existsIntentCode(String intentCode) {
        return StrUtil.isNotBlank(intentCode) && intentNodeMapper.selectCount(
                Wrappers.lambdaQuery(IntentNodeDO.class)
                        .eq(IntentNodeDO::getIntentCode, intentCode)
                        .eq(IntentNodeDO::getDeleted, 0)
        ) > 0;
    }

    private boolean missingKbBinding(IntentNodeCandidateDO candidate) {
        Integer kind = candidate.getKind() == null ? IntentKind.KB.getCode() : candidate.getKind();
        Integer level = candidate.getLevel() == null ? IntentLevel.TOPIC.getCode() : candidate.getLevel();
        return Objects.equals(kind, IntentKind.KB.getCode())
                && Objects.equals(level, IntentLevel.TOPIC.getCode())
                && StrUtil.isBlank(candidate.getKbId());
    }

    private void updateRunStatus(String runId, String status) {
        ConfigBootstrapRunDO run = loadRun(runId);
        run.setStatus(status);
        run.setUpdateBy(UserContext.getUsername());
        runMapper.updateById(run);
    }

    private ConfigBootstrapRunVO toRunVO(ConfigBootstrapRunDO run,
                                         List<TermMappingCandidateDO> terms,
                                         List<IntentNodeCandidateDO> intents) {
        return ConfigBootstrapRunVO.builder()
                .id(run.getId())
                .status(run.getStatus())
                .kbIds(run.getKbIds())
                .useLlm(run.getUseLlm() != null && run.getUseLlm() == 1)
                .documentCount(run.getDocumentCount())
                .chunkCount(run.getChunkCount())
                .termCandidateCount(run.getTermCandidateCount())
                .intentCandidateCount(run.getIntentCandidateCount())
                .summary(run.getSummary())
                .sampleJson(run.getSampleJson())
                .errorMessage(run.getErrorMessage())
                .createTime(run.getCreateTime())
                .updateTime(run.getUpdateTime())
                .termMappings(terms.stream().map(this::toTermVO).toList())
                .intentNodes(intents.stream().map(this::toIntentVO).toList())
                .build();
    }

    private TermMappingCandidateVO toTermVO(TermMappingCandidateDO candidate) {
        return TermMappingCandidateVO.builder()
                .id(candidate.getId())
                .runId(candidate.getRunId())
                .sourceTerm(candidate.getSourceTerm())
                .targetTerm(candidate.getTargetTerm())
                .confidence(candidate.getConfidence())
                .riskLevel(candidate.getRiskLevel())
                .generationSource(candidate.getGenerationSource())
                .evidence(parseStringList(candidate.getEvidenceJson()))
                .status(candidate.getStatus())
                .reviewComment(candidate.getReviewComment())
                .publishedId(candidate.getPublishedId())
                .createTime(candidate.getCreateTime())
                .updateTime(candidate.getUpdateTime())
                .build();
    }

    private IntentNodeCandidateVO toIntentVO(IntentNodeCandidateDO candidate) {
        return IntentNodeCandidateVO.builder()
                .id(candidate.getId())
                .runId(candidate.getRunId())
                .intentCode(candidate.getIntentCode())
                .name(candidate.getName())
                .level(candidate.getLevel())
                .parentCode(candidate.getParentCode())
                .description(candidate.getDescription())
                .examples(parseStringList(candidate.getExamplesJson()))
                .kbId(candidate.getKbId())
                .collectionName(candidate.getCollectionName())
                .topK(candidate.getTopK())
                .kind(candidate.getKind())
                .sortOrder(candidate.getSortOrder())
                .confidence(candidate.getConfidence())
                .riskLevel(candidate.getRiskLevel())
                .generationSource(candidate.getGenerationSource())
                .evidence(parseStringList(candidate.getEvidenceJson()))
                .status(candidate.getStatus())
                .reviewComment(candidate.getReviewComment())
                .publishedId(candidate.getPublishedId())
                .createTime(candidate.getCreateTime())
                .updateTime(candidate.getUpdateTime())
                .build();
    }

    private List<String> parseStringList(String json) {
        if (StrUtil.isBlank(json)) {
            return List.of();
        }
        try {
            List<String> values = gson.fromJson(json, STRING_LIST_TYPE);
            return values == null ? List.of() : values;
        } catch (Exception ignored) {
            return List.of();
        }
    }

    private String normalizeReviewStatus(String status) {
        String normalized = StrUtil.blankToDefault(status, "").trim().toUpperCase(Locale.ROOT);
        if (!Set.of(STATUS_APPROVED, STATUS_REJECTED).contains(normalized)) {
            throw new ClientException("审核状态仅支持 APPROVED / REJECTED");
        }
        return normalized;
    }

    private List<String> normalizeIds(List<String> ids) {
        if (CollUtil.isEmpty(ids)) {
            return List.of();
        }
        return ids.stream()
                .filter(StrUtil::isNotBlank)
                .map(String::trim)
                .distinct()
                .toList();
    }

    private int normalizeLimit(Integer value, int defaultValue, int maxValue) {
        if (value == null || value <= 0) {
            return defaultValue;
        }
        return Math.min(value, maxValue);
    }

    private String truncateSample(String value) {
        if (value == null || value.length() <= MAX_SAMPLE_CONTENT_LENGTH) {
            return value;
        }
        return value.substring(0, MAX_SAMPLE_CONTENT_LENGTH);
    }

    private String normalizeRiskLevel(String riskLevel) {
        String normalized = StrUtil.blankToDefault(riskLevel, "MEDIUM").trim().toUpperCase(Locale.ROOT);
        return switch (normalized) {
            case "LOW", "MEDIUM", "HIGH" -> normalized;
            default -> "MEDIUM";
        };
    }

    private String normalizeGenerationSource(String generationSource) {
        String normalized = StrUtil.blankToDefault(
                generationSource,
                ConfigBootstrapSuggestion.SOURCE_UNKNOWN
        ).trim().toUpperCase(Locale.ROOT);
        return switch (normalized) {
            case ConfigBootstrapSuggestion.SOURCE_RULE, ConfigBootstrapSuggestion.SOURCE_LLM -> normalized;
            default -> ConfigBootstrapSuggestion.SOURCE_UNKNOWN;
        };
    }

    private String buildIntentCode(String name) {
        String normalized = StrUtil.blankToDefault(name, "topic")
                .toLowerCase(Locale.ROOT)
                .replaceAll("[^\\p{IsHan}a-z0-9]+", "-")
                .replaceAll("^-+|-+$", "")
                .replaceAll("-{2,}", "-");
        if (normalized.length() > 48) {
            normalized = normalized.substring(0, 48);
        }
        return "ai-" + normalized;
    }
}
