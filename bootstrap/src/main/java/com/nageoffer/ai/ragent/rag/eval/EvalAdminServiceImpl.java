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

package com.nageoffer.ai.ragent.rag.eval;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.lang.Assert;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nageoffer.ai.ragent.framework.exception.ClientException;
import com.nageoffer.ai.ragent.rag.dao.entity.EvalCaseDO;
import com.nageoffer.ai.ragent.rag.dao.entity.EvalRunDO;
import com.nageoffer.ai.ragent.rag.dao.entity.EvalScoreDO;
import com.nageoffer.ai.ragent.rag.dao.entity.EvalSuiteDO;
import com.nageoffer.ai.ragent.rag.dao.entity.EvalTrialDO;
import com.nageoffer.ai.ragent.rag.dao.mapper.EvalCaseMapper;
import com.nageoffer.ai.ragent.rag.dao.mapper.EvalRunMapper;
import com.nageoffer.ai.ragent.rag.dao.mapper.EvalScoreMapper;
import com.nageoffer.ai.ragent.rag.dao.mapper.EvalSuiteMapper;
import com.nageoffer.ai.ragent.rag.dao.mapper.EvalTrialMapper;
import com.nageoffer.ai.ragent.rag.eval.EvalRequests.CaseCreateRequest;
import com.nageoffer.ai.ragent.rag.eval.EvalRequests.CasePageRequest;
import com.nageoffer.ai.ragent.rag.eval.EvalRequests.CaseUpdateRequest;
import com.nageoffer.ai.ragent.rag.eval.EvalRequests.RunCreateRequest;
import com.nageoffer.ai.ragent.rag.eval.EvalRequests.RunPageRequest;
import com.nageoffer.ai.ragent.rag.eval.EvalRequests.SuiteCreateRequest;
import com.nageoffer.ai.ragent.rag.eval.EvalRequests.SuitePageRequest;
import com.nageoffer.ai.ragent.rag.eval.EvalRequests.SuiteUpdateRequest;
import com.nageoffer.ai.ragent.rag.eval.EvalViews.CaseVO;
import com.nageoffer.ai.ragent.rag.eval.EvalViews.RunDetailVO;
import com.nageoffer.ai.ragent.rag.eval.EvalViews.RunVO;
import com.nageoffer.ai.ragent.rag.eval.EvalViews.ScoreVO;
import com.nageoffer.ai.ragent.rag.eval.EvalViews.SuiteVO;
import com.nageoffer.ai.ragent.rag.eval.EvalViews.TrialVO;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.stream.Collectors;

/**
 * 自动化评测后台服务。
 */
@Service
@RequiredArgsConstructor
public class EvalAdminServiceImpl implements EvalAdminService {

    private static final String STATUS_RUNNING = "RUNNING";
    private static final String STATUS_SUCCESS = "SUCCESS";
    private static final String STATUS_ERROR = "ERROR";

    private final EvalSuiteMapper suiteMapper;
    private final EvalCaseMapper caseMapper;
    private final EvalRunMapper runMapper;
    private final EvalTrialMapper trialMapper;
    private final EvalScoreMapper scoreMapper;
    private final EvalRagRunner evalRagRunner;
    private final EvalProperties evalProperties;
    private final ObjectMapper objectMapper;
    private final List<EvalScorer> scorers;

    @Qualifier("evalExecutionExecutor")
    private final Executor evalExecutionExecutor;

    @Override
    public IPage<SuiteVO> pageSuites(SuitePageRequest request) {
        SuitePageRequest safeRequest = request == null ? new SuitePageRequest() : request;
        String keyword = StrUtil.trimToNull(safeRequest.getKeyword());
        Page<EvalSuiteDO> page = new Page<>(safeRequest.getCurrent(), safeRequest.getSize());
        return suiteMapper.selectPage(
                page,
                Wrappers.lambdaQuery(EvalSuiteDO.class)
                        .and(StrUtil.isNotBlank(keyword), wrapper -> wrapper
                                .like(EvalSuiteDO::getName, keyword)
                                .or()
                                .like(EvalSuiteDO::getDescription, keyword)
                                .or()
                                .like(EvalSuiteDO::getVersion, keyword))
                        .orderByDesc(EvalSuiteDO::getUpdateTime)
        ).convert(this::toSuiteVO);
    }

    @Override
    public SuiteVO getSuite(String id) {
        return toSuiteVO(loadSuite(id));
    }

    @Override
    public String createSuite(SuiteCreateRequest request) {
        Assert.notNull(request, () -> new ClientException("请求不能为空"));
        String name = StrUtil.trimToNull(request.getName());
        Assert.notBlank(name, () -> new ClientException("评测套件名称不能为空"));
        EvalSuiteDO suite = EvalSuiteDO.builder()
                .name(name)
                .description(StrUtil.trimToNull(request.getDescription()))
                .version(StrUtil.blankToDefault(StrUtil.trimToNull(request.getVersion()), "v1"))
                .enabled(1)
                .build();
        suiteMapper.insert(suite);
        return suite.getId();
    }

    @Override
    public void updateSuite(String id, SuiteUpdateRequest request) {
        Assert.notNull(request, () -> new ClientException("请求不能为空"));
        EvalSuiteDO suite = loadSuite(id);
        if (request.getName() != null) {
            String name = StrUtil.trimToNull(request.getName());
            Assert.notBlank(name, () -> new ClientException("评测套件名称不能为空"));
            suite.setName(name);
        }
        if (request.getDescription() != null) {
            suite.setDescription(StrUtil.trimToNull(request.getDescription()));
        }
        if (request.getVersion() != null) {
            suite.setVersion(StrUtil.trimToNull(request.getVersion()));
        }
        if (request.getEnabled() != null) {
            suite.setEnabled(toEnabled(request.getEnabled()));
        }
        suiteMapper.updateById(suite);
    }

    @Override
    public void deleteSuite(String id) {
        EvalSuiteDO suite = loadSuite(id);
        suiteMapper.deleteById(suite.getId());
    }

    @Override
    public IPage<CaseVO> pageCases(CasePageRequest request) {
        CasePageRequest safeRequest = request == null ? new CasePageRequest() : request;
        String keyword = StrUtil.trimToNull(safeRequest.getKeyword());
        String category = StrUtil.trimToNull(safeRequest.getCategory());
        String suiteId = StrUtil.trimToNull(safeRequest.getSuiteId());
        Page<EvalCaseDO> page = new Page<>(safeRequest.getCurrent(), safeRequest.getSize());
        return caseMapper.selectPage(
                page,
                Wrappers.lambdaQuery(EvalCaseDO.class)
                        .eq(StrUtil.isNotBlank(suiteId), EvalCaseDO::getSuiteId, suiteId)
                        .eq(StrUtil.isNotBlank(category), EvalCaseDO::getCategory, category)
                        .and(StrUtil.isNotBlank(keyword), wrapper -> wrapper
                                .like(EvalCaseDO::getQuestion, keyword)
                                .or()
                                .like(EvalCaseDO::getCategory, keyword))
                        .orderByDesc(EvalCaseDO::getUpdateTime)
        ).convert(this::toCaseVO);
    }

    @Override
    public CaseVO getCase(String id) {
        return toCaseVO(loadCase(id));
    }

    @Override
    public String createCase(CaseCreateRequest request) {
        Assert.notNull(request, () -> new ClientException("请求不能为空"));
        EvalCaseDO evalCase = buildCase(request);
        caseMapper.insert(evalCase);
        return evalCase.getId();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public int importCases(String suiteId, List<CaseCreateRequest> requests) {
        Assert.notBlank(suiteId, () -> new ClientException("评测套件 ID 不能为空"));
        loadSuite(suiteId);
        if (CollUtil.isEmpty(requests)) {
            return 0;
        }
        int count = 0;
        for (CaseCreateRequest request : requests) {
            request.setSuiteId(suiteId);
            caseMapper.insert(buildCase(request));
            count++;
        }
        return count;
    }

    @Override
    public void updateCase(String id, CaseUpdateRequest request) {
        Assert.notNull(request, () -> new ClientException("请求不能为空"));
        EvalCaseDO evalCase = loadCase(id);
        if (request.getQuestion() != null) {
            String question = StrUtil.trimToNull(request.getQuestion());
            Assert.notBlank(question, () -> new ClientException("评测问题不能为空"));
            evalCase.setQuestion(question);
        }
        if (request.getCategory() != null) {
            evalCase.setCategory(StrUtil.trimToNull(request.getCategory()));
        }
        if (request.getGroundTruth() != null) {
            evalCase.setGroundTruth(normalizeGroundTruth(request.getGroundTruth()));
        }
        if (request.getEnabled() != null) {
            evalCase.setEnabled(toEnabled(request.getEnabled()));
        }
        caseMapper.updateById(evalCase);
    }

    @Override
    public void deleteCase(String id) {
        EvalCaseDO evalCase = loadCase(id);
        caseMapper.deleteById(evalCase.getId());
    }

    @Override
    public RunDetailVO runSuite(RunCreateRequest request) {
        Assert.notNull(request, () -> new ClientException("请求不能为空"));
        String suiteId = StrUtil.trimToNull(request.getSuiteId());
        Assert.notBlank(suiteId, () -> new ClientException("评测套件 ID 不能为空"));
        loadSuite(suiteId);
        int trialCount = resolveTrialCount(request.getTrialCount());
        List<EvalCaseDO> cases = loadRunnableCases(suiteId, request.getCaseIds());
        Assert.isTrue(CollUtil.isNotEmpty(cases), () -> new ClientException("没有可执行的评测用例"));

        int totalTrials = cases.size() * trialCount;
        Date startedAt = new Date();
        EvalRunDO run = EvalRunDO.builder()
                .suiteId(suiteId)
                .trialCount(trialCount)
                .status(STATUS_RUNNING)
                .totalCases(cases.size())
                .totalTrials(totalTrials)
                .completedTrials(0)
                .passedTrials(0)
                .successRate(0d)
                .passAtK(0d)
                .startedAt(startedAt)
                .build();
        runMapper.insert(run);

        List<CompletableFuture<EvalTrialDO>> futures = new ArrayList<>();
        for (EvalCaseDO evalCase : cases) {
            for (int trialIndex = 1; trialIndex <= trialCount; trialIndex++) {
                int currentTrial = trialIndex;
                futures.add(CompletableFuture.supplyAsync(
                        () -> executeTrial(run.getId(), suiteId, evalCase, currentTrial),
                        evalExecutionExecutor
                ));
            }
        }
        futures.forEach(CompletableFuture::join);

        finishRun(run, startedAt);
        return getRun(run.getId());
    }

    @Override
    public IPage<RunVO> pageRuns(RunPageRequest request) {
        RunPageRequest safeRequest = request == null ? new RunPageRequest() : request;
        String suiteId = StrUtil.trimToNull(safeRequest.getSuiteId());
        String status = StrUtil.trimToNull(safeRequest.getStatus());
        Page<EvalRunDO> page = new Page<>(safeRequest.getCurrent(), safeRequest.getSize());
        return runMapper.selectPage(
                page,
                Wrappers.lambdaQuery(EvalRunDO.class)
                        .eq(StrUtil.isNotBlank(suiteId), EvalRunDO::getSuiteId, suiteId)
                        .eq(StrUtil.isNotBlank(status), EvalRunDO::getStatus, status)
                        .orderByDesc(EvalRunDO::getStartedAt)
                        .orderByDesc(EvalRunDO::getCreateTime)
        ).convert(this::toRunVO);
    }

    @Override
    public RunDetailVO getRun(String runId) {
        EvalRunDO run = loadRun(runId);
        return RunDetailVO.builder()
                .run(toRunVO(run))
                .trials(listTrials(runId))
                .report(report(runId))
                .build();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteRun(String runId) {
        EvalRunDO run = loadRun(runId);
        Assert.isFalse(STATUS_RUNNING.equals(run.getStatus()), () -> new ClientException("运行中的评测不能删除"));
        scoreMapper.delete(Wrappers.lambdaQuery(EvalScoreDO.class)
                .eq(EvalScoreDO::getRunId, run.getId()));
        trialMapper.delete(Wrappers.lambdaQuery(EvalTrialDO.class)
                .eq(EvalTrialDO::getRunId, run.getId()));
        runMapper.deleteById(run.getId());
    }

    @Override
    public List<TrialVO> listTrials(String runId) {
        List<EvalTrialDO> trials = trialMapper.selectList(Wrappers.lambdaQuery(EvalTrialDO.class)
                .eq(EvalTrialDO::getRunId, runId)
                .orderByAsc(EvalTrialDO::getCaseId)
                .orderByAsc(EvalTrialDO::getTrialIndex));
        if (CollUtil.isEmpty(trials)) {
            return List.of();
        }
        List<EvalScoreDO> scores = scoreMapper.selectList(Wrappers.lambdaQuery(EvalScoreDO.class)
                .eq(EvalScoreDO::getRunId, runId)
                .orderByAsc(EvalScoreDO::getMetricName));
        Map<String, List<EvalScoreDO>> scoreMap = scores.stream()
                .collect(Collectors.groupingBy(EvalScoreDO::getTrialId));
        Set<String> caseIds = trials.stream()
                .map(EvalTrialDO::getCaseId)
                .filter(StrUtil::isNotBlank)
                .collect(Collectors.toSet());
        Map<String, EvalCaseDO> caseMap = caseIds.isEmpty()
                ? Map.of()
                : caseMapper.selectList(Wrappers.lambdaQuery(EvalCaseDO.class)
                        .in(EvalCaseDO::getId, caseIds))
                        .stream()
                        .collect(Collectors.toMap(EvalCaseDO::getId, item -> item, (left, right) -> left));
        return trials.stream()
                .map(trial -> toTrialVO(trial, caseMap.get(trial.getCaseId()), scoreMap.getOrDefault(trial.getId(), List.of())))
                .toList();
    }

    @Override
    public Map<String, Object> report(String runId) {
        EvalRunDO run = loadRun(runId);
        List<EvalTrialDO> trials = trialMapper.selectList(Wrappers.lambdaQuery(EvalTrialDO.class)
                .eq(EvalTrialDO::getRunId, runId));
        List<EvalScoreDO> scores = scoreMapper.selectList(Wrappers.lambdaQuery(EvalScoreDO.class)
                .eq(EvalScoreDO::getRunId, runId));

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("runId", runId);
        report.put("suiteId", run.getSuiteId());
        report.put("trialCount", run.getTrialCount());
        report.put("totalCases", run.getTotalCases());
        report.put("totalTrials", run.getTotalTrials());
        report.put("completedTrials", trials.size());
        report.put("passedTrials", countPassedTrials(trials));
        report.put("successRate", ratio(countPassedTrials(trials), trials.size()));
        report.put("passAtK", passAtK(trials));
        report.put("metricAverages", metricAverages(scores));
        report.put("failedCases", failedCaseIds(trials));
        return report;
    }

    private EvalCaseDO buildCase(CaseCreateRequest request) {
        Assert.notNull(request, () -> new ClientException("评测用例不能为空"));
        String suiteId = StrUtil.trimToNull(request.getSuiteId());
        String question = StrUtil.trimToNull(request.getQuestion());
        Assert.notBlank(suiteId, () -> new ClientException("评测套件 ID 不能为空"));
        Assert.notBlank(question, () -> new ClientException("评测问题不能为空"));
        loadSuite(suiteId);
        return EvalCaseDO.builder()
                .suiteId(suiteId)
                .question(question)
                .category(StrUtil.trimToNull(request.getCategory()))
                .groundTruth(normalizeGroundTruth(request.getGroundTruth()))
                .enabled(request.getEnabled() == null ? 1 : toEnabled(request.getEnabled()))
                .build();
    }

    private EvalTrialDO executeTrial(String runId, String suiteId, EvalCaseDO evalCase, int trialIndex) {
        Date startedAt = new Date();
        long startMillis = System.currentTimeMillis();
        EvalTrialDO trial = EvalTrialDO.builder()
                .runId(runId)
                .suiteId(suiteId)
                .caseId(evalCase.getId())
                .trialIndex(trialIndex)
                .status(STATUS_RUNNING)
                .success(0)
                .startedAt(startedAt)
                .build();
        trialMapper.insert(trial);

        try {
            EvalGroundTruth groundTruth = parseGroundTruth(evalCase.getGroundTruth());
            EvalSnapshot snapshot = evalRagRunner.run(evalCase.getQuestion());
            List<EvalMetricResult> metricResults = score(evalCase, groundTruth, snapshot);
            for (EvalMetricResult metricResult : metricResults) {
                scoreMapper.insert(toScoreDO(runId, suiteId, evalCase.getId(), trial.getId(), metricResult));
            }

            boolean success = metricResults.isEmpty() || metricResults.stream().allMatch(EvalMetricResult::isPassed);
            trial.setStatus(STATUS_SUCCESS);
            trial.setSuccess(success ? 1 : 0);
            trial.setTraceId(snapshot.getTraceId());
            trial.setLatencyMs(snapshot.getLatencyMs());
            trial.setSnapshotJson(toJson(snapshot));
            trial.setFinishedAt(new Date());
            trial.setDurationMs(System.currentTimeMillis() - startMillis);
            trialMapper.updateById(trial);
            return trial;
        } catch (Exception ex) {
            trial.setStatus(STATUS_ERROR);
            trial.setSuccess(0);
            trial.setErrorMessage(truncateError(ex));
            trial.setFinishedAt(new Date());
            trial.setDurationMs(System.currentTimeMillis() - startMillis);
            trialMapper.updateById(trial);
            return trial;
        }
    }

    private List<EvalMetricResult> score(EvalCaseDO evalCase, EvalGroundTruth groundTruth, EvalSnapshot snapshot) {
        if (CollUtil.isEmpty(scorers)) {
            return List.of();
        }
        return scorers.stream()
                .flatMap(scorer -> scorer.score(evalCase, groundTruth, snapshot).stream())
                .toList();
    }

    private void finishRun(EvalRunDO run, Date startedAt) {
        List<EvalTrialDO> trials = trialMapper.selectList(Wrappers.lambdaQuery(EvalTrialDO.class)
                .eq(EvalTrialDO::getRunId, run.getId()));
        long errorCount = trials.stream()
                .filter(trial -> STATUS_ERROR.equals(trial.getStatus()))
                .count();
        long passedTrials = countPassedTrials(trials);
        Map<String, Object> summary = report(run.getId());

        run.setStatus(errorCount > 0 ? STATUS_ERROR : STATUS_SUCCESS);
        run.setCompletedTrials(trials.size());
        run.setPassedTrials((int) passedTrials);
        run.setSuccessRate(ratio(passedTrials, trials.size()));
        run.setPassAtK(passAtK(trials));
        run.setFinishedAt(new Date());
        run.setDurationMs(System.currentTimeMillis() - startedAt.getTime());
        run.setSummaryJson(toJson(summary));
        runMapper.updateById(run);
    }

    private EvalScoreDO toScoreDO(String runId,
                                  String suiteId,
                                  String caseId,
                                  String trialId,
                                  EvalMetricResult metricResult) {
        return EvalScoreDO.builder()
                .runId(runId)
                .suiteId(suiteId)
                .caseId(caseId)
                .trialId(trialId)
                .metricName(metricResult.getMetricName())
                .scoreValue(metricResult.getScore())
                .passed(metricResult.isPassed() ? 1 : 0)
                .reason(metricResult.getReason())
                .detailJson(toJson(metricResult.getDetail()))
                .build();
    }

    private EvalGroundTruth parseGroundTruth(String groundTruthJson) throws JsonProcessingException {
        if (StrUtil.isBlank(groundTruthJson)) {
            return new EvalGroundTruth();
        }
        return objectMapper.readValue(groundTruthJson, EvalGroundTruth.class);
    }

    private String normalizeGroundTruth(Object groundTruth) {
        if (groundTruth == null) {
            return null;
        }
        if (groundTruth instanceof String text) {
            return StrUtil.trimToNull(text);
        }
        return toJson(groundTruth);
    }

    private String toJson(Object value) {
        if (value == null) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new ClientException("JSON 序列化失败: " + ex.getMessage());
        }
    }

    private List<EvalCaseDO> loadRunnableCases(String suiteId, List<String> caseIds) {
        return caseMapper.selectList(Wrappers.lambdaQuery(EvalCaseDO.class)
                .eq(EvalCaseDO::getSuiteId, suiteId)
                .eq(EvalCaseDO::getEnabled, 1)
                .in(CollUtil.isNotEmpty(caseIds), EvalCaseDO::getId, caseIds)
                .orderByAsc(EvalCaseDO::getId));
    }

    private EvalSuiteDO loadSuite(String id) {
        EvalSuiteDO suite = suiteMapper.selectOne(Wrappers.lambdaQuery(EvalSuiteDO.class)
                .eq(EvalSuiteDO::getId, id));
        Assert.notNull(suite, () -> new ClientException("评测套件不存在"));
        return suite;
    }

    private EvalCaseDO loadCase(String id) {
        EvalCaseDO evalCase = caseMapper.selectOne(Wrappers.lambdaQuery(EvalCaseDO.class)
                .eq(EvalCaseDO::getId, id));
        Assert.notNull(evalCase, () -> new ClientException("评测用例不存在"));
        return evalCase;
    }

    private EvalRunDO loadRun(String id) {
        EvalRunDO run = runMapper.selectOne(Wrappers.lambdaQuery(EvalRunDO.class)
                .eq(EvalRunDO::getId, id));
        Assert.notNull(run, () -> new ClientException("评测运行不存在"));
        return run;
    }

    private int resolveTrialCount(Integer trialCount) {
        int resolved = trialCount == null ? evalProperties.getExecution().getDefaultTrialCount() : trialCount;
        return Math.max(1, resolved);
    }

    private int toEnabled(Integer enabled) {
        return enabled != null && enabled > 0 ? 1 : 0;
    }

    private long countPassedTrials(List<EvalTrialDO> trials) {
        if (CollUtil.isEmpty(trials)) {
            return 0;
        }
        return trials.stream()
                .filter(trial -> Objects.equals(trial.getSuccess(), 1))
                .count();
    }

    private double passAtK(List<EvalTrialDO> trials) {
        if (CollUtil.isEmpty(trials)) {
            return 0d;
        }
        Map<String, List<EvalTrialDO>> byCase = trials.stream()
                .collect(Collectors.groupingBy(EvalTrialDO::getCaseId));
        long passedCases = byCase.values().stream()
                .filter(caseTrials -> caseTrials.stream().anyMatch(trial -> Objects.equals(trial.getSuccess(), 1)))
                .count();
        return ratio(passedCases, byCase.size());
    }

    private double ratio(long numerator, long denominator) {
        if (denominator <= 0) {
            return 0d;
        }
        return numerator * 1.0d / denominator;
    }

    private Map<String, Double> metricAverages(List<EvalScoreDO> scores) {
        if (CollUtil.isEmpty(scores)) {
            return Map.of();
        }
        return scores.stream()
                .collect(Collectors.groupingBy(
                        EvalScoreDO::getMetricName,
                        LinkedHashMap::new,
                        Collectors.averagingDouble(score -> score.getScoreValue() == null ? 0d : score.getScoreValue())
                ));
    }

    private List<String> failedCaseIds(List<EvalTrialDO> trials) {
        if (CollUtil.isEmpty(trials)) {
            return List.of();
        }
        Map<String, List<EvalTrialDO>> byCase = trials.stream()
                .collect(Collectors.groupingBy(EvalTrialDO::getCaseId));
        return byCase.entrySet().stream()
                .filter(entry -> entry.getValue().stream().noneMatch(trial -> Objects.equals(trial.getSuccess(), 1)))
                .map(Map.Entry::getKey)
                .distinct()
                .toList();
    }

    private String truncateError(Exception ex) {
        String message = ex.getClass().getSimpleName() + ": " + StrUtil.blankToDefault(ex.getMessage(), "");
        return message.length() <= 1000 ? message : message.substring(0, 1000);
    }

    private SuiteVO toSuiteVO(EvalSuiteDO suite) {
        return SuiteVO.builder()
                .id(suite.getId())
                .name(suite.getName())
                .description(suite.getDescription())
                .version(suite.getVersion())
                .enabled(suite.getEnabled())
                .createTime(suite.getCreateTime())
                .updateTime(suite.getUpdateTime())
                .build();
    }

    private CaseVO toCaseVO(EvalCaseDO evalCase) {
        return CaseVO.builder()
                .id(evalCase.getId())
                .suiteId(evalCase.getSuiteId())
                .question(evalCase.getQuestion())
                .category(evalCase.getCategory())
                .groundTruth(evalCase.getGroundTruth())
                .enabled(evalCase.getEnabled())
                .createTime(evalCase.getCreateTime())
                .updateTime(evalCase.getUpdateTime())
                .build();
    }

    private RunVO toRunVO(EvalRunDO run) {
        return RunVO.builder()
                .id(run.getId())
                .suiteId(run.getSuiteId())
                .trialCount(run.getTrialCount())
                .status(run.getStatus())
                .totalCases(run.getTotalCases())
                .totalTrials(run.getTotalTrials())
                .completedTrials(run.getCompletedTrials())
                .passedTrials(run.getPassedTrials())
                .successRate(run.getSuccessRate())
                .passAtK(run.getPassAtK())
                .startedAt(run.getStartedAt())
                .finishedAt(run.getFinishedAt())
                .durationMs(run.getDurationMs())
                .errorMessage(run.getErrorMessage())
                .build();
    }

    private TrialVO toTrialVO(EvalTrialDO trial, EvalCaseDO evalCase, List<EvalScoreDO> scores) {
        List<ScoreVO> scoreViews = scores.stream()
                .sorted(Comparator.comparing(EvalScoreDO::getMetricName))
                .map(this::toScoreVO)
                .toList();
        return TrialVO.builder()
                .id(trial.getId())
                .runId(trial.getRunId())
                .suiteId(trial.getSuiteId())
                .caseId(trial.getCaseId())
                .caseQuestion(evalCase == null ? null : evalCase.getQuestion())
                .caseCategory(evalCase == null ? null : evalCase.getCategory())
                .trialIndex(trial.getTrialIndex())
                .status(trial.getStatus())
                .success(trial.getSuccess())
                .traceId(trial.getTraceId())
                .latencyMs(trial.getLatencyMs())
                .startedAt(trial.getStartedAt())
                .finishedAt(trial.getFinishedAt())
                .durationMs(trial.getDurationMs())
                .snapshotJson(trial.getSnapshotJson())
                .errorMessage(trial.getErrorMessage())
                .scores(scoreViews)
                .build();
    }

    private ScoreVO toScoreVO(EvalScoreDO score) {
        return ScoreVO.builder()
                .id(score.getId())
                .runId(score.getRunId())
                .trialId(score.getTrialId())
                .suiteId(score.getSuiteId())
                .caseId(score.getCaseId())
                .metricName(score.getMetricName())
                .scoreValue(score.getScoreValue())
                .passed(score.getPassed())
                .reason(score.getReason())
                .detailJson(score.getDetailJson())
                .build();
    }
}
