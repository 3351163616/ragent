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

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.nageoffer.ai.ragent.framework.convention.Result;
import com.nageoffer.ai.ragent.framework.web.Results;
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
import com.nageoffer.ai.ragent.rag.eval.EvalViews.SuiteVO;
import com.nageoffer.ai.ragent.rag.eval.EvalViews.TrialVO;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

/**
 * 自动化评测后台接口。
 */
@RestController
@RequiredArgsConstructor
@RequestMapping("/admin/eval")
public class EvalAdminController {

    private final EvalAdminService evalAdminService;

    @GetMapping("/suites")
    public Result<IPage<SuiteVO>> pageSuites(SuitePageRequest request) {
        return Results.success(evalAdminService.pageSuites(request));
    }

    @GetMapping("/suites/{id}")
    public Result<SuiteVO> getSuite(@PathVariable String id) {
        return Results.success(evalAdminService.getSuite(id));
    }

    @PostMapping("/suites")
    public Result<String> createSuite(@RequestBody SuiteCreateRequest request) {
        return Results.success(evalAdminService.createSuite(request));
    }

    @PutMapping("/suites/{id}")
    public Result<Void> updateSuite(@PathVariable String id, @RequestBody SuiteUpdateRequest request) {
        evalAdminService.updateSuite(id, request);
        return Results.success();
    }

    @DeleteMapping("/suites/{id}")
    public Result<Void> deleteSuite(@PathVariable String id) {
        evalAdminService.deleteSuite(id);
        return Results.success();
    }

    @GetMapping("/cases")
    public Result<IPage<CaseVO>> pageCases(CasePageRequest request) {
        return Results.success(evalAdminService.pageCases(request));
    }

    @GetMapping("/cases/{id}")
    public Result<CaseVO> getCase(@PathVariable String id) {
        return Results.success(evalAdminService.getCase(id));
    }

    @PostMapping("/cases")
    public Result<String> createCase(@RequestBody CaseCreateRequest request) {
        return Results.success(evalAdminService.createCase(request));
    }

    @PostMapping("/suites/{suiteId}/cases/import")
    public Result<Integer> importCases(@PathVariable String suiteId, @RequestBody List<CaseCreateRequest> requests) {
        return Results.success(evalAdminService.importCases(suiteId, requests));
    }

    @PutMapping("/cases/{id}")
    public Result<Void> updateCase(@PathVariable String id, @RequestBody CaseUpdateRequest request) {
        evalAdminService.updateCase(id, request);
        return Results.success();
    }

    @DeleteMapping("/cases/{id}")
    public Result<Void> deleteCase(@PathVariable String id) {
        evalAdminService.deleteCase(id);
        return Results.success();
    }

    @PostMapping("/runs")
    public Result<RunDetailVO> runSuite(@RequestBody RunCreateRequest request) {
        return Results.success(evalAdminService.runSuite(request));
    }

    @GetMapping("/runs")
    public Result<IPage<RunVO>> pageRuns(RunPageRequest request) {
        return Results.success(evalAdminService.pageRuns(request));
    }

    @GetMapping("/runs/{runId}")
    public Result<RunDetailVO> getRun(@PathVariable String runId) {
        return Results.success(evalAdminService.getRun(runId));
    }

    @DeleteMapping("/runs/{runId}")
    public Result<Void> deleteRun(@PathVariable String runId) {
        evalAdminService.deleteRun(runId);
        return Results.success();
    }

    @GetMapping("/runs/{runId}/trials")
    public Result<List<TrialVO>> listTrials(@PathVariable String runId) {
        return Results.success(evalAdminService.listTrials(runId));
    }

    @GetMapping("/runs/{runId}/report")
    public Result<Map<String, Object>> report(@PathVariable String runId) {
        return Results.success(evalAdminService.report(runId));
    }
}
