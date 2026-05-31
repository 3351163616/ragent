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

import java.util.List;
import java.util.Map;

public interface EvalAdminService {

    IPage<SuiteVO> pageSuites(SuitePageRequest request);

    SuiteVO getSuite(String id);

    String createSuite(SuiteCreateRequest request);

    void updateSuite(String id, SuiteUpdateRequest request);

    void deleteSuite(String id);

    IPage<CaseVO> pageCases(CasePageRequest request);

    CaseVO getCase(String id);

    String createCase(CaseCreateRequest request);

    int importCases(String suiteId, List<CaseCreateRequest> requests);

    void updateCase(String id, CaseUpdateRequest request);

    void deleteCase(String id);

    RunDetailVO runSuite(RunCreateRequest request);

    IPage<RunVO> pageRuns(RunPageRequest request);

    RunDetailVO getRun(String runId);

    void deleteRun(String runId);

    List<TrialVO> listTrials(String runId);

    Map<String, Object> report(String runId);
}
