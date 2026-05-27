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

package com.nageoffer.ai.ragent.rag.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.nageoffer.ai.ragent.framework.convention.Result;
import com.nageoffer.ai.ragent.framework.web.Results;
import com.nageoffer.ai.ragent.rag.controller.request.ConfigBootstrapCreateRequest;
import com.nageoffer.ai.ragent.rag.controller.request.ConfigBootstrapPublishRequest;
import com.nageoffer.ai.ragent.rag.controller.request.ConfigBootstrapReviewRequest;
import com.nageoffer.ai.ragent.rag.controller.request.ConfigBootstrapRunPageRequest;
import com.nageoffer.ai.ragent.rag.controller.vo.ConfigBootstrapPublishVO;
import com.nageoffer.ai.ragent.rag.controller.vo.ConfigBootstrapRunVO;
import com.nageoffer.ai.ragent.rag.service.ConfigBootstrapService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class ConfigBootstrapController {

    private final ConfigBootstrapService configBootstrapService;

    /**
     * 从现有知识库文档生成术语映射和意图节点候选。
     */
    @PostMapping("/config-bootstrap/runs")
    public Result<ConfigBootstrapRunVO> createRun(@RequestBody(required = false) ConfigBootstrapCreateRequest requestParam) {
        return Results.success(configBootstrapService.createRun(requestParam));
    }

    /**
     * 分页查看初始化配置候选生成任务。
     */
    @GetMapping("/config-bootstrap/runs")
    public Result<IPage<ConfigBootstrapRunVO>> pageRuns(ConfigBootstrapRunPageRequest requestParam) {
        return Results.success(configBootstrapService.pageRuns(requestParam));
    }

    /**
     * 查看一次初始化配置候选生成结果。
     */
    @GetMapping("/config-bootstrap/runs/{runId}")
    public Result<ConfigBootstrapRunVO> queryRun(@PathVariable String runId) {
        return Results.success(configBootstrapService.queryRun(runId));
    }

    /**
     * 审核术语映射候选，status 支持 APPROVED / REJECTED。
     */
    @PutMapping("/config-bootstrap/term-candidates/{candidateId}/review")
    public Result<Void> reviewTermCandidate(@PathVariable String candidateId,
                                            @RequestBody ConfigBootstrapReviewRequest requestParam) {
        configBootstrapService.reviewTermCandidate(candidateId, requestParam);
        return Results.success();
    }

    /**
     * 审核意图节点候选，status 支持 APPROVED / REJECTED。
     */
    @PutMapping("/config-bootstrap/intent-candidates/{candidateId}/review")
    public Result<Void> reviewIntentCandidate(@PathVariable String candidateId,
                                              @RequestBody ConfigBootstrapReviewRequest requestParam) {
        configBootstrapService.reviewIntentCandidate(candidateId, requestParam);
        return Results.success();
    }

    /**
     * 发布已审核通过的候选到正式配置表。
     */
    @PostMapping("/config-bootstrap/runs/{runId}/publish")
    public Result<ConfigBootstrapPublishVO> publishRun(@PathVariable String runId,
                                                       @RequestBody(required = false) ConfigBootstrapPublishRequest requestParam) {
        return Results.success(configBootstrapService.publishRun(runId, requestParam));
    }

    /**
     * 回滚本次发布：将本 run 发布出的正式配置置为停用。
     */
    @PostMapping("/config-bootstrap/runs/{runId}/rollback")
    public Result<ConfigBootstrapPublishVO> rollbackRun(@PathVariable String runId) {
        return Results.success(configBootstrapService.rollbackRun(runId));
    }
}
