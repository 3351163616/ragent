import { api } from "@/services/api";
import type { PageResult } from "@/services/ragTraceService";

export interface EvalSuite {
  id: string;
  name: string;
  description?: string | null;
  version?: string | null;
  enabled?: number | null;
  createTime?: string | null;
  updateTime?: string | null;
}

export interface EvalCase {
  id: string;
  suiteId: string;
  question: string;
  category?: string | null;
  groundTruth?: string | null;
  enabled?: number | null;
  createTime?: string | null;
  updateTime?: string | null;
}

export interface EvalRun {
  id: string;
  suiteId: string;
  trialCount?: number | null;
  status?: string | null;
  totalCases?: number | null;
  totalTrials?: number | null;
  completedTrials?: number | null;
  passedTrials?: number | null;
  successRate?: number | null;
  passAtK?: number | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  durationMs?: number | null;
  errorMessage?: string | null;
}

export interface EvalScore {
  id: string;
  runId: string;
  trialId: string;
  suiteId: string;
  caseId: string;
  metricName: string;
  scoreValue?: number | null;
  passed?: number | null;
  reason?: string | null;
  detailJson?: string | null;
}

export interface EvalTrial {
  id: string;
  runId: string;
  suiteId: string;
  caseId: string;
  caseQuestion?: string | null;
  caseCategory?: string | null;
  trialIndex?: number | null;
  status?: string | null;
  success?: number | null;
  traceId?: string | null;
  latencyMs?: number | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  durationMs?: number | null;
  snapshotJson?: string | null;
  errorMessage?: string | null;
  scores?: EvalScore[];
}

export interface EvalRunDetail {
  run: EvalRun;
  trials: EvalTrial[];
  report: EvalRunReport;
}

export interface EvalRunReport {
  runId?: string;
  suiteId?: string;
  trialCount?: number;
  totalCases?: number;
  totalTrials?: number;
  completedTrials?: number;
  passedTrials?: number;
  successRate?: number;
  passAtK?: number;
  metricAverages?: Record<string, number>;
  failedCases?: string[];
}

export interface EvalSuitePayload {
  name: string;
  description?: string | null;
  version?: string | null;
}

export interface EvalCasePayload {
  suiteId: string;
  question: string;
  category?: string | null;
  groundTruth?: string | null;
  enabled?: number | null;
}

export interface EvalRunPayload {
  suiteId: string;
  trialCount?: number;
  caseIds?: string[];
}

export async function getEvalSuites(params: {
  current?: number;
  size?: number;
  keyword?: string;
} = {}): Promise<PageResult<EvalSuite>> {
  return api.get<PageResult<EvalSuite>, PageResult<EvalSuite>>("/admin/eval/suites", {
    params: {
      current: params.current ?? 1,
      size: params.size ?? 20,
      keyword: params.keyword || undefined
    }
  });
}

export async function createEvalSuite(payload: EvalSuitePayload): Promise<string> {
  return api.post<string, string>("/admin/eval/suites", payload);
}

export async function updateEvalSuite(id: string, payload: Partial<EvalSuitePayload> & { enabled?: number }): Promise<void> {
  return api.put<void, void>(`/admin/eval/suites/${id}`, payload);
}

export async function deleteEvalSuite(id: string): Promise<void> {
  return api.delete<void, void>(`/admin/eval/suites/${id}`);
}

export async function getEvalCases(params: {
  current?: number;
  size?: number;
  suiteId?: string;
  keyword?: string;
  category?: string;
} = {}): Promise<PageResult<EvalCase>> {
  return api.get<PageResult<EvalCase>, PageResult<EvalCase>>("/admin/eval/cases", {
    params: {
      current: params.current ?? 1,
      size: params.size ?? 10,
      suiteId: params.suiteId || undefined,
      keyword: params.keyword || undefined,
      category: params.category || undefined
    }
  });
}

export async function createEvalCase(payload: EvalCasePayload): Promise<string> {
  return api.post<string, string>("/admin/eval/cases", payload);
}

export async function updateEvalCase(id: string, payload: Partial<Omit<EvalCasePayload, "suiteId">>): Promise<void> {
  return api.put<void, void>(`/admin/eval/cases/${id}`, payload);
}

export async function deleteEvalCase(id: string): Promise<void> {
  return api.delete<void, void>(`/admin/eval/cases/${id}`);
}

export async function runEvalSuite(payload: EvalRunPayload): Promise<EvalRunDetail> {
  return api.post<EvalRunDetail, EvalRunDetail>("/admin/eval/runs", payload);
}

export async function getEvalRuns(params: {
  current?: number;
  size?: number;
  suiteId?: string;
  status?: string;
} = {}): Promise<PageResult<EvalRun>> {
  return api.get<PageResult<EvalRun>, PageResult<EvalRun>>("/admin/eval/runs", {
    params: {
      current: params.current ?? 1,
      size: params.size ?? 10,
      suiteId: params.suiteId || undefined,
      status: params.status || undefined
    }
  });
}

export async function getEvalRun(runId: string): Promise<EvalRunDetail> {
  return api.get<EvalRunDetail, EvalRunDetail>(`/admin/eval/runs/${runId}`);
}

export async function deleteEvalRun(runId: string): Promise<void> {
  return api.delete<void, void>(`/admin/eval/runs/${runId}`);
}
