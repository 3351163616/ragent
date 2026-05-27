import { api } from "@/services/api";

const CONFIG_BOOTSTRAP_CREATE_TIMEOUT_MS = 10 * 60 * 1000;

export interface PageResult<T> {
  records: T[];
  total: number;
  size: number;
  current: number;
  pages: number;
}

export interface ConfigBootstrapCreatePayload {
  kbIds?: string[];
  maxDocuments?: number;
  maxChunksPerDocument?: number;
  useLlm?: boolean;
}

export interface ConfigBootstrapReviewPayload {
  status: "APPROVED" | "REJECTED";
  reviewComment?: string | null;
}

export interface ConfigBootstrapPublishPayload {
  enablePublishedCandidates?: boolean;
}

export interface TermMappingCandidate {
  id: string;
  runId: string;
  sourceTerm: string;
  targetTerm: string;
  confidence?: number | null;
  riskLevel?: string | null;
  evidence?: string[];
  status: string;
  reviewComment?: string | null;
  publishedId?: string | null;
  createTime?: string | null;
  updateTime?: string | null;
}

export interface IntentNodeCandidate {
  id: string;
  runId: string;
  intentCode: string;
  name: string;
  level?: number | null;
  parentCode?: string | null;
  description?: string | null;
  examples?: string[];
  kbId?: string | null;
  collectionName?: string | null;
  topK?: number | null;
  kind?: number | null;
  sortOrder?: number | null;
  confidence?: number | null;
  riskLevel?: string | null;
  evidence?: string[];
  status: string;
  reviewComment?: string | null;
  publishedId?: string | null;
  createTime?: string | null;
  updateTime?: string | null;
}

export interface ConfigBootstrapSampleChunk {
  chunkId?: string | null;
  chunkIndex?: number | null;
  content?: string | null;
}

export interface ConfigBootstrapSampleDocument {
  kbId?: string | null;
  kbName?: string | null;
  collectionName?: string | null;
  docId?: string | null;
  docName?: string | null;
  chunks?: ConfigBootstrapSampleChunk[];
}

export interface ConfigBootstrapRun {
  id: string;
  status: string;
  kbIds?: string | null;
  useLlm?: boolean | null;
  documentCount?: number | null;
  chunkCount?: number | null;
  termCandidateCount?: number | null;
  intentCandidateCount?: number | null;
  summary?: string | null;
  sampleJson?: string | null;
  errorMessage?: string | null;
  createTime?: string | null;
  updateTime?: string | null;
  termMappings?: TermMappingCandidate[];
  intentNodes?: IntentNodeCandidate[];
}

export interface ConfigBootstrapPublishResult {
  runId: string;
  publishedTermMappings?: number | null;
  publishedIntentNodes?: number | null;
  skippedTermMappings?: number | null;
  skippedIntentNodes?: number | null;
  rolledBackTermMappings?: number | null;
  rolledBackIntentNodes?: number | null;
}

export async function getConfigBootstrapRuns(
  current = 1,
  size = 10,
  status?: string
): Promise<PageResult<ConfigBootstrapRun>> {
  return api.get<PageResult<ConfigBootstrapRun>, PageResult<ConfigBootstrapRun>>("/config-bootstrap/runs", {
    params: { current, size, status: status || undefined }
  });
}

export async function createConfigBootstrapRun(
  payload: ConfigBootstrapCreatePayload
): Promise<ConfigBootstrapRun> {
  return api.post<ConfigBootstrapRun, ConfigBootstrapRun>("/config-bootstrap/runs", payload, {
    timeout: CONFIG_BOOTSTRAP_CREATE_TIMEOUT_MS
  });
}

export async function getConfigBootstrapRun(runId: string): Promise<ConfigBootstrapRun> {
  return api.get<ConfigBootstrapRun, ConfigBootstrapRun>(`/config-bootstrap/runs/${runId}`);
}

export async function reviewTermCandidate(
  candidateId: string,
  payload: ConfigBootstrapReviewPayload
): Promise<void> {
  await api.put(`/config-bootstrap/term-candidates/${candidateId}/review`, payload);
}

export async function reviewIntentCandidate(
  candidateId: string,
  payload: ConfigBootstrapReviewPayload
): Promise<void> {
  await api.put(`/config-bootstrap/intent-candidates/${candidateId}/review`, payload);
}

export async function publishConfigBootstrapRun(
  runId: string,
  payload: ConfigBootstrapPublishPayload
): Promise<ConfigBootstrapPublishResult> {
  return api.post<ConfigBootstrapPublishResult, ConfigBootstrapPublishResult>(
    `/config-bootstrap/runs/${runId}/publish`,
    payload
  );
}

export async function rollbackConfigBootstrapRun(runId: string): Promise<ConfigBootstrapPublishResult> {
  return api.post<ConfigBootstrapPublishResult, ConfigBootstrapPublishResult>(
    `/config-bootstrap/runs/${runId}/rollback`
  );
}
