import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Check,
  FileText,
  GitBranch,
  Loader2,
  RefreshCw,
  Rocket,
  Sparkles,
  Undo2,
  X
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { RelativeTime } from "@/components/RelativeTime";
import { cn } from "@/lib/utils";
import {
  createConfigBootstrapRun,
  getConfigBootstrapRun,
  getConfigBootstrapRuns,
  publishConfigBootstrapRun,
  reviewIntentCandidate,
  reviewTermCandidate,
  rollbackConfigBootstrapRun,
  type ConfigBootstrapPublishResult,
  type ConfigBootstrapRun,
  type ConfigBootstrapSampleDocument,
  type IntentNodeCandidate,
  type PageResult,
  type TermMappingCandidate
} from "@/services/configBootstrapService";
import { getKnowledgeBases, type KnowledgeBase } from "@/services/knowledgeService";
import { getErrorMessage } from "@/utils/error";

const PAGE_SIZE = 8;

const statusText: Record<string, string> = {
  RUNNING: "运行中",
  COMPLETED: "待审核",
  FAILED: "失败",
  APPROVED: "已通过",
  REJECTED: "已拒绝",
  PUBLISHED: "已发布",
  ROLLED_BACK: "已回滚",
  PENDING: "待审核"
};

const levelText: Record<number, string> = {
  0: "DOMAIN",
  1: "CATEGORY",
  2: "TOPIC"
};

const kindText: Record<number, string> = {
  0: "KB",
  1: "SYSTEM",
  2: "MCP"
};

const badgeVariant = (status?: string | null) => {
  const normalized = (status || "").toUpperCase();
  if (normalized === "FAILED" || normalized === "REJECTED") return "destructive";
  if (normalized === "COMPLETED" || normalized === "APPROVED" || normalized === "PUBLISHED") return "default";
  if (normalized === "RUNNING" || normalized === "PENDING") return "secondary";
  return "outline";
};

const riskVariant = (risk?: string | null) => {
  const normalized = (risk || "").toUpperCase();
  if (normalized === "HIGH") return "destructive";
  if (normalized === "MEDIUM") return "secondary";
  return "outline";
};

const formatScore = (value?: number | null) => {
  if (value === null || value === undefined) return "-";
  return `${Math.round(value * 100)}%`;
};

const isTimeoutError = (error: unknown) => {
  if (!error || typeof error !== "object") {
    return false;
  }
  const maybeError = error as { code?: unknown; message?: unknown };
  return (
    maybeError.code === "ECONNABORTED" ||
    (typeof maybeError.message === "string" && maybeError.message.toLowerCase().includes("timeout"))
  );
};

const parseKbIds = (value?: string | null) => {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.map((item) => String(item)).filter(Boolean) : [];
  } catch {
    return [];
  }
};

const parseSampleDocuments = (value?: string | null): ConfigBootstrapSampleDocument[] => {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

const countByStatus = <T extends { status?: string | null }>(items: T[], status: string) => {
  return items.filter((item) => (item.status || "").toUpperCase() === status).length;
};

const isReviewLocked = (status?: string | null) => {
  const normalized = (status || "").toUpperCase();
  return normalized === "PUBLISHED" || normalized === "ROLLED_BACK";
};

function EvidenceList({ values }: { values?: string[] }) {
  const items = (values || []).filter(Boolean).slice(0, 2);
  if (items.length === 0) {
    return <span className="text-muted-foreground">-</span>;
  }
  return (
    <div className="space-y-1 text-xs text-muted-foreground">
      {items.map((item, index) => (
        <div key={`${item}-${index}`} className="line-clamp-2" title={item}>
          {item}
        </div>
      ))}
    </div>
  );
}

function CandidateStatusBadge({ status }: { status?: string | null }) {
  const normalized = (status || "").toUpperCase();
  return <Badge variant={badgeVariant(normalized)}>{statusText[normalized] || status || "-"}</Badge>;
}

function PublishResultLine({ result }: { result: ConfigBootstrapPublishResult | null }) {
  if (!result) return null;
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
      发布术语 {result.publishedTermMappings ?? 0} 条，发布意图 {result.publishedIntentNodes ?? 0} 条，
      跳过术语 {result.skippedTermMappings ?? 0} 条，跳过意图 {result.skippedIntentNodes ?? 0} 条，
      回滚术语 {result.rolledBackTermMappings ?? 0} 条，回滚意图 {result.rolledBackIntentNodes ?? 0} 条
    </div>
  );
}

function SampleDocumentsList({ items }: { items: ConfigBootstrapSampleDocument[] }) {
  if (items.length === 0) return null;
  return (
    <div className="rounded-lg border border-slate-200">
      <div className="border-b border-slate-200 px-4 py-3 text-sm font-medium text-slate-700">
        采样明细
      </div>
      <div className="max-h-[360px] divide-y divide-slate-100 overflow-auto">
        {items.map((item, index) => {
          const chunks = item.chunks || [];
          return (
            <div key={item.docId || `${item.docName}-${index}`} className="space-y-2 px-4 py-3">
              <div className="min-w-0">
                <div
                  className="truncate text-sm font-medium text-slate-700"
                  title={item.docName || ""}
                >
                  {item.docName || item.docId || "-"}
                </div>
                <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
                  <span>{item.kbName || item.kbId || "未知知识库"}</span>
                  {item.collectionName ? <span>{item.collectionName}</span> : null}
                  <span>{chunks.length} chunks</span>
                </div>
              </div>
              <div className="space-y-2">
                {chunks.map((chunk, chunkIndex) => (
                  <div
                    key={chunk.chunkId || `${item.docId}-${chunkIndex}`}
                    className="rounded-md bg-slate-50 px-3 py-2"
                  >
                    <div className="mb-1 flex flex-wrap gap-2 text-xs text-slate-500">
                      <span>chunk-{chunk.chunkIndex ?? chunkIndex}</span>
                      {chunk.chunkId ? <span className="font-mono">{chunk.chunkId}</span> : null}
                    </div>
                    <div
                      className="line-clamp-2 break-words text-xs leading-5 text-slate-600"
                      title={chunk.content || ""}
                    >
                      {chunk.content || "-"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ConfigBootstrapPage() {
  const [runsPage, setRunsPage] = useState<PageResult<ConfigBootstrapRun> | null>(null);
  const [runPageNo, setRunPageNo] = useState(1);
  const [currentRun, setCurrentRun] = useState<ConfigBootstrapRun | null>(null);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [selectedKbIds, setSelectedKbIds] = useState<string[]>([]);
  const [maxDocuments, setMaxDocuments] = useState(100);
  const [maxChunksPerDocument, setMaxChunksPerDocument] = useState(3);
  const [useLlm, setUseLlm] = useState(true);
  const [enablePublishedCandidates, setEnablePublishedCandidates] = useState(false);
  const [reviewComment, setReviewComment] = useState("");
  const [activeTab, setActiveTab] = useState<"terms" | "intents">("terms");
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingRunDetail, setLoadingRunDetail] = useState(false);
  const [creating, setCreating] = useState(false);
  const [reviewingId, setReviewingId] = useState<string | null>(null);
  const [publishing, setPublishing] = useState(false);
  const [rollingBack, setRollingBack] = useState(false);
  const [publishResult, setPublishResult] = useState<ConfigBootstrapPublishResult | null>(null);

  const termMappings = useMemo(() => currentRun?.termMappings || [], [currentRun?.termMappings]);
  const intentNodes = useMemo(() => currentRun?.intentNodes || [], [currentRun?.intentNodes]);
  const sampleDocuments = useMemo(() => parseSampleDocuments(currentRun?.sampleJson), [currentRun?.sampleJson]);
  const selectedAllKb = selectedKbIds.length === 0;

  const stats = useMemo(() => {
    const pending = countByStatus(termMappings, "PENDING") + countByStatus(intentNodes, "PENDING");
    const approved = countByStatus(termMappings, "APPROVED") + countByStatus(intentNodes, "APPROVED");
    const rejected = countByStatus(termMappings, "REJECTED") + countByStatus(intentNodes, "REJECTED");
    const published = countByStatus(termMappings, "PUBLISHED") + countByStatus(intentNodes, "PUBLISHED");
    return { pending, approved, rejected, published };
  }, [intentNodes, termMappings]);

  const loadRuns = useCallback(async (pageNo = 1) => {
    try {
      setLoadingRuns(true);
      const data = await getConfigBootstrapRuns(pageNo, PAGE_SIZE);
      setRunsPage(data);
    } catch (error) {
      toast.error(getErrorMessage(error, "加载初始化任务失败"));
      console.error(error);
    } finally {
      setLoadingRuns(false);
    }
  }, []);

  const loadRunDetail = useCallback(async (runId: string, clearPublishResult = true) => {
    try {
      setLoadingRunDetail(true);
      const data = await getConfigBootstrapRun(runId);
      setCurrentRun(data);
      if (clearPublishResult) {
        setPublishResult(null);
      }
    } catch (error) {
      toast.error(getErrorMessage(error, "加载候选详情失败"));
      console.error(error);
    } finally {
      setLoadingRunDetail(false);
    }
  }, []);

  const loadKnowledgeBases = useCallback(async () => {
    try {
      const data = await getKnowledgeBases(1, 200);
      setKnowledgeBases(data || []);
    } catch (error) {
      toast.error(getErrorMessage(error, "加载知识库失败"));
      console.error(error);
    }
  }, []);

  useEffect(() => {
    loadKnowledgeBases();
  }, [loadKnowledgeBases]);

  useEffect(() => {
    loadRuns(runPageNo);
  }, [loadRuns, runPageNo]);

  useEffect(() => {
    if (!currentRun && runsPage?.records.length) {
      loadRunDetail(runsPage.records[0].id);
    }
  }, [currentRun, loadRunDetail, runsPage]);

  const toggleKb = (kbId: string, checked: boolean) => {
    setSelectedKbIds((prev) => {
      if (checked) {
        return prev.includes(kbId) ? prev : [...prev, kbId];
      }
      return prev.filter((item) => item !== kbId);
    });
  };

  const handleCreate = async () => {
    if (maxDocuments <= 0 || maxChunksPerDocument <= 0) {
      toast.error("采样文档数和每篇 chunk 数必须大于 0");
      return;
    }

    const toastId = toast.loading("正在生成候选，LLM 开启时可能需要几分钟，请稍候...");
    try {
      setCreating(true);
      const run = await createConfigBootstrapRun({
        kbIds: selectedKbIds,
        maxDocuments,
        maxChunksPerDocument,
        useLlm
      });
      setCurrentRun(run);
      setRunPageNo(1);
      await loadRuns(1);
      toast.success(run.status === "FAILED" ? "任务执行失败，请查看错误信息" : "候选生成完成", {
        id: toastId
      });
    } catch (error) {
      if (isTimeoutError(error)) {
        toast.warning(
          "候选生成等待超时，后端任务可能仍在运行。已刷新任务列表，请稍后再点刷新任务查看结果。",
          {
            id: toastId,
            duration: 8000
          }
        );
        await loadRuns(1);
      } else {
        toast.error(getErrorMessage(error, "创建初始化任务失败"), {
          id: toastId
        });
      }
      console.error(error);
    } finally {
      setCreating(false);
    }
  };

  const refreshCurrentRun = async () => {
    if (!currentRun) return;
    await loadRunDetail(currentRun.id, false);
    await loadRuns(runPageNo);
  };

  const handleReviewTerm = async (candidate: TermMappingCandidate, status: "APPROVED" | "REJECTED") => {
    try {
      setReviewingId(candidate.id);
      await reviewTermCandidate(candidate.id, {
        status,
        reviewComment: reviewComment.trim() || null
      });
      toast.success(status === "APPROVED" ? "术语候选已通过" : "术语候选已拒绝");
      await refreshCurrentRun();
    } catch (error) {
      toast.error(getErrorMessage(error, "审核术语候选失败"));
      console.error(error);
    } finally {
      setReviewingId(null);
    }
  };

  const handleReviewIntent = async (candidate: IntentNodeCandidate, status: "APPROVED" | "REJECTED") => {
    try {
      setReviewingId(candidate.id);
      await reviewIntentCandidate(candidate.id, {
        status,
        reviewComment: reviewComment.trim() || null
      });
      toast.success(status === "APPROVED" ? "意图候选已通过" : "意图候选已拒绝");
      await refreshCurrentRun();
    } catch (error) {
      toast.error(getErrorMessage(error, "审核意图候选失败"));
      console.error(error);
    } finally {
      setReviewingId(null);
    }
  };

  const handlePublish = async () => {
    if (!currentRun) return;
    if (stats.approved === 0) {
      toast.error("请先通过至少一条候选");
      return;
    }

    try {
      setPublishing(true);
      const result = await publishConfigBootstrapRun(currentRun.id, {
        enablePublishedCandidates
      });
      setPublishResult(result);
      toast.success("已发布审核通过的候选");
      await refreshCurrentRun();
    } catch (error) {
      toast.error(getErrorMessage(error, "发布候选失败"));
      console.error(error);
    } finally {
      setPublishing(false);
    }
  };

  const handleRollback = async () => {
    if (!currentRun) return;
    try {
      setRollingBack(true);
      const result = await rollbackConfigBootstrapRun(currentRun.id);
      setPublishResult(result);
      toast.success("已回滚本次发布的配置");
      await refreshCurrentRun();
    } catch (error) {
      toast.error(getErrorMessage(error, "回滚失败"));
      console.error(error);
    } finally {
      setRollingBack(false);
    }
  };

  const runKbIds = parseKbIds(currentRun?.kbIds);
  const candidateCount = termMappings.length + intentNodes.length;

  return (
    <div className="admin-page">
      <div className="admin-page-header">
        <div>
          <h1 className="admin-page-title">AI 配置初始化</h1>
          <p className="admin-page-subtitle">从已入库文档生成术语映射和意图树候选，并通过人工审核后发布</p>
        </div>
        <div className="admin-page-actions">
          <Button variant="outline" onClick={() => loadRuns(runPageNo)} disabled={loadingRuns}>
            <RefreshCw className={cn("h-4 w-4", loadingRuns && "animate-spin")} />
            刷新任务
          </Button>
          <Button className="admin-primary-gradient" onClick={handleCreate} disabled={creating}>
            {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            {creating ? "生成中..." : "生成候选"}
          </Button>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[360px_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">采样配置</CardTitle>
              <CardDescription>留空知识库范围时分析全部知识库</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">最大文档数</label>
                  <Input
                    type="number"
                    min={1}
                    max={500}
                    value={maxDocuments}
                    onChange={(event) => setMaxDocuments(Number(event.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-700">每篇 chunk</label>
                  <Input
                    type="number"
                    min={1}
                    max={10}
                    value={maxChunksPerDocument}
                    onChange={(event) => setMaxChunksPerDocument(Number(event.target.value))}
                  />
                </div>
              </div>

              <label className="flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-700">
                <Checkbox checked={useLlm} onCheckedChange={(checked) => setUseLlm(Boolean(checked))} />
                调用 LLM 生成候选
              </label>

              {creating ? (
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                  正在生成候选，LLM 开启时可能需要数分钟。任务完成前候选详情不会刷新，可稍后使用刷新任务查看进度。
                </div>
              ) : null}

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-slate-700">知识库范围</span>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedKbIds([])}>
                    全部
                  </Button>
                </div>
                <div className="max-h-[220px] space-y-2 overflow-auto rounded-lg border border-slate-200 p-2">
                  {knowledgeBases.length === 0 ? (
                    <div className="py-4 text-center text-sm text-muted-foreground">暂无知识库</div>
                  ) : (
                    knowledgeBases.map((kb) => {
                      const checked = selectedKbIds.includes(kb.id);
                      return (
                        <label key={kb.id} className="flex items-start gap-2 rounded-md px-2 py-1.5 hover:bg-slate-50">
                          <Checkbox checked={checked} onCheckedChange={(value) => toggleKb(kb.id, Boolean(value))} />
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-sm font-medium text-slate-700">{kb.name}</span>
                            <span className="block truncate text-xs text-muted-foreground">{kb.collectionName || kb.id}</span>
                          </span>
                        </label>
                      );
                    })
                  )}
                </div>
                <p className="text-xs text-muted-foreground">
                  当前范围：{selectedAllKb ? "全部知识库" : `${selectedKbIds.length} 个知识库`}
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">最近任务</CardTitle>
              <CardDescription>选择任务查看候选详情</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {loadingRuns && !runsPage ? (
                <div className="py-6 text-center text-sm text-muted-foreground">加载中...</div>
              ) : (runsPage?.records || []).length === 0 ? (
                <div className="py-6 text-center text-sm text-muted-foreground">暂无初始化任务</div>
              ) : (
                <div className="space-y-2">
                  {(runsPage?.records || []).map((run) => {
                    const active = currentRun?.id === run.id;
                    return (
                      <button
                        key={run.id}
                        type="button"
                        onClick={() => loadRunDetail(run.id)}
                        className={cn(
                          "w-full rounded-lg border px-3 py-2 text-left transition hover:bg-slate-50",
                          active ? "border-primary bg-primary/5" : "border-slate-200"
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate text-sm font-medium text-slate-700">{run.id}</span>
                          <CandidateStatusBadge status={run.status} />
                        </div>
                        <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
                          <span>术语 {run.termCandidateCount ?? 0}</span>
                          <span>意图 {run.intentCandidateCount ?? 0}</span>
                          <RelativeTime value={run.createTime} />
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
              {runsPage ? (
                <div className="flex items-center justify-between text-sm text-slate-500">
                  <span>共 {runsPage.total} 次</span>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={runsPage.current <= 1}
                      onClick={() => setRunPageNo((prev) => Math.max(1, prev - 1))}
                    >
                      上一页
                    </Button>
                    <span>{runsPage.current} / {runsPage.pages || 1}</span>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={runsPage.current >= runsPage.pages}
                      onClick={() => setRunPageNo((prev) => Math.min(runsPage.pages || 1, prev + 1))}
                    >
                      下一页
                    </Button>
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <CardTitle className="text-base">候选详情</CardTitle>
                  <CardDescription>
                    {currentRun ? `任务 ${currentRun.id}` : "生成或选择一次任务后查看候选"}
                  </CardDescription>
                </div>
                {currentRun ? <CandidateStatusBadge status={currentRun.status} /> : null}
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {!currentRun ? (
                <div className="py-16 text-center text-sm text-muted-foreground">暂无候选详情</div>
              ) : loadingRunDetail ? (
                <div className="py-16 text-center text-sm text-muted-foreground">加载中...</div>
              ) : (
                <>
                  <div className="grid gap-3 md:grid-cols-4">
                    {[
                      { label: "采样文档", value: currentRun.documentCount ?? 0, icon: FileText },
                      { label: "采样 chunk", value: currentRun.chunkCount ?? 0, icon: FileText },
                      { label: "术语候选", value: termMappings.length, icon: Sparkles },
                      { label: "意图候选", value: intentNodes.length, icon: GitBranch }
                    ].map((item) => (
                      <div key={item.label} className="rounded-lg border border-slate-200 px-4 py-3">
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <item.icon className="h-4 w-4" />
                          {item.label}
                        </div>
                        <div className="mt-2 text-2xl font-semibold tabular-nums text-slate-800">{item.value}</div>
                      </div>
                    ))}
                  </div>

                  {currentRun.errorMessage ? (
                    <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                      {currentRun.errorMessage}
                    </div>
                  ) : null}

                  {currentRun.summary ? (
                    <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                      {currentRun.summary}
                    </div>
                  ) : null}

                  <SampleDocumentsList items={sampleDocuments} />

                  <div className="flex flex-wrap items-center gap-2 text-sm text-slate-500">
                    <Badge variant="secondary">待审核 {stats.pending}</Badge>
                    <Badge variant="default">已通过 {stats.approved}</Badge>
                    <Badge variant="outline">已拒绝 {stats.rejected}</Badge>
                    <Badge variant="outline">已发布 {stats.published}</Badge>
                    <span>LLM：{currentRun.useLlm ? "启用" : "未启用"}</span>
                    <span>范围：{runKbIds.length === 0 ? "全部知识库" : `${runKbIds.length} 个知识库`}</span>
                  </div>

                  <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_260px]">
                    <Textarea
                      value={reviewComment}
                      onChange={(event) => setReviewComment(event.target.value)}
                      placeholder="审核备注，可选"
                      className="min-h-[76px]"
                    />
                    <div className="space-y-3 rounded-lg border border-slate-200 p-3">
                      <label className="flex items-center gap-2 text-sm text-slate-700">
                        <Checkbox
                          checked={enablePublishedCandidates}
                          onCheckedChange={(checked) => setEnablePublishedCandidates(Boolean(checked))}
                        />
                        发布后立即启用
                      </label>
                      <div className="flex flex-wrap gap-2">
                        <Button size="sm" onClick={handlePublish} disabled={publishing || candidateCount === 0}>
                          {publishing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Rocket className="h-4 w-4" />}
                          发布通过项
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={handleRollback}
                          disabled={rollingBack || currentRun.status !== "PUBLISHED"}
                        >
                          {rollingBack ? <Loader2 className="h-4 w-4 animate-spin" /> : <Undo2 className="h-4 w-4" />}
                          回滚
                        </Button>
                      </div>
                    </div>
                  </div>

                  <PublishResultLine result={publishResult} />

                  <div className="flex gap-2 border-b border-slate-200">
                    {[
                      { key: "terms", label: `术语映射 ${termMappings.length}` },
                      { key: "intents", label: `意图节点 ${intentNodes.length}` }
                    ].map((tab) => (
                      <button
                        key={tab.key}
                        type="button"
                        onClick={() => setActiveTab(tab.key as "terms" | "intents")}
                        className={cn(
                          "border-b-2 px-3 py-2 text-sm font-medium transition",
                          activeTab === tab.key
                            ? "border-primary text-primary"
                            : "border-transparent text-muted-foreground hover:text-slate-700"
                        )}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>

                  {activeTab === "terms" ? (
                    <TermCandidateTable
                      items={termMappings}
                      reviewingId={reviewingId}
                      onReview={handleReviewTerm}
                    />
                  ) : (
                    <IntentCandidateTable
                      items={intentNodes}
                      reviewingId={reviewingId}
                      onReview={handleReviewIntent}
                    />
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function TermCandidateTable({
  items,
  reviewingId,
  onReview
}: {
  items: TermMappingCandidate[];
  reviewingId: string | null;
  onReview: (candidate: TermMappingCandidate, status: "APPROVED" | "REJECTED") => void;
}) {
  if (items.length === 0) {
    return <div className="py-10 text-center text-sm text-muted-foreground">暂无术语映射候选</div>;
  }

  return (
    <Table className="min-w-[960px]">
      <TableHeader>
        <TableRow>
          <TableHead className="w-[150px]">原始词</TableHead>
          <TableHead className="w-[150px]">目标词</TableHead>
          <TableHead className="w-[90px]">置信度</TableHead>
          <TableHead className="w-[90px]">风险</TableHead>
          <TableHead className="w-[100px]">状态</TableHead>
          <TableHead>证据</TableHead>
          <TableHead className="w-[170px]">更新时间</TableHead>
          <TableHead className="w-[170px]">操作</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {items.map((item) => (
          <TableRow key={item.id}>
            <TableCell className="font-medium">{item.sourceTerm}</TableCell>
            <TableCell>{item.targetTerm}</TableCell>
            <TableCell className="tabular-nums">{formatScore(item.confidence)}</TableCell>
            <TableCell>
              <Badge variant={riskVariant(item.riskLevel)}>{item.riskLevel || "-"}</Badge>
            </TableCell>
            <TableCell>
              <CandidateStatusBadge status={item.status} />
            </TableCell>
            <TableCell>
              <EvidenceList values={item.evidence} />
            </TableCell>
            <TableCell>
              <RelativeTime value={item.updateTime || item.createTime} />
            </TableCell>
            <TableCell>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={reviewingId === item.id || isReviewLocked(item.status)}
                  onClick={() => onReview(item, "APPROVED")}
                >
                  <Check className="h-4 w-4" />
                  通过
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-destructive hover:text-destructive"
                  disabled={reviewingId === item.id || isReviewLocked(item.status)}
                  onClick={() => onReview(item, "REJECTED")}
                >
                  <X className="h-4 w-4" />
                  拒绝
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function IntentCandidateTable({
  items,
  reviewingId,
  onReview
}: {
  items: IntentNodeCandidate[];
  reviewingId: string | null;
  onReview: (candidate: IntentNodeCandidate, status: "APPROVED" | "REJECTED") => void;
}) {
  if (items.length === 0) {
    return <div className="py-10 text-center text-sm text-muted-foreground">暂无意图节点候选</div>;
  }

  return (
    <Table className="min-w-[1180px]">
      <TableHeader>
        <TableRow>
          <TableHead className="w-[150px]">名称</TableHead>
          <TableHead className="w-[180px]">编码</TableHead>
          <TableHead className="w-[120px]">层级/类型</TableHead>
          <TableHead className="w-[160px]">父节点</TableHead>
          <TableHead className="w-[160px]">知识库</TableHead>
          <TableHead className="w-[90px]">置信度</TableHead>
          <TableHead className="w-[90px]">风险</TableHead>
          <TableHead className="w-[100px]">状态</TableHead>
          <TableHead>示例/证据</TableHead>
          <TableHead className="w-[170px]">操作</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {items.map((item) => (
          <TableRow key={item.id}>
            <TableCell className="font-medium">
              <div className="max-w-[150px] truncate" title={item.name}>
                {item.name}
              </div>
            </TableCell>
            <TableCell>
              <div className="max-w-[170px] truncate font-mono text-xs" title={item.intentCode}>
                {item.intentCode}
              </div>
            </TableCell>
            <TableCell>
              <div className="space-y-1 text-xs">
                <Badge variant="secondary">{levelText[item.level ?? -1] || item.level || "-"}</Badge>
                <div className="text-muted-foreground">{kindText[item.kind ?? -1] || item.kind || "-"}</div>
              </div>
            </TableCell>
            <TableCell className="max-w-[150px] truncate" title={item.parentCode || ""}>
              {item.parentCode || "-"}
            </TableCell>
            <TableCell>
              <div className="max-w-[150px] truncate text-xs" title={item.collectionName || item.kbId || ""}>
                {item.collectionName || item.kbId || "-"}
              </div>
              <div className="text-xs text-muted-foreground">topK {item.topK ?? "-"}</div>
            </TableCell>
            <TableCell className="tabular-nums">{formatScore(item.confidence)}</TableCell>
            <TableCell>
              <Badge variant={riskVariant(item.riskLevel)}>{item.riskLevel || "-"}</Badge>
            </TableCell>
            <TableCell>
              <CandidateStatusBadge status={item.status} />
            </TableCell>
            <TableCell>
              <EvidenceList values={[...(item.examples || []), ...(item.evidence || [])]} />
            </TableCell>
            <TableCell>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={reviewingId === item.id || isReviewLocked(item.status)}
                  onClick={() => onReview(item, "APPROVED")}
                >
                  <Check className="h-4 w-4" />
                  通过
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-destructive hover:text-destructive"
                  disabled={reviewingId === item.id || isReviewLocked(item.status)}
                  onClick={() => onReview(item, "REJECTED")}
                >
                  <X className="h-4 w-4" />
                  拒绝
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
