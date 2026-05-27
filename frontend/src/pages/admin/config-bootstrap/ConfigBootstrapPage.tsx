import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Database,
  FileText,
  GitBranch,
  ListChecks,
  Loader2,
  RefreshCw,
  Rocket,
  Shuffle,
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { PageSizeSelect } from "@/components/admin/PageSizeSelect";
import { PAGE_SIZE_OPTIONS } from "@/components/admin/pagination";
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

type ReviewStatus = "APPROVED" | "REJECTED";
type CandidateGroup = "terms" | "intents";

const badgeVariant = (status?: string | null) => {
  const normalized = (status || "").toUpperCase();
  if (normalized === "FAILED" || normalized === "REJECTED") return "destructive";
  if (normalized === "COMPLETED" || normalized === "APPROVED" || normalized === "PUBLISHED")
    return "default";
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

const getReviewableCandidateIds = <T extends { id: string; status?: string | null }>(items: T[]) =>
  items.filter((item) => !isReviewLocked(item.status)).map((item) => item.id);

const reviewActionText: Record<ReviewStatus, string> = {
  APPROVED: "通过",
  REJECTED: "拒绝"
};

const generationSourceText: Record<string, string> = {
  RULE: "规则",
  LLM: "LLM",
  UNKNOWN: "未知"
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
  return (
    <Badge variant={badgeVariant(normalized)}>{statusText[normalized] || status || "-"}</Badge>
  );
}

function GenerationSourceBadge({ source }: { source?: string | null }) {
  const normalized = (source || "UNKNOWN").toUpperCase();
  return (
    <Badge
      variant={normalized === "LLM" ? "default" : normalized === "RULE" ? "secondary" : "outline"}
    >
      {generationSourceText[normalized] || source || "未知"}
    </Badge>
  );
}

function PublishResultLine({ result }: { result: ConfigBootstrapPublishResult | null }) {
  if (!result) return null;
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
      发布术语 {result.publishedTermMappings ?? 0} 条，发布意图 {result.publishedIntentNodes ?? 0}{" "}
      条， 跳过术语 {result.skippedTermMappings ?? 0} 条，跳过意图 {result.skippedIntentNodes ?? 0}{" "}
      条， 回滚术语 {result.rolledBackTermMappings ?? 0} 条，回滚意图{" "}
      {result.rolledBackIntentNodes ?? 0} 条
    </div>
  );
}

function SampleDocumentsList({
  items,
  knowledgeBases
}: {
  items: ConfigBootstrapSampleDocument[];
  knowledgeBases: KnowledgeBase[];
}) {
  const [expandedDocKeys, setExpandedDocKeys] = useState<Set<string>>(new Set());

  if (items.length === 0) return null;
  const knowledgeBaseMap = new Map(knowledgeBases.map((kb) => [kb.id, kb]));
  const totalChunks = items.reduce((sum, item) => sum + (item.chunks?.length || 0), 0);
  const toggleDocumentChunks = (key: string) => {
    setExpandedDocKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };
  const groups = items.reduce<
    Array<{
      key: string;
      kbName: string;
      kbId?: string | null;
      collectionName?: string | null;
      totalDocumentCount?: number | null;
      totalChunkCount?: number | null;
      sampledDocumentChunkTotal: number;
      documents: ConfigBootstrapSampleDocument[];
      chunkCount: number;
    }>
  >((acc, item) => {
    const key = item.kbId || item.kbName || item.collectionName || "unknown";
    let group = acc.find((entry) => entry.key === key);
    if (!group) {
      const currentKnowledgeBase = item.kbId ? knowledgeBaseMap.get(item.kbId) : undefined;
      group = {
        key,
        kbName: item.kbName || currentKnowledgeBase?.name || item.kbId || "未知知识库",
        kbId: item.kbId,
        collectionName: item.collectionName || currentKnowledgeBase?.collectionName,
        totalDocumentCount: item.kbDocumentCount ?? currentKnowledgeBase?.documentCount ?? null,
        totalChunkCount: item.kbChunkCount ?? null,
        sampledDocumentChunkTotal: 0,
        documents: [],
        chunkCount: 0
      };
      acc.push(group);
    }
    if (item.kbChunkCount != null) {
      group.totalChunkCount = item.kbChunkCount;
    }
    group.documents.push(item);
    group.chunkCount += item.chunks?.length || 0;
    group.sampledDocumentChunkTotal += item.docChunkCount ?? 0;
    return acc;
  }, []);

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 px-4 py-3">
        <div>
          <div className="text-sm font-medium text-slate-800">采样来源</div>
          <div className="mt-0.5 text-xs text-muted-foreground">
            按知识库分组展示实际送入候选生成的文档和 chunk
          </div>
        </div>
        <div className="text-xs text-muted-foreground">
          {groups.length} 个知识库 / {items.length} 篇文档 / {totalChunks} 个 chunk
        </div>
      </div>
      <div className="max-h-[520px] space-y-4 overflow-auto p-4">
        {groups.map((group) => {
          const sampledDocumentCount = group.documents.length;
          const totalDocumentCount = group.totalDocumentCount;
          const documentRatio =
            totalDocumentCount && totalDocumentCount > 0
              ? Math.min(100, Math.round((sampledDocumentCount / totalDocumentCount) * 100))
              : null;
          const totalChunkCount =
            group.totalChunkCount ??
            (group.sampledDocumentChunkTotal > 0
              ? group.sampledDocumentChunkTotal
              : group.chunkCount);
          const chunkRatio =
            totalChunkCount && totalChunkCount > 0
              ? Math.min(100, Math.round((group.chunkCount / totalChunkCount) * 100))
              : null;
          return (
            <section
              key={group.key}
              className="overflow-hidden rounded-lg border border-slate-200 bg-slate-50/70"
            >
              <div className="border-b border-slate-200 bg-white px-4 py-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="flex min-w-0 gap-3">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                      <Database className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                      <div className="text-xs text-muted-foreground">来源知识库</div>
                      <div
                        className="truncate text-base font-semibold text-slate-800"
                        title={group.kbName}
                      >
                        {group.kbName}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-2">
                        {group.collectionName ? (
                          <Badge variant="outline">{group.collectionName}</Badge>
                        ) : null}
                        {group.kbId ? (
                          <span className="font-mono text-xs text-muted-foreground">
                            {group.kbId}
                          </span>
                        ) : null}
                      </div>
                    </div>
                  </div>
                  <div className="grid min-w-[180px] gap-2 text-xs text-muted-foreground">
                    <div>
                      <div className="flex items-center justify-between gap-3">
                        <span>文档</span>
                        <span className="font-medium text-slate-700">
                          {sampledDocumentCount}
                          {totalDocumentCount != null ? ` / ${totalDocumentCount}` : ""} 篇
                        </span>
                      </div>
                      {documentRatio != null ? (
                        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-100">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${documentRatio}%` }}
                          />
                        </div>
                      ) : null}
                    </div>
                    <div>
                      <div className="flex items-center justify-between gap-3">
                        <span>chunk</span>
                        <span className="font-medium text-slate-700">
                          {group.chunkCount}
                          {totalChunkCount != null ? ` / ${totalChunkCount}` : ""} 个
                        </span>
                      </div>
                      {chunkRatio != null ? (
                        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-100">
                          <div
                            className="h-full rounded-full bg-emerald-500"
                            style={{ width: `${chunkRatio}%` }}
                          />
                        </div>
                      ) : null}
                    </div>
                  </div>
                </div>
              </div>
              <div className="divide-y divide-slate-200">
                {group.documents.map((item, index) => {
                  const docKey =
                    item.docId || `${group.key}-${item.docName || "document"}-${index}`;
                  const chunks = item.chunks || [];
                  const sampledChunkCount = chunks.length;
                  const totalChunkCount =
                    item.docChunkCount ?? (sampledChunkCount > 0 ? sampledChunkCount : null);
                  const chunkLimit = item.chunkLimit;
                  const chunkTarget =
                    totalChunkCount != null && chunkLimit != null
                      ? Math.min(totalChunkCount, chunkLimit)
                      : chunkLimit;
                  const documentChunkRatio =
                    totalChunkCount && totalChunkCount > 0
                      ? Math.min(100, Math.round((sampledChunkCount / totalChunkCount) * 100))
                      : null;
                  const chunkFilled =
                    chunkTarget != null ? sampledChunkCount >= chunkTarget : false;
                  const chunkStatus =
                    totalChunkCount != null && sampledChunkCount >= totalChunkCount
                      ? "已覆盖全部 chunk"
                      : chunkLimit != null && sampledChunkCount >= chunkLimit
                        ? "已达每篇上限"
                        : chunkFilled
                          ? "已取满"
                          : "未取满";
                  return (
                    <div key={item.docId || `${item.docName}-${index}`} className="px-4 py-3">
                      <div className="flex items-start gap-3">
                        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded bg-white text-slate-500">
                          <FileText className="h-4 w-4" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div
                            className="truncate text-sm font-medium text-slate-800"
                            title={item.docName || ""}
                          >
                            {item.docName || item.docId || "-"}
                          </div>
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                            {item.docId ? <span className="font-mono">{item.docId}</span> : null}
                            <button
                              type="button"
                              aria-expanded={expandedDocKeys.has(docKey)}
                              className={cn(
                                "inline-flex h-6 items-center gap-1 rounded border px-2 font-medium transition-colors",
                                expandedDocKeys.has(docKey)
                                  ? "border-primary/30 bg-primary/10 text-primary"
                                  : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50"
                              )}
                              onClick={() => toggleDocumentChunks(docKey)}
                            >
                              {expandedDocKeys.has(docKey) ? (
                                <ChevronDown className="h-3.5 w-3.5" />
                              ) : (
                                <ChevronRight className="h-3.5 w-3.5" />
                              )}
                              <span>
                                chunk {sampledChunkCount}
                                {totalChunkCount != null ? ` / ${totalChunkCount}` : ""} 个
                              </span>
                            </button>
                            {chunkLimit != null ? <span>上限 {chunkLimit}</span> : null}
                            <Badge variant={chunkStatus === "未取满" ? "outline" : "secondary"}>
                              {chunkStatus}
                            </Badge>
                          </div>
                          {documentChunkRatio != null ? (
                            <div className="mt-2 h-1.5 max-w-[260px] overflow-hidden rounded-full bg-slate-100">
                              <div
                                className="h-full rounded-full bg-emerald-500"
                                style={{ width: `${documentChunkRatio}%` }}
                              />
                            </div>
                          ) : null}
                        </div>
                      </div>
                      {expandedDocKeys.has(docKey) ? (
                        <div className="ml-10 mt-3 rounded-md border border-slate-200 bg-white px-3 py-2">
                          <div className="flex flex-wrap gap-2">
                            {chunks.length === 0 ? (
                              <span className="text-xs text-muted-foreground">未采样 chunk</span>
                            ) : (
                              chunks.map((chunk, chunkIndex) => (
                                <span
                                  key={chunk.chunkId || `${item.docId}-${chunkIndex}`}
                                  title={chunk.chunkId || ""}
                                >
                                  <Badge
                                    variant="secondary"
                                    className="rounded px-1.5 py-0 font-mono"
                                  >
                                    chunk-{chunk.chunkIndex ?? chunkIndex}
                                  </Badge>
                                </span>
                              ))
                            )}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}

export function ConfigBootstrapPage() {
  const [runsPage, setRunsPage] = useState<PageResult<ConfigBootstrapRun> | null>(null);
  const [runPageNo, setRunPageNo] = useState(1);
  const [runPageSize, setRunPageSize] = useState(PAGE_SIZE_OPTIONS[0]);
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
  const [batchReviewing, setBatchReviewing] = useState<{
    group: CandidateGroup;
    status: ReviewStatus;
  } | null>(null);
  const [publishing, setPublishing] = useState(false);
  const [rollingBack, setRollingBack] = useState(false);
  const [publishResult, setPublishResult] = useState<ConfigBootstrapPublishResult | null>(null);
  const [selectedTermIds, setSelectedTermIds] = useState<string[]>([]);
  const [selectedIntentIds, setSelectedIntentIds] = useState<string[]>([]);

  const termMappings = useMemo(() => currentRun?.termMappings || [], [currentRun?.termMappings]);
  const intentNodes = useMemo(() => currentRun?.intentNodes || [], [currentRun?.intentNodes]);
  const reviewableTermIds = useMemo(() => getReviewableCandidateIds(termMappings), [termMappings]);
  const reviewableIntentIds = useMemo(() => getReviewableCandidateIds(intentNodes), [intentNodes]);
  const reviewableTermIdSet = useMemo(() => new Set(reviewableTermIds), [reviewableTermIds]);
  const reviewableIntentIdSet = useMemo(() => new Set(reviewableIntentIds), [reviewableIntentIds]);
  const selectedTermIdSet = useMemo(() => new Set(selectedTermIds), [selectedTermIds]);
  const selectedIntentIdSet = useMemo(() => new Set(selectedIntentIds), [selectedIntentIds]);
  const selectedTerms = useMemo(
    () =>
      termMappings.filter(
        (item) => selectedTermIdSet.has(item.id) && reviewableTermIdSet.has(item.id)
      ),
    [reviewableTermIdSet, selectedTermIdSet, termMappings]
  );
  const selectedIntents = useMemo(
    () =>
      intentNodes.filter(
        (item) => selectedIntentIdSet.has(item.id) && reviewableIntentIdSet.has(item.id)
      ),
    [intentNodes, reviewableIntentIdSet, selectedIntentIdSet]
  );
  const sampleDocuments = useMemo(
    () => parseSampleDocuments(currentRun?.sampleJson),
    [currentRun?.sampleJson]
  );
  const selectedAllKb = selectedKbIds.length === 0;

  const stats = useMemo(() => {
    const pending = countByStatus(termMappings, "PENDING") + countByStatus(intentNodes, "PENDING");
    const approved =
      countByStatus(termMappings, "APPROVED") + countByStatus(intentNodes, "APPROVED");
    const rejected =
      countByStatus(termMappings, "REJECTED") + countByStatus(intentNodes, "REJECTED");
    const published =
      countByStatus(termMappings, "PUBLISHED") + countByStatus(intentNodes, "PUBLISHED");
    return { pending, approved, rejected, published };
  }, [intentNodes, termMappings]);

  const loadRuns = useCallback(
    async (pageNo = 1) => {
      try {
        setLoadingRuns(true);
        const data = await getConfigBootstrapRuns(pageNo, runPageSize);
        setRunsPage(data);
      } catch (error) {
        toast.error(getErrorMessage(error, "加载初始化任务失败"));
        console.error(error);
      } finally {
        setLoadingRuns(false);
      }
    },
    [runPageSize]
  );

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

  useEffect(() => {
    setSelectedTermIds((prev) => prev.filter((id) => reviewableTermIdSet.has(id)));
  }, [reviewableTermIdSet]);

  useEffect(() => {
    setSelectedIntentIds((prev) => prev.filter((id) => reviewableIntentIdSet.has(id)));
  }, [reviewableIntentIdSet]);

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

  const toggleTermSelect = (id: string, checked: boolean) => {
    if (!reviewableTermIdSet.has(id)) return;
    setSelectedTermIds((prev) => {
      if (checked) {
        return prev.includes(id) ? prev : [...prev, id];
      }
      return prev.filter((item) => item !== id);
    });
  };

  const toggleIntentSelect = (id: string, checked: boolean) => {
    if (!reviewableIntentIdSet.has(id)) return;
    setSelectedIntentIds((prev) => {
      if (checked) {
        return prev.includes(id) ? prev : [...prev, id];
      }
      return prev.filter((item) => item !== id);
    });
  };

  const invertTermSelection = () => {
    setSelectedTermIds((prev) => {
      const selected = new Set(prev);
      return reviewableTermIds.filter((id) => !selected.has(id));
    });
  };

  const invertIntentSelection = () => {
    setSelectedIntentIds((prev) => {
      const selected = new Set(prev);
      return reviewableIntentIds.filter((id) => !selected.has(id));
    });
  };

  const handleReviewTerm = async (candidate: TermMappingCandidate, status: ReviewStatus) => {
    try {
      setReviewingId(candidate.id);
      await reviewTermCandidate(candidate.id, {
        status,
        reviewComment: reviewComment.trim() || null
      });
      toast.success(status === "APPROVED" ? "术语候选已通过" : "术语候选已拒绝");
      setSelectedTermIds((prev) => prev.filter((id) => id !== candidate.id));
      await refreshCurrentRun();
    } catch (error) {
      toast.error(getErrorMessage(error, "审核术语候选失败"));
      console.error(error);
    } finally {
      setReviewingId(null);
    }
  };

  const handleReviewIntent = async (candidate: IntentNodeCandidate, status: ReviewStatus) => {
    try {
      setReviewingId(candidate.id);
      await reviewIntentCandidate(candidate.id, {
        status,
        reviewComment: reviewComment.trim() || null
      });
      toast.success(status === "APPROVED" ? "意图候选已通过" : "意图候选已拒绝");
      setSelectedIntentIds((prev) => prev.filter((id) => id !== candidate.id));
      await refreshCurrentRun();
    } catch (error) {
      toast.error(getErrorMessage(error, "审核意图候选失败"));
      console.error(error);
    } finally {
      setReviewingId(null);
    }
  };

  const handleBatchReviewTerms = async (status: ReviewStatus) => {
    if (selectedTerms.length === 0 || batchReviewing) return;

    const toastId = toast.loading(`正在批量${reviewActionText[status]}术语候选...`);
    try {
      setBatchReviewing({ group: "terms", status });
      const results = await Promise.allSettled(
        selectedTerms.map((candidate) =>
          reviewTermCandidate(candidate.id, {
            status,
            reviewComment: reviewComment.trim() || null
          })
        )
      );
      const failedIds = selectedTerms
        .filter((_, index) => results[index].status === "rejected")
        .map((candidate) => candidate.id);
      const successCount = selectedTerms.length - failedIds.length;
      setSelectedTermIds(failedIds);
      if (failedIds.length > 0) {
        toast.warning(
          `已${reviewActionText[status]} ${successCount} 条术语候选，失败 ${failedIds.length} 条`,
          { id: toastId }
        );
      } else {
        toast.success(`已${reviewActionText[status]} ${successCount} 条术语候选`, { id: toastId });
      }
      await refreshCurrentRun();
    } catch (error) {
      toast.error(getErrorMessage(error, "批量审核术语候选失败"), { id: toastId });
      console.error(error);
    } finally {
      setBatchReviewing(null);
    }
  };

  const handleBatchReviewIntents = async (status: ReviewStatus) => {
    if (selectedIntents.length === 0 || batchReviewing) return;

    const toastId = toast.loading(`正在批量${reviewActionText[status]}意图候选...`);
    try {
      setBatchReviewing({ group: "intents", status });
      const results = await Promise.allSettled(
        selectedIntents.map((candidate) =>
          reviewIntentCandidate(candidate.id, {
            status,
            reviewComment: reviewComment.trim() || null
          })
        )
      );
      const failedIds = selectedIntents
        .filter((_, index) => results[index].status === "rejected")
        .map((candidate) => candidate.id);
      const successCount = selectedIntents.length - failedIds.length;
      setSelectedIntentIds(failedIds);
      if (failedIds.length > 0) {
        toast.warning(
          `已${reviewActionText[status]} ${successCount} 条意图候选，失败 ${failedIds.length} 条`,
          { id: toastId }
        );
      } else {
        toast.success(`已${reviewActionText[status]} ${successCount} 条意图候选`, { id: toastId });
      }
      await refreshCurrentRun();
    } catch (error) {
      toast.error(getErrorMessage(error, "批量审核意图候选失败"), { id: toastId });
      console.error(error);
    } finally {
      setBatchReviewing(null);
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
          <p className="admin-page-subtitle">
            从已入库文档生成术语映射和意图树候选，并通过人工审核后发布
          </p>
        </div>
        <div className="admin-page-actions">
          <Button variant="outline" onClick={() => loadRuns(runPageNo)} disabled={loadingRuns}>
            <RefreshCw className={cn("h-4 w-4", loadingRuns && "animate-spin")} />
            刷新任务
          </Button>
          <Button className="admin-primary-gradient" onClick={handleCreate} disabled={creating}>
            {creating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
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
                    max={50}
                    value={maxChunksPerDocument}
                    onChange={(event) => setMaxChunksPerDocument(Number(event.target.value))}
                  />
                </div>
              </div>

              <label className="flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-700">
                <Checkbox
                  checked={useLlm}
                  onCheckedChange={(checked) => setUseLlm(Boolean(checked))}
                />
                调用 LLM 生成候选
              </label>

              {creating ? (
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                  正在生成候选，LLM
                  开启时可能需要数分钟。任务完成前候选详情不会刷新，可稍后使用刷新任务查看进度。
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
                        <label
                          key={kb.id}
                          className="flex items-start gap-2 rounded-md px-2 py-1.5 hover:bg-slate-50"
                        >
                          <Checkbox
                            checked={checked}
                            onCheckedChange={(value) => toggleKb(kb.id, Boolean(value))}
                          />
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-sm font-medium text-slate-700">
                              {kb.name}
                            </span>
                            <span className="block truncate text-xs text-muted-foreground">
                              {kb.collectionName || kb.id}
                            </span>
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
                          <span className="truncate text-sm font-medium text-slate-700">
                            {run.id}
                          </span>
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
                    <PageSizeSelect
                      value={runPageSize}
                      onChange={(value) => {
                        setRunPageSize(value);
                        setRunPageNo(1);
                      }}
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={runsPage.current <= 1}
                      onClick={() => setRunPageNo((prev) => Math.max(1, prev - 1))}
                    >
                      上一页
                    </Button>
                    <span>
                      {runsPage.current} / {runsPage.pages || 1}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={runsPage.current >= runsPage.pages}
                      onClick={() =>
                        setRunPageNo((prev) => Math.min(runsPage.pages || 1, prev + 1))
                      }
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
                      <div
                        key={item.label}
                        className="rounded-lg border border-slate-200 px-4 py-3"
                      >
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <item.icon className="h-4 w-4" />
                          {item.label}
                        </div>
                        <div className="mt-2 text-2xl font-semibold tabular-nums text-slate-800">
                          {item.value}
                        </div>
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

                  <SampleDocumentsList items={sampleDocuments} knowledgeBases={knowledgeBases} />

                  <div className="flex flex-wrap items-center gap-2 text-sm text-slate-500">
                    <Badge variant="secondary">待审核 {stats.pending}</Badge>
                    <Badge variant="default">已通过 {stats.approved}</Badge>
                    <Badge variant="outline">已拒绝 {stats.rejected}</Badge>
                    <Badge variant="outline">已发布 {stats.published}</Badge>
                    <span>LLM：{currentRun.useLlm ? "启用" : "未启用"}</span>
                    <span>
                      范围：{runKbIds.length === 0 ? "全部知识库" : `${runKbIds.length} 个知识库`}
                    </span>
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
                          onCheckedChange={(checked) =>
                            setEnablePublishedCandidates(Boolean(checked))
                          }
                        />
                        发布后立即启用
                      </label>
                      <div className="flex flex-wrap gap-2">
                        <Button
                          size="sm"
                          onClick={handlePublish}
                          disabled={publishing || candidateCount === 0}
                        >
                          {publishing ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Rocket className="h-4 w-4" />
                          )}
                          发布通过项
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={handleRollback}
                          disabled={rollingBack || currentRun.status !== "PUBLISHED"}
                        >
                          {rollingBack ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Undo2 className="h-4 w-4" />
                          )}
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
                      selectedIds={selectedTermIdSet}
                      reviewingId={reviewingId}
                      batchReviewingStatus={
                        batchReviewing?.group === "terms" ? batchReviewing.status : null
                      }
                      reviewDisabled={batchReviewing !== null || reviewingId !== null}
                      onToggleSelect={toggleTermSelect}
                      onSelectAll={() => setSelectedTermIds(reviewableTermIds)}
                      onInvertSelection={invertTermSelection}
                      onClearSelection={() => setSelectedTermIds([])}
                      onReview={handleReviewTerm}
                      onBatchReview={handleBatchReviewTerms}
                    />
                  ) : (
                    <IntentCandidateTable
                      items={intentNodes}
                      selectedIds={selectedIntentIdSet}
                      reviewingId={reviewingId}
                      batchReviewingStatus={
                        batchReviewing?.group === "intents" ? batchReviewing.status : null
                      }
                      reviewDisabled={batchReviewing !== null || reviewingId !== null}
                      onToggleSelect={toggleIntentSelect}
                      onSelectAll={() => setSelectedIntentIds(reviewableIntentIds)}
                      onInvertSelection={invertIntentSelection}
                      onClearSelection={() => setSelectedIntentIds([])}
                      onReview={handleReviewIntent}
                      onBatchReview={handleBatchReviewIntents}
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

function CandidateBulkReviewToolbar({
  label,
  reviewableCount,
  selectedCount,
  disabled,
  batchReviewingStatus,
  onSelectAll,
  onInvertSelection,
  onClearSelection,
  onBatchReview
}: {
  label: string;
  reviewableCount: number;
  selectedCount: number;
  disabled: boolean;
  batchReviewingStatus: ReviewStatus | null;
  onSelectAll: () => void;
  onInvertSelection: () => void;
  onClearSelection: () => void;
  onBatchReview: (status: ReviewStatus) => void;
}) {
  const hasSelected = selectedCount > 0;
  const controlsDisabled = disabled || reviewableCount === 0;
  const approveBusy = batchReviewingStatus === "APPROVED";
  const rejectBusy = batchReviewingStatus === "REJECTED";

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-medium text-slate-700">{label}</span>
        <span className="text-muted-foreground">
          可审核 {reviewableCount} 项，已选 {selectedCount} 项
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          className="h-8 px-3 text-xs"
          disabled={controlsDisabled}
          onClick={onSelectAll}
        >
          <ListChecks className="h-4 w-4" />
          全选
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="h-8 px-3 text-xs"
          disabled={controlsDisabled}
          onClick={onInvertSelection}
        >
          <Shuffle className="h-4 w-4" />
          反选
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="h-8 px-3 text-xs"
          disabled={disabled || !hasSelected}
          onClick={onClearSelection}
        >
          <X className="h-4 w-4" />
          清空
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="h-8 px-3 text-xs"
          disabled={disabled || !hasSelected}
          onClick={() => onBatchReview("APPROVED")}
        >
          {approveBusy ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Check className="h-4 w-4" />
          )}
          批量通过
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="h-8 px-3 text-xs text-destructive hover:text-destructive"
          disabled={disabled || !hasSelected}
          onClick={() => onBatchReview("REJECTED")}
        >
          {rejectBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <X className="h-4 w-4" />}
          批量拒绝
        </Button>
      </div>
    </div>
  );
}

function TermCandidateTable({
  items,
  selectedIds,
  reviewingId,
  batchReviewingStatus,
  reviewDisabled,
  onToggleSelect,
  onSelectAll,
  onInvertSelection,
  onClearSelection,
  onReview,
  onBatchReview
}: {
  items: TermMappingCandidate[];
  selectedIds: Set<string>;
  reviewingId: string | null;
  batchReviewingStatus: ReviewStatus | null;
  reviewDisabled: boolean;
  onToggleSelect: (id: string, checked: boolean) => void;
  onSelectAll: () => void;
  onInvertSelection: () => void;
  onClearSelection: () => void;
  onReview: (candidate: TermMappingCandidate, status: ReviewStatus) => void;
  onBatchReview: (status: ReviewStatus) => void;
}) {
  if (items.length === 0) {
    return <div className="py-10 text-center text-sm text-muted-foreground">暂无术语映射候选</div>;
  }

  const reviewableItems = items.filter((item) => !isReviewLocked(item.status));
  const allSelected =
    reviewableItems.length > 0 && reviewableItems.every((item) => selectedIds.has(item.id));
  const someSelected = !allSelected && reviewableItems.some((item) => selectedIds.has(item.id));
  const selectedCount = reviewableItems.filter((item) => selectedIds.has(item.id)).length;

  return (
    <div className="space-y-3">
      <CandidateBulkReviewToolbar
        label="术语候选"
        reviewableCount={reviewableItems.length}
        selectedCount={selectedCount}
        disabled={reviewDisabled}
        batchReviewingStatus={batchReviewingStatus}
        onSelectAll={onSelectAll}
        onInvertSelection={onInvertSelection}
        onClearSelection={onClearSelection}
        onBatchReview={onBatchReview}
      />
      <Table className="min-w-[1110px]">
        <TableHeader>
          <TableRow>
            <TableHead className="w-[48px]">
              <Checkbox
                checked={allSelected ? true : someSelected ? "indeterminate" : false}
                onCheckedChange={(checked) =>
                  checked === true ? onSelectAll() : onClearSelection()
                }
                aria-label="全选术语候选"
                disabled={reviewDisabled || reviewableItems.length === 0}
              />
            </TableHead>
            <TableHead className="w-[150px]">原始词</TableHead>
            <TableHead className="w-[150px]">目标词</TableHead>
            <TableHead className="w-[90px]">置信度</TableHead>
            <TableHead className="w-[90px]">风险</TableHead>
            <TableHead className="w-[90px]">来源</TableHead>
            <TableHead className="w-[100px]">状态</TableHead>
            <TableHead>证据</TableHead>
            <TableHead className="w-[170px]">更新时间</TableHead>
            <TableHead className="w-[170px]">操作</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((item) => {
            const locked = isReviewLocked(item.status);
            return (
              <TableRow key={item.id}>
                <TableCell>
                  <Checkbox
                    checked={selectedIds.has(item.id)}
                    onCheckedChange={(checked) => onToggleSelect(item.id, checked === true)}
                    aria-label={`选择术语候选 ${item.sourceTerm}`}
                    disabled={reviewDisabled || locked}
                  />
                </TableCell>
                <TableCell className="font-medium">{item.sourceTerm}</TableCell>
                <TableCell>{item.targetTerm}</TableCell>
                <TableCell className="tabular-nums">{formatScore(item.confidence)}</TableCell>
                <TableCell>
                  <Badge variant={riskVariant(item.riskLevel)}>{item.riskLevel || "-"}</Badge>
                </TableCell>
                <TableCell>
                  <GenerationSourceBadge source={item.generationSource} />
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
                      disabled={reviewDisabled || reviewingId === item.id || locked}
                      onClick={() => onReview(item, "APPROVED")}
                    >
                      <Check className="h-4 w-4" />
                      通过
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive hover:text-destructive"
                      disabled={reviewDisabled || reviewingId === item.id || locked}
                      onClick={() => onReview(item, "REJECTED")}
                    >
                      <X className="h-4 w-4" />
                      拒绝
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

function IntentCandidateTable({
  items,
  selectedIds,
  reviewingId,
  batchReviewingStatus,
  reviewDisabled,
  onToggleSelect,
  onSelectAll,
  onInvertSelection,
  onClearSelection,
  onReview,
  onBatchReview
}: {
  items: IntentNodeCandidate[];
  selectedIds: Set<string>;
  reviewingId: string | null;
  batchReviewingStatus: ReviewStatus | null;
  reviewDisabled: boolean;
  onToggleSelect: (id: string, checked: boolean) => void;
  onSelectAll: () => void;
  onInvertSelection: () => void;
  onClearSelection: () => void;
  onReview: (candidate: IntentNodeCandidate, status: ReviewStatus) => void;
  onBatchReview: (status: ReviewStatus) => void;
}) {
  if (items.length === 0) {
    return <div className="py-10 text-center text-sm text-muted-foreground">暂无意图节点候选</div>;
  }

  const reviewableItems = items.filter((item) => !isReviewLocked(item.status));
  const allSelected =
    reviewableItems.length > 0 && reviewableItems.every((item) => selectedIds.has(item.id));
  const someSelected = !allSelected && reviewableItems.some((item) => selectedIds.has(item.id));
  const selectedCount = reviewableItems.filter((item) => selectedIds.has(item.id)).length;

  return (
    <div className="space-y-3">
      <CandidateBulkReviewToolbar
        label="意图候选"
        reviewableCount={reviewableItems.length}
        selectedCount={selectedCount}
        disabled={reviewDisabled}
        batchReviewingStatus={batchReviewingStatus}
        onSelectAll={onSelectAll}
        onInvertSelection={onInvertSelection}
        onClearSelection={onClearSelection}
        onBatchReview={onBatchReview}
      />
      <Table className="min-w-[1330px]">
        <TableHeader>
          <TableRow>
            <TableHead className="w-[48px]">
              <Checkbox
                checked={allSelected ? true : someSelected ? "indeterminate" : false}
                onCheckedChange={(checked) =>
                  checked === true ? onSelectAll() : onClearSelection()
                }
                aria-label="全选意图候选"
                disabled={reviewDisabled || reviewableItems.length === 0}
              />
            </TableHead>
            <TableHead className="w-[150px]">名称</TableHead>
            <TableHead className="w-[180px]">编码</TableHead>
            <TableHead className="w-[120px]">层级/类型</TableHead>
            <TableHead className="w-[160px]">父节点</TableHead>
            <TableHead className="w-[160px]">知识库</TableHead>
            <TableHead className="w-[90px]">置信度</TableHead>
            <TableHead className="w-[90px]">风险</TableHead>
            <TableHead className="w-[90px]">来源</TableHead>
            <TableHead className="w-[100px]">状态</TableHead>
            <TableHead>示例/证据</TableHead>
            <TableHead className="w-[170px]">操作</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((item) => {
            const locked = isReviewLocked(item.status);
            return (
              <TableRow key={item.id}>
                <TableCell>
                  <Checkbox
                    checked={selectedIds.has(item.id)}
                    onCheckedChange={(checked) => onToggleSelect(item.id, checked === true)}
                    aria-label={`选择意图候选 ${item.name}`}
                    disabled={reviewDisabled || locked}
                  />
                </TableCell>
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
                    <Badge variant="secondary">
                      {levelText[item.level ?? -1] || item.level || "-"}
                    </Badge>
                    <div className="text-muted-foreground">
                      {kindText[item.kind ?? -1] || item.kind || "-"}
                    </div>
                  </div>
                </TableCell>
                <TableCell className="max-w-[150px] truncate" title={item.parentCode || ""}>
                  {item.parentCode || "-"}
                </TableCell>
                <TableCell>
                  <div
                    className="max-w-[150px] truncate text-xs"
                    title={item.collectionName || item.kbId || ""}
                  >
                    {item.collectionName || item.kbId || "-"}
                  </div>
                  <div className="text-xs text-muted-foreground">topK {item.topK ?? "-"}</div>
                </TableCell>
                <TableCell className="tabular-nums">{formatScore(item.confidence)}</TableCell>
                <TableCell>
                  <Badge variant={riskVariant(item.riskLevel)}>{item.riskLevel || "-"}</Badge>
                </TableCell>
                <TableCell>
                  <GenerationSourceBadge source={item.generationSource} />
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
                      disabled={reviewDisabled || reviewingId === item.id || locked}
                      onClick={() => onReview(item, "APPROVED")}
                    >
                      <Check className="h-4 w-4" />
                      通过
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive hover:text-destructive"
                      disabled={reviewDisabled || reviewingId === item.id || locked}
                      onClick={() => onReview(item, "REJECTED")}
                    >
                      <X className="h-4 w-4" />
                      拒绝
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
