import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { Pencil, Plus, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type {
  AIModelSelection,
  AIProviders,
  ModelCandidate,
  SystemSettings
} from "@/services/settingsService";
import {
  getSystemSettings,
  updateAIModelSelection,
  updateAIProviders
} from "@/services/settingsService";
import { getErrorMessage } from "@/utils/error";

type ProviderConfig = AIProviders[string];
type ModelGroupConfig = SystemSettings["ai"]["chat"];

interface ProviderForm {
  name: string;
  url: string;
  apiKey: string;
  endpointsText: string;
}

const emptyProviderForm: ProviderForm = {
  name: "",
  url: "",
  apiKey: "",
  endpointsText: ""
};

interface CurrentModelUsage {
  label: string;
  modelId?: string | null;
  provider?: string;
  model?: string;
}

const emptyModelSelection: AIModelSelection = {
  chatDefaultModel: "",
  chatDeepThinkingModel: "",
  embeddingDefaultModel: "",
  rerankDefaultModel: ""
};

const BoolBadge = ({ value }: { value: boolean }) => (
  <Badge variant={value ? "default" : "outline"}>{value ? "启用" : "禁用"}</Badge>
);

function InfoItem({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-slate-200/70 bg-white px-4 py-3">
      <span className="text-xs text-slate-500">{label}</span>
      <div className="text-sm font-medium text-slate-800">{value}</div>
    </div>
  );
}

function endpointsToText(endpoints: Record<string, string> = {}) {
  return Object.entries(endpoints)
    .map(([key, value]) => `${key}=${value}`)
    .join("\n");
}

function parseEndpoints(text: string) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .reduce<Record<string, string>>((acc, line) => {
      const separatorIndex = line.indexOf("=");
      if (separatorIndex <= 0) {
        throw new Error("Endpoint 格式不正确");
      }
      const key = line.slice(0, separatorIndex).trim();
      const value = line.slice(separatorIndex + 1).trim();
      if (!key || !value) {
        throw new Error("Endpoint 名称和路径不能为空");
      }
      acc[key] = value;
      return acc;
    }, {});
}

function resolveModelUsage(
  label: string,
  group: ModelGroupConfig,
  modelId?: string | null
): CurrentModelUsage {
  const candidate = group?.candidates?.find(
    (item) => item.id === modelId || `${item.provider}::${item.model}` === modelId
  );
  return {
    label,
    modelId,
    provider: candidate?.provider,
    model: candidate?.model
  };
}

function resolveCandidateId(candidate: ModelCandidate) {
  return candidate.id || `${candidate.provider}::${candidate.model}`;
}

function buildCandidateLabel(candidate: ModelCandidate) {
  const id = resolveCandidateId(candidate);
  return `${id} · ${candidate.provider} · ${candidate.model}`;
}

function ModelSelectField({
  label,
  value,
  candidates,
  providers,
  onChange
}: {
  label: string;
  value: string;
  candidates: ModelCandidate[];
  providers: AIProviders;
  onChange: (value: string) => void;
}) {
  const selected = candidates.find((candidate) => resolveCandidateId(candidate) === value);
  const providerReady = !!selected?.provider && !!providers[selected.provider];

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-slate-200/70 bg-white px-4 py-3">
      <Label>{label}</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger>
          <SelectValue placeholder="请选择模型" />
        </SelectTrigger>
        <SelectContent>
          {candidates.map((candidate) => {
            const id = resolveCandidateId(candidate);
            return (
              <SelectItem key={id} value={id}>
                {buildCandidateLabel(candidate)}
              </SelectItem>
            );
          })}
        </SelectContent>
      </Select>
      <div className="flex min-h-6 flex-wrap items-center gap-2 text-xs text-muted-foreground">
        {selected ? (
          <>
            <Badge variant={providerReady ? "default" : "destructive"}>{selected.provider}</Badge>
            <span>{selected.model}</span>
          </>
        ) : (
          <span>当前模型未匹配候选列表</span>
        )}
      </div>
    </div>
  );
}

export function SystemSettingsPage() {
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [providerDialog, setProviderDialog] = useState<{
    open: boolean;
    mode: "create" | "edit";
    originalName: string | null;
  }>({
    open: false,
    mode: "create",
    originalName: null
  });
  const [providerForm, setProviderForm] = useState<ProviderForm>(emptyProviderForm);
  const [providerSaving, setProviderSaving] = useState(false);
  const [deleteProvider, setDeleteProvider] = useState<string | null>(null);
  const [modelSelection, setModelSelection] = useState<AIModelSelection>(emptyModelSelection);
  const [modelSelectionSaving, setModelSelectionSaving] = useState(false);

  const loadSettings = async () => {
    try {
      setLoading(true);
      const data = await getSystemSettings();
      setSettings(data);
    } catch (error) {
      toast.error(getErrorMessage(error, "加载系统配置失败"));
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSettings();
  }, []);

  useEffect(() => {
    if (!settings) {
      return;
    }
    setModelSelection({
      chatDefaultModel: settings.ai.chat.defaultModel || "",
      chatDeepThinkingModel: settings.ai.chat.deepThinkingModel || "",
      embeddingDefaultModel: settings.ai.embedding.defaultModel || "",
      rerankDefaultModel: settings.ai.rerank.defaultModel || ""
    });
  }, [settings]);

  if (loading) {
    return (
      <div className="admin-page">
        <div className="text-sm text-muted-foreground">加载中...</div>
      </div>
    );
  }

  if (!settings) {
    return (
      <div className="admin-page">
        <div className="text-sm text-muted-foreground">暂无可展示的配置</div>
      </div>
    );
  }

  const { rag, ai } = settings;
  const providers = Object.entries(ai.providers || {});
  const currentModelUsages = [
    resolveModelUsage("Chat 默认", ai.chat, ai.chat.defaultModel),
    resolveModelUsage("深度思考", ai.chat, ai.chat.deepThinkingModel),
    resolveModelUsage("Embedding", ai.embedding, ai.embedding.defaultModel),
    resolveModelUsage("Rerank", ai.rerank, ai.rerank.defaultModel)
  ].filter((item) => item.modelId);
  const chatCandidates = ai.chat.candidates || [];
  const deepThinkingCandidates = chatCandidates.filter((candidate) =>
    Boolean(candidate.supportsThinking)
  );
  const embeddingCandidates = ai.embedding.candidates || [];
  const rerankCandidates = ai.rerank.candidates || [];
  const modelSelectionChanged =
    modelSelection.chatDefaultModel !== (ai.chat.defaultModel || "") ||
    modelSelection.chatDeepThinkingModel !== (ai.chat.deepThinkingModel || "") ||
    modelSelection.embeddingDefaultModel !== (ai.embedding.defaultModel || "") ||
    modelSelection.rerankDefaultModel !== (ai.rerank.defaultModel || "");
  const providerUsageMap = currentModelUsages.reduce<Record<string, CurrentModelUsage[]>>(
    (acc, usage) => {
      if (usage.provider) {
        acc[usage.provider] = [...(acc[usage.provider] || []), usage];
      }
      return acc;
    },
    {}
  );

  const openCreateProviderDialog = () => {
    setProviderForm(emptyProviderForm);
    setProviderDialog({ open: true, mode: "create", originalName: null });
  };

  const openEditProviderDialog = (name: string, provider: ProviderConfig) => {
    setProviderForm({
      name,
      url: provider.url || "",
      apiKey: provider.apiKey || "",
      endpointsText: endpointsToText(provider.endpoints)
    });
    setProviderDialog({ open: true, mode: "edit", originalName: name });
  };

  const handleProviderSave = async () => {
    const name = providerForm.name.trim();
    const url = providerForm.url.trim();
    if (!name) {
      toast.error("Provider 名称不能为空");
      return;
    }
    if (!url) {
      toast.error("URL 不能为空");
      return;
    }

    let endpoints: Record<string, string>;
    try {
      endpoints = parseEndpoints(providerForm.endpointsText);
    } catch (error) {
      toast.error(getErrorMessage(error, "Endpoint 格式不正确"));
      return;
    }

    const nextProviders: AIProviders = { ...(settings.ai.providers || {}) };
    if (
      providerDialog.mode === "edit" &&
      providerDialog.originalName &&
      providerDialog.originalName !== name
    ) {
      delete nextProviders[providerDialog.originalName];
    }
    nextProviders[name] = {
      url,
      apiKey: providerForm.apiKey,
      endpoints
    };

    try {
      setProviderSaving(true);
      const updatedProviders = await updateAIProviders(nextProviders);
      setSettings((prev) =>
        prev
          ? {
              ...prev,
              ai: {
                ...prev.ai,
                providers: updatedProviders
              }
            }
          : prev
      );
      setProviderDialog({ open: false, mode: "create", originalName: null });
      toast.success("Provider 配置已保存");
    } catch (error) {
      toast.error(getErrorMessage(error, "保存 Provider 配置失败"));
      console.error(error);
    } finally {
      setProviderSaving(false);
    }
  };

  const handleProviderDelete = async () => {
    if (!deleteProvider) {
      return;
    }
    const nextProviders: AIProviders = { ...(settings.ai.providers || {}) };
    delete nextProviders[deleteProvider];

    try {
      const updatedProviders = await updateAIProviders(nextProviders);
      setSettings((prev) =>
        prev
          ? {
              ...prev,
              ai: {
                ...prev.ai,
                providers: updatedProviders
              }
            }
          : prev
      );
      toast.success("Provider 配置已删除");
    } catch (error) {
      toast.error(getErrorMessage(error, "删除 Provider 配置失败"));
      console.error(error);
    } finally {
      setDeleteProvider(null);
    }
  };

  const handleModelSelectionSave = async () => {
    if (
      !modelSelection.chatDefaultModel ||
      !modelSelection.chatDeepThinkingModel ||
      !modelSelection.embeddingDefaultModel ||
      !modelSelection.rerankDefaultModel
    ) {
      toast.error("请完整选择所有默认模型");
      return;
    }

    try {
      setModelSelectionSaving(true);
      const updatedAI = await updateAIModelSelection(modelSelection);
      setSettings((prev) =>
        prev
          ? {
              ...prev,
              ai: updatedAI
            }
          : prev
      );
      toast.success("模型配置已保存");
    } catch (error) {
      toast.error(getErrorMessage(error, "保存模型配置失败"));
      console.error(error);
    } finally {
      setModelSelectionSaving(false);
    }
  };

  return (
    <div className="admin-page">
      <div className="admin-page-header">
        <div>
          <h1 className="admin-page-title">系统配置</h1>
          <p className="admin-page-subtitle">
            查看当前 application 配置，并维护运行中的模型服务提供方
          </p>
        </div>
        <Button variant="outline" onClick={loadSettings} disabled={loading}>
          <RefreshCw className="mr-2 h-4 w-4" />
          刷新
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>RAG 默认配置</CardTitle>
          <CardDescription>向量空间与检索基础参数</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <InfoItem label="Collection" value={rag.default.collectionName} />
          <InfoItem label="Dimension" value={rag.default.dimension} />
          <InfoItem label="Metric Type" value={rag.default.metricType} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>查询改写</CardTitle>
          <CardDescription>历史上下文压缩与改写策略</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <InfoItem label="Enabled" value={<BoolBadge value={rag.queryRewrite.enabled} />} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>全局限流</CardTitle>
          <CardDescription>并发与租约控制</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <InfoItem label="Enabled" value={<BoolBadge value={rag.rateLimit.global.enabled} />} />
          <InfoItem label="Max Concurrent" value={rag.rateLimit.global.maxConcurrent} />
          <InfoItem label="Max Wait Seconds" value={rag.rateLimit.global.maxWaitSeconds} />
          <InfoItem label="Lease Seconds" value={rag.rateLimit.global.leaseSeconds} />
          <InfoItem label="Poll Interval (ms)" value={rag.rateLimit.global.pollIntervalMs} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>记忆管理</CardTitle>
          <CardDescription>摘要与上下文保留策略</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <InfoItem label="History Keep Turns" value={rag.memory.historyKeepTurns} />
          <InfoItem label="Summary Start Turns" value={rag.memory.summaryStartTurns} />
          <InfoItem
            label="Summary Enabled"
            value={<BoolBadge value={rag.memory.summaryEnabled} />}
          />
          <InfoItem label="Summary Max Chars" value={rag.memory.summaryMaxChars} />
          <InfoItem label="Title Max Length" value={rag.memory.titleMaxLength} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle>当前模型路由</CardTitle>
            <CardDescription>选择默认模型并查看实际对应的服务提供方</CardDescription>
          </div>
          <Button
            onClick={handleModelSelectionSave}
            disabled={modelSelectionSaving || !modelSelectionChanged}
          >
            保存模型配置
          </Button>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <ModelSelectField
            label="Chat 默认模型"
            value={modelSelection.chatDefaultModel}
            candidates={chatCandidates}
            providers={ai.providers || {}}
            onChange={(value) =>
              setModelSelection((prev) => ({ ...prev, chatDefaultModel: value }))
            }
          />
          <ModelSelectField
            label="深度思考模型"
            value={modelSelection.chatDeepThinkingModel}
            candidates={deepThinkingCandidates}
            providers={ai.providers || {}}
            onChange={(value) =>
              setModelSelection((prev) => ({ ...prev, chatDeepThinkingModel: value }))
            }
          />
          <ModelSelectField
            label="Embedding 模型"
            value={modelSelection.embeddingDefaultModel}
            candidates={embeddingCandidates}
            providers={ai.providers || {}}
            onChange={(value) =>
              setModelSelection((prev) => ({ ...prev, embeddingDefaultModel: value }))
            }
          />
          <ModelSelectField
            label="Rerank 模型"
            value={modelSelection.rerankDefaultModel}
            candidates={rerankCandidates}
            providers={ai.providers || {}}
            onChange={(value) =>
              setModelSelection((prev) => ({ ...prev, rerankDefaultModel: value }))
            }
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle>模型服务提供方</CardTitle>
            <CardDescription>接入地址与端点配置</CardDescription>
          </div>
          <Button className="admin-primary-gradient" onClick={openCreateProviderDialog}>
            <Plus className="mr-2 h-4 w-4" />
            新增 Provider
          </Button>
        </CardHeader>
        <CardContent>
          <Table className="min-w-[760px]">
            <TableHeader>
              <TableRow>
                <TableHead className="w-[140px]">Provider</TableHead>
                <TableHead className="w-[240px]">URL</TableHead>
                <TableHead className="w-[200px]">API Key</TableHead>
                <TableHead>Endpoints</TableHead>
                <TableHead className="w-[150px] text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {providers.map(([name, provider]) => (
                <TableRow key={name}>
                  <TableCell>
                    <div className="flex flex-col gap-2">
                      <span className="font-medium">{name}</span>
                      {providerUsageMap[name]?.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {providerUsageMap[name].map((usage) => (
                            <Badge key={usage.label} variant="secondary">
                              当前 {usage.label}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>{provider.url}</TableCell>
                  <TableCell>{provider.apiKey ? provider.apiKey : "-"}</TableCell>
                  <TableCell>
                    <div className="space-y-1 text-xs text-muted-foreground">
                      {Object.entries(provider.endpoints).map(([key, value]) => (
                        <div key={key}>
                          {key}: {value}
                        </div>
                      ))}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex justify-end gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        aria-label={`编辑 ${name}`}
                        title="编辑"
                        onClick={() => openEditProviderDialog(name, provider)}
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        aria-label={`删除 ${name}`}
                        title="删除"
                        onClick={() => setDeleteProvider(name)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
              {providers.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="py-8 text-center text-sm text-muted-foreground">
                    暂无 Provider 配置
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>模型选择策略</CardTitle>
          <CardDescription>熔断与选择阈值</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <InfoItem label="Failure Threshold" value={ai.selection.failureThreshold} />
          <InfoItem label="Open Duration (ms)" value={ai.selection.openDurationMs} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>流式响应</CardTitle>
          <CardDescription>输出分片大小</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <InfoItem label="Message Chunk Size" value={ai.stream.messageChunkSize} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Chat 模型配置</CardTitle>
          <CardDescription>默认模型与候选列表</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <InfoItem label="Default Model" value={ai.chat.defaultModel} />
            <InfoItem label="Deep Thinking Model" value={ai.chat.deepThinkingModel} />
          </div>
          <Table className="min-w-[720px]">
            <TableHeader>
              <TableRow>
                <TableHead className="w-[220px]">ID</TableHead>
                <TableHead className="w-[120px]">Provider</TableHead>
                <TableHead className="w-[200px]">Model</TableHead>
                <TableHead className="w-[100px]">Thinking</TableHead>
                <TableHead className="w-[90px]">Priority</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ai.chat.candidates.map((item) => (
                <TableRow key={item.id}>
                  <TableCell className="font-medium">{item.id}</TableCell>
                  <TableCell>{item.provider}</TableCell>
                  <TableCell>{item.model}</TableCell>
                  <TableCell>{item.supportsThinking ? "支持" : "-"}</TableCell>
                  <TableCell>{item.priority}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Embedding 模型配置</CardTitle>
          <CardDescription>向量化模型列表</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <InfoItem label="Default Model" value={ai.embedding.defaultModel} />
          </div>
          <Table className="min-w-[720px]">
            <TableHeader>
              <TableRow>
                <TableHead className="w-[220px]">ID</TableHead>
                <TableHead className="w-[120px]">Provider</TableHead>
                <TableHead className="w-[200px]">Model</TableHead>
                <TableHead className="w-[110px]">Dimension</TableHead>
                <TableHead className="w-[90px]">Priority</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ai.embedding.candidates.map((item) => (
                <TableRow key={item.id}>
                  <TableCell className="font-medium">{item.id}</TableCell>
                  <TableCell>{item.provider}</TableCell>
                  <TableCell>{item.model}</TableCell>
                  <TableCell>{item.dimension}</TableCell>
                  <TableCell>{item.priority}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Rerank 模型配置</CardTitle>
          <CardDescription>重排模型列表</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <InfoItem label="Default Model" value={ai.rerank.defaultModel} />
          </div>
          <Table className="min-w-[640px]">
            <TableHeader>
              <TableRow>
                <TableHead className="w-[220px]">ID</TableHead>
                <TableHead className="w-[120px]">Provider</TableHead>
                <TableHead className="w-[200px]">Model</TableHead>
                <TableHead className="w-[90px]">Priority</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ai.rerank.candidates.map((item) => (
                <TableRow key={item.id}>
                  <TableCell className="font-medium">{item.id}</TableCell>
                  <TableCell>{item.provider}</TableCell>
                  <TableCell>{item.model}</TableCell>
                  <TableCell>{item.priority}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog
        open={providerDialog.open}
        onOpenChange={(open) =>
          setProviderDialog((prev) => ({
            ...prev,
            open
          }))
        }
      >
        <DialogContent className="sm:max-w-[620px]">
          <DialogHeader>
            <DialogTitle>
              {providerDialog.mode === "create" ? "新增 Provider" : "编辑 Provider"}
            </DialogTitle>
            <DialogDescription>修改后的配置会立即用于当前服务进程的模型路由</DialogDescription>
          </DialogHeader>

          <div className="grid gap-4">
            <div className="grid gap-2">
              <Label htmlFor="provider-name">Provider</Label>
              <Input
                id="provider-name"
                value={providerForm.name}
                onChange={(event) =>
                  setProviderForm((prev) => ({ ...prev, name: event.target.value }))
                }
                placeholder="siliconflow"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="provider-url">URL</Label>
              <Input
                id="provider-url"
                value={providerForm.url}
                onChange={(event) =>
                  setProviderForm((prev) => ({ ...prev, url: event.target.value }))
                }
                placeholder="https://api.example.com"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="provider-api-key">API Key</Label>
              <Input
                id="provider-api-key"
                type="password"
                value={providerForm.apiKey}
                onChange={(event) =>
                  setProviderForm((prev) => ({ ...prev, apiKey: event.target.value }))
                }
                placeholder="sk-..."
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="provider-endpoints">Endpoints</Label>
              <Textarea
                id="provider-endpoints"
                className="min-h-[130px] font-mono"
                value={providerForm.endpointsText}
                onChange={(event) =>
                  setProviderForm((prev) => ({ ...prev, endpointsText: event.target.value }))
                }
                placeholder={"chat=/v1/chat/completions\nembedding=/v1/embeddings"}
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setProviderDialog({ open: false, mode: "create", originalName: null })}
              disabled={providerSaving}
            >
              取消
            </Button>
            <Button onClick={handleProviderSave} disabled={providerSaving}>
              保存
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={!!deleteProvider} onOpenChange={() => setDeleteProvider(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除</AlertDialogTitle>
            <AlertDialogDescription>
              删除 Provider「{deleteProvider}」后，引用它的候选模型会在路由时被跳过。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleProviderDelete}
              className="bg-destructive text-destructive-foreground"
            >
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
