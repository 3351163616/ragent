import * as React from "react";
import { Activity, Brain, Layers, MessageSquare, Sparkles, Wrench } from "lucide-react";

import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores/chatStore";
import type { Message } from "@/types";

interface AnthropicInspectorProps {
  currentMessage: Message | null;
  isStreaming: boolean;
}

interface TimelineStep {
  name: string;
  ms: string;
  body: string;
  chips: { label: string; tone?: "default" | "coral" | "teal" }[];
  state: "done" | "active" | "pending";
}

/**
 * 右侧 Inspector — 思考过程时间线。
 * 数据派生自 currentMessage：
 *   - thinking 字段 → 深度思考阶段
 *   - citations 数量 → 检索通道、命中数
 *   - status === "streaming" → 当前 active step
 */
export function AnthropicInspector({ currentMessage, isStreaming }: AnthropicInspectorProps) {
  const deepThinkingEnabled = useChatStore((s) => s.deepThinkingEnabled);
  const selectedModelId = useChatStore((s) => s.selectedModelId);
  const messages = useChatStore((s) => s.messages);

  const citationCount = currentMessage?.citations?.length ?? 0;
  const hasThinking = Boolean(currentMessage?.thinking?.trim());
  const thinkingSeconds = currentMessage?.thinkingDuration ?? null;
  const turns = messages.filter((m) => m.role === "user").length;

  const steps: TimelineStep[] = React.useMemo(() => {
    const done: TimelineStep["state"] = "done";
    const active: TimelineStep["state"] = isStreaming ? "active" : "done";
    const pending: TimelineStep["state"] = "pending";

    return [
      {
        name: "术语归一化",
        ms: "12 ms",
        body: "QueryTermMappingService 把口语化表达映射到 KB 标准术语。",
        chips: [
          { label: "cache hit", tone: "teal" },
          { label: "QueryTermMapping", tone: "default" }
        ],
        state: done
      },
      {
        name: "Multi-Question Rewrite",
        ms: "142 ms",
        body: "基于最近 2 轮历史做指代消解，按需拆分多个子问题。",
        chips: [
          { label: "user-question-rewrite.st", tone: "default" },
          { label: `${Math.max(1, citationCount > 4 ? 3 : 2)} sub-Qs`, tone: "default" }
        ],
        state: done
      },
      {
        name: "意图识别",
        ms: "208 ms",
        body: "DefaultIntentClassifier 并行匹配 KB / SYSTEM / MCP 三类意图。",
        chips: [
          { label: "intent-classifier.st", tone: "default" },
          { label: deepThinkingEnabled ? "deep" : "fast", tone: "coral" }
        ],
        state: done
      },
      {
        name: "多通道检索",
        ms: "381 ms",
        body: "vector-global × intent-directed 并行；rerank top-k 后归并。",
        chips: [
          { label: `${Math.max(citationCount, 1) * 4} candidates`, tone: "default" },
          { label: `rerank top-${citationCount || 4}`, tone: "default" }
        ],
        state: active
      },
      {
        name: "流式生成",
        ms: isStreaming ? "streaming…" : "609 ms",
        body: "answer-chat-mcp-kb-mixed.st · prompt scene 自动路由。",
        chips: [
          { label: `model: ${selectedModelId || "sonnet-4-6"}`, tone: "default" },
          { label: "routing healthy", tone: "teal" }
        ],
        state: isStreaming ? active : done
      },
      {
        name: turns >= 5 ? "摘要压缩" : "下一轮：摘要压缩",
        ms: turns >= 5 ? "queued" : "pending",
        body: "达到 6 轮后触发 conversation-summary.st 压缩历史。",
        chips: [{ label: `${turns}/6 turns`, tone: "default" }],
        state: pending
      }
    ];
  }, [citationCount, deepThinkingEnabled, isStreaming, selectedModelId, turns]);

  return (
    <div className="flex flex-col gap-5 p-5">
      <div className="flex items-center justify-between">
        <h3 className="anthropic-display text-[19px] leading-none text-anthropic-ink">
          Thinking trace
        </h3>
        <span className="anthropic-mono text-[11px] text-anthropic-muted">
          {hasThinking && thinkingSeconds ? `${thinkingSeconds}s thinking` : "live"}
        </span>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 rounded-anthropic-md border border-anthropic-hairline-soft bg-anthropic-canvas p-1">
        {[
          { label: "Pipeline", active: true },
          { label: "Routing" },
          { label: "Memory" }
        ].map((tab) => (
          <button
            key={tab.label}
            type="button"
            className={cn(
              "flex-1 rounded-anthropic-sm px-2 py-1.5 text-[12px] font-medium transition-colors",
              tab.active
                ? "bg-anthropic-surface-card text-anthropic-ink"
                : "text-anthropic-muted hover:text-anthropic-ink"
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Timeline */}
      <div className="relative pl-7">
        <span className="absolute left-[10px] top-2 bottom-2 w-px bg-anthropic-hairline" />
        {steps.map((step) => (
          <div key={step.name} className="relative py-3">
            <span
              className={cn(
                "absolute -left-[18px] top-[18px] h-[10px] w-[10px] rounded-full border-2",
                step.state === "done" &&
                  "border-anthropic-coral bg-anthropic-coral",
                step.state === "active" &&
                  "border-anthropic-coral bg-anthropic-canvas animate-pulse",
                step.state === "pending" && "border-anthropic-hairline bg-anthropic-canvas"
              )}
            />
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-[13px] font-medium text-anthropic-ink">{step.name}</span>
              <span className="anthropic-mono text-[11px] text-anthropic-muted">{step.ms}</span>
            </div>
            <p className="mt-1 text-[12.5px] leading-[1.5] text-anthropic-muted">{step.body}</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {step.chips.map((chip) => (
                <span
                  key={chip.label}
                  className={cn(
                    "anthropic-mono rounded-anthropic-xs border px-1.5 py-[1px] text-[10.5px]",
                    chip.tone === "coral" &&
                      "border-anthropic-coral bg-anthropic-coral text-white",
                    chip.tone === "teal" &&
                      "border-[#5db8a6]/40 bg-[#5db8a6]/15 text-[#2d8772]",
                    (!chip.tone || chip.tone === "default") &&
                      "border-anthropic-hairline bg-anthropic-canvas text-anthropic-ink"
                  )}
                >
                  {chip.label}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Model routing card */}
      <div className="flex flex-col gap-3 rounded-anthropic-lg border border-anthropic-hairline-soft bg-anthropic-canvas p-4">
        <div className="flex items-baseline justify-between">
          <span className="anthropic-mono text-[11px] uppercase tracking-[1.4px] text-anthropic-muted">
            Model routing
          </span>
          <span className="text-[11px] font-medium text-anthropic-coral">view all</span>
        </div>
        {[
          { name: selectedModelId || "claude-sonnet-4-6", latency: "162 ms", state: "closed" },
          { name: "bce-rerank-v2", latency: "94 ms", state: "closed" },
          { name: "qwen3-embedding-1.5b", latency: "probe…", state: "half" },
          { name: "moyu-fallback-gpt4", latency: "— ms", state: "open" }
        ].map((row, idx) => (
          <div
            key={row.name}
            className={cn(
              "flex items-center gap-2.5 py-1.5 text-[13px]",
              idx !== 0 && "border-t border-anthropic-hairline-soft pt-2.5"
            )}
          >
            <span
              className={cn(
                "h-2 w-2 flex-none rounded-full",
                row.state === "closed" && "bg-anthropic-success",
                row.state === "half" && "bg-anthropic-amber",
                row.state === "open" && "bg-anthropic-error"
              )}
            />
            <span className="flex-1 font-medium text-anthropic-ink">{row.name}</span>
            <span className="anthropic-mono text-[11.5px] text-anthropic-muted">
              {row.latency}
            </span>
            <span
              className={cn(
                "anthropic-mono rounded-anthropic-xs px-1.5 py-[1px] text-[10.5px] uppercase tracking-[1.2px]",
                row.state === "closed" && "bg-[#5db872]/16 text-[#2d7a45]",
                row.state === "half" && "bg-[#d4a017]/16 text-[#946a08]",
                row.state === "open" && "bg-[#c64545]/16 text-[#8d2929]"
              )}
            >
              {row.state}
            </span>
          </div>
        ))}
      </div>

      {/* Memory window card */}
      <div className="flex flex-col gap-3 rounded-anthropic-lg border border-anthropic-hairline-soft bg-anthropic-canvas p-4">
        <div className="flex items-baseline justify-between">
          <span className="anthropic-mono text-[11px] uppercase tracking-[1.4px] text-anthropic-muted">
            Memory window
          </span>
          <span className="anthropic-mono text-[11px] text-anthropic-muted">{turns}/6 turns</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-anthropic-hairline">
            <div
              className="h-full rounded-full bg-anthropic-coral transition-all"
              style={{ width: `${Math.min(100, (turns / 6) * 100)}%` }}
            />
          </div>
          <span className="anthropic-mono text-[11px] text-anthropic-muted">
            {Math.round(Math.min(100, (turns / 6) * 100))}%
          </span>
        </div>
        <p className="text-[12px] leading-[1.5] text-anthropic-muted">
          再过 {Math.max(0, 6 - turns)} 轮将触发 LLM 摘要压缩，旧消息会被替换为
          <code className="ml-1 anthropic-mono text-[11px] text-anthropic-ink">[summary]</code>。
        </p>
      </div>

      {/* Mini stats */}
      <div className="grid grid-cols-2 gap-3">
        {[
          { icon: MessageSquare, label: "turns", value: turns },
          { icon: Layers, label: "citations", value: citationCount },
          {
            icon: Brain,
            label: "thinking",
            value: hasThinking && thinkingSeconds ? `${thinkingSeconds}s` : "—"
          },
          { icon: Wrench, label: "mcp", value: 0 }
        ].map((s) => {
          const Icon = s.icon;
          return (
            <div
              key={s.label}
              className="flex items-center gap-2.5 rounded-anthropic-md border border-anthropic-hairline-soft bg-anthropic-canvas px-3 py-2.5"
            >
              <Icon className="h-3.5 w-3.5 text-anthropic-muted" />
              <div className="flex flex-1 items-baseline justify-between">
                <span className="anthropic-mono text-[10.5px] uppercase tracking-[1.2px] text-anthropic-muted">
                  {s.label}
                </span>
                <span className="anthropic-mono text-[14px] font-medium text-anthropic-ink">
                  {s.value}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="rounded-anthropic-lg bg-anthropic-surface-dark p-4 text-anthropic-on-dark">
        <div className="flex items-center gap-2">
          <Activity className="h-3.5 w-3.5 text-anthropic-coral" />
          <span className="anthropic-mono text-[11px] uppercase tracking-[1.4px] text-anthropic-on-dark-soft">
            Live signal
          </span>
        </div>
        <p className="mt-2 text-[12.5px] leading-[1.55] text-anthropic-on-dark-soft">
          {isStreaming ? (
            <>
              <Sparkles className="mr-1 inline h-3 w-3 text-anthropic-coral" />
              SSE 流式生成中，token 持续注入...
            </>
          ) : (
            "Pipeline 处于空闲。下一次提问会重置 trace。"
          )}
        </p>
      </div>

      <div className="anthropic-switcher-spacer" />
    </div>
  );
}