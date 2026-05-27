import * as React from "react";
import { ExternalLink, FileText } from "lucide-react";

import type { Citation } from "@/types";

interface AnthropicCitationsProps {
  citations?: Citation[];
}

/**
 * 深海军蓝来源卡片网格 — 替代 legacy 的浅色 CitationList。
 *   - 每张卡片有 score bar、通道角标（推断自 chunkId / sourceUrl）
 *   - hover 浮起 + 珊瑚色阴影
 *   - 内嵌引用号 [n] 与正文 .cite chip 配对
 */
export function AnthropicCitations({ citations }: AnthropicCitationsProps) {
  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-4">
      <div className="mb-3 flex items-baseline justify-between">
        <h4 className="anthropic-display text-[18px] leading-[1.25] text-anthropic-ink">
          <FileText className="-mt-0.5 mr-2 inline h-4 w-4 text-anthropic-coral" />
          {citations.length} 条引用证据
        </h4>
        <span className="anthropic-mono text-[11.5px] text-anthropic-muted">
          rerank · top-{citations.length}
        </span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {citations.map((c) => {
          const title = c.docName || c.chunkId || "未知来源";
          const chunkLabel =
            c.chunkIndex !== null && c.chunkIndex !== undefined ? `chunk-${c.chunkIndex}` : null;
          const channel = c.sourceUrl ? "mcp-tool" : c.kbName ? "vector-global" : "intent-directed";
          const score = c.score ?? 0;
          const scorePct = Math.max(0, Math.min(100, score * 100));

          return (
            <article
              key={`${c.index}-${c.chunkId || title}`}
              className="group/citation relative flex flex-col gap-3 overflow-hidden rounded-anthropic-lg bg-anthropic-surface-dark p-4 text-anthropic-on-dark transition-all duration-200 hover:-translate-y-0.5 hover:shadow-anthropic-dark-lift"
            >
              <div
                className="pointer-events-none absolute inset-0 opacity-70"
                style={{
                  background:
                    "radial-gradient(120% 80% at 100% 0%, rgba(204,120,92,0.08) 0%, transparent 60%)"
                }}
              />
              <div className="relative flex items-center justify-between gap-2 anthropic-mono text-[11px]">
                <span className="flex items-center gap-2 text-anthropic-on-dark-soft">
                  <span className="rounded-anthropic-xs bg-anthropic-surface-dark-elevated px-1.5 py-[1px] text-anthropic-on-dark">
                    [{c.index}]
                  </span>
                  <span className="truncate">{c.kbName || "kb"}</span>
                </span>
                <span className="text-[#5db8a6]">{channel}</span>
              </div>

              <div className="relative text-[14px] font-medium leading-[1.4] text-anthropic-on-dark">
                {title}
              </div>

              {c.snippet ? (
                <div
                  className="relative line-clamp-3 border-l-2 border-anthropic-surface-dark-elevated pl-3 text-[12.5px] leading-[1.55] text-anthropic-on-dark-soft"
                  style={{ wordBreak: "break-word" }}
                >
                  {c.snippet}
                </div>
              ) : null}

              <div className="relative mt-auto flex items-center justify-between border-t border-anthropic-hairline-dark pt-2.5">
                <div className="flex items-center gap-2 anthropic-mono text-[11px] text-anthropic-on-dark-soft">
                  <span>score {score.toFixed(2)}</span>
                  <span className="h-1 w-16 overflow-hidden rounded-full bg-anthropic-surface-dark-elevated">
                    <span
                      className="block h-full rounded-full bg-[#5db8a6]"
                      style={{ width: `${scorePct}%` }}
                    />
                  </span>
                </div>
                <span className="flex items-center gap-2 anthropic-mono text-[10.5px] text-anthropic-on-dark-soft">
                  {chunkLabel ? <span>{chunkLabel}</span> : null}
                  {c.sourceUrl ? (
                    <a
                      href={c.sourceUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-1 transition-colors hover:text-anthropic-coral"
                    >
                      OPEN
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  ) : (
                    <span>chunk</span>
                  )}
                </span>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}