import * as React from "react";
import { Brain, ChevronDown, Copy, ExternalLink, FileText, PencilLine } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { FeedbackButtons } from "@/components/chat/FeedbackButtons";
import { MarkdownRenderer } from "@/components/chat/MarkdownRenderer";
import { ThinkingIndicator } from "@/components/chat/ThinkingIndicator";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores/chatStore";
import type { Citation, Message } from "@/types";

interface MessageItemProps {
  message: Message;
  isLast?: boolean;
}

function CitationList({ citations }: { citations?: Citation[] }) {
  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 space-y-2">
      <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
        <FileText className="h-3.5 w-3.5" />
        引用来源
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {citations.map((citation) => {
          const title = citation.docName || citation.chunkId || "未知来源";
          const chunkLabel =
            citation.chunkIndex !== null && citation.chunkIndex !== undefined
              ? `chunk-${citation.chunkIndex}`
              : null;
          const source = citation.sourceUrl || undefined;
          return (
            <div
              key={`${citation.index}-${citation.chunkId || title}`}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 rounded bg-slate-100 px-1.5 py-0.5 font-medium text-slate-700">
                  [{citation.index}]
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <span className="truncate font-medium text-slate-800">{title}</span>
                    {source ? (
                      <a
                        href={source}
                        target="_blank"
                        rel="noreferrer"
                        className="shrink-0 text-slate-400 hover:text-[#2563EB]"
                        title="打开来源"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                      </a>
                    ) : null}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1.5 text-[11px] text-slate-500">
                    {citation.kbName ? <span>{citation.kbName}</span> : null}
                    {chunkLabel ? <span>{chunkLabel}</span> : null}
                    {citation.score !== null && citation.score !== undefined ? (
                      <span>{citation.score.toFixed(2)}</span>
                    ) : null}
                  </div>
                  {citation.snippet ? (
                    <p className="mt-1 line-clamp-2 break-words text-[11px] leading-relaxed text-slate-500">
                      {citation.snippet}
                    </p>
                  ) : null}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export const MessageItem = React.memo(function MessageItem({ message, isLast }: MessageItemProps) {
  const isUser = message.role === "user";
  const beginEditingMessage = useChatStore((state) => state.beginEditingMessage);
  const isLoading = useChatStore((state) => state.isLoading);
  const isStreaming = useChatStore((state) => state.isStreaming);
  const showFeedback =
    message.role === "assistant" &&
    message.status !== "streaming" &&
    message.id &&
    !message.id.startsWith("assistant-");
  const isThinking = Boolean(message.isThinking);
  const [thinkingExpanded, setThinkingExpanded] = React.useState(false);
  const hasThinking = Boolean(message.thinking && message.thinking.trim().length > 0);
  const hasContent = message.content.trim().length > 0;
  const isWaiting = message.status === "streaming" && !isThinking && !hasContent;
  const disableUserActions = isLoading || isStreaming;

  const handleCopyUserMessage = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      toast.success("复制成功");
    } catch {
      toast.error("复制失败");
    }
  };

  if (isUser) {
    return (
      <div className="group/user flex flex-col items-end gap-1">
        <div className="user-message">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
        <div className="mr-1 flex items-center gap-1 opacity-0 transition-opacity group-hover/user:opacity-100 focus-within:opacity-100">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={handleCopyUserMessage}
            title="复制"
            aria-label="复制用户消息"
            className="h-7 w-7 text-[#999999] hover:bg-[#F5F5F5] hover:text-[#666666] focus-visible:ring-[#BFDBFE]"
          >
            <Copy className="h-3.5 w-3.5" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => beginEditingMessage(message.id, message.content)}
            disabled={disableUserActions}
            title="编辑"
            aria-label="编辑历史消息"
            className="h-7 w-7 text-[#999999] hover:bg-[#F5F5F5] hover:text-[#2563EB] focus-visible:ring-[#BFDBFE]"
          >
            <PencilLine className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
    );
  }

  const thinkingDuration = message.thinkingDuration ? `${message.thinkingDuration}秒` : "";
  return (
    <div className="group flex">
      <div className="min-w-0 flex-1 space-y-4">
        {isThinking ? (
          <ThinkingIndicator content={message.thinking} duration={message.thinkingDuration} />
        ) : null}
        {!isThinking && hasThinking ? (
          <div className="overflow-hidden rounded-lg border border-[#BFDBFE] bg-[#DBEAFE]">
            <button
              type="button"
              onClick={() => setThinkingExpanded((prev) => !prev)}
              className="flex w-full items-center gap-2 px-4 py-3 text-left transition-colors hover:bg-[#BFDBFE]/30"
            >
              <div className="flex flex-1 items-center gap-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#BFDBFE]">
                  <Brain className="h-4 w-4 text-[#2563EB]" />
                </div>
                <span className="text-sm font-medium text-[#2563EB]">深度思考</span>
                {thinkingDuration ? (
                  <span className="rounded-full bg-[#BFDBFE] px-2 py-0.5 text-xs text-[#2563EB]">
                    {thinkingDuration}
                  </span>
                ) : null}
              </div>
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-[#3B82F6] transition-transform",
                  thinkingExpanded && "rotate-180"
                )}
              />
            </button>
            {thinkingExpanded ? (
              <div className="border-t border-[#BFDBFE] px-4 pb-4">
                <div className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-[#1E40AF]">
                  {message.thinking}
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
        <div className="space-y-2">
          {isWaiting ? (
            <div className="ai-wait" aria-label="思考中">
              <span className="ai-wait-dots" aria-hidden="true">
                <span className="ai-wait-dot" />
                <span className="ai-wait-dot" />
                <span className="ai-wait-dot" />
              </span>
            </div>
          ) : null}
          {hasContent ? <MarkdownRenderer content={message.content} /> : null}
          <CitationList citations={message.citations} />
          {message.status === "error" ? (
            <p className="text-xs text-rose-500">生成已中断。</p>
          ) : null}
          {showFeedback ? (
            <FeedbackButtons
              messageId={message.id}
              feedback={message.feedback ?? null}
              content={message.content}
              alwaysVisible={Boolean(isLast)}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
});
