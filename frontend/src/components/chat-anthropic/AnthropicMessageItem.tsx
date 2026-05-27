import * as React from "react";
import { Brain, ChevronDown, Copy, PencilLine } from "lucide-react";
import { toast } from "sonner";

import { AnthropicCitations } from "@/components/chat-anthropic/AnthropicCitations";
import { AnthropicMarkdown } from "@/components/chat-anthropic/AnthropicMarkdown";
import { Spike } from "@/components/chat-anthropic/Spike";
import { FeedbackButtons } from "@/components/chat/FeedbackButtons";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores/chatStore";
import type { Message } from "@/types";

interface AnthropicMessageItemProps {
  message: Message;
  isLast?: boolean;
}

export const AnthropicMessageItem = React.memo(function AnthropicMessageItem({
  message,
  isLast
}: AnthropicMessageItemProps) {
  const isUser = message.role === "user";
  const beginEditingMessage = useChatStore((s) => s.beginEditingMessage);
  const isLoading = useChatStore((s) => s.isLoading);
  const isStreaming = useChatStore((s) => s.isStreaming);

  const isThinking = Boolean(message.isThinking);
  const hasThinking = Boolean(message.thinking && message.thinking.trim().length > 0);
  const hasContent = message.content.trim().length > 0;
  const isWaiting = message.status === "streaming" && !isThinking && !hasContent;
  const isStreamingThis = message.status === "streaming";

  const showFeedback =
    message.role === "assistant" &&
    message.status !== "streaming" &&
    message.id &&
    !message.id.startsWith("assistant-");

  const disableUserActions = isLoading || isStreaming;
  const [thinkingExpanded, setThinkingExpanded] = React.useState(false);

  const handleCopyUserMessage = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      toast.success("已复制");
    } catch {
      toast.error("复制失败");
    }
  };

  if (isUser) {
    return (
      <div className="group/user flex flex-col items-end gap-1.5">
        <div className="anthropic-user-bubble">{message.content}</div>
        <div className="mr-0.5 flex items-center gap-0.5 opacity-0 transition-opacity group-hover/user:opacity-100 focus-within:opacity-100">
          <button
            type="button"
            onClick={handleCopyUserMessage}
            className="flex h-7 w-7 items-center justify-center rounded-anthropic-sm text-anthropic-muted transition-colors hover:bg-anthropic-surface-soft hover:text-anthropic-ink"
            aria-label="复制消息"
            title="复制"
          >
            <Copy className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={() => beginEditingMessage(message.id, message.content)}
            disabled={disableUserActions}
            className="flex h-7 w-7 items-center justify-center rounded-anthropic-sm text-anthropic-muted transition-colors hover:bg-anthropic-surface-soft hover:text-anthropic-coral disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="编辑历史消息"
            title="编辑"
          >
            <PencilLine className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="group flex gap-3.5">
      <div className="mt-1 flex h-8 w-8 flex-none items-center justify-center rounded-full border border-anthropic-hairline bg-anthropic-canvas text-anthropic-ink">
        <Spike size={14} />
      </div>
      <div className="min-w-0 flex-1 space-y-4">
        {/* Assistant meta line */}
        <div className="flex items-center gap-2 anthropic-mono text-[11px] uppercase tracking-[1.2px] text-anthropic-muted">
          <span
            className="h-1.5 w-1.5 rounded-full bg-anthropic-coral"
            style={{
              boxShadow: isStreamingThis
                ? "0 0 0 3px rgba(204,120,92,0.22)"
                : "0 0 0 3px rgba(204,120,92,0.12)"
            }}
          />
          Ragent · pipeline {isStreamingThis ? "streaming" : "done"}
          {message.thinkingDuration ? (
            <span className="anthropic-mono text-[11px] text-anthropic-muted">
              · {message.thinkingDuration}s thinking
            </span>
          ) : null}
        </div>

        {/* Thinking active */}
        {isThinking ? (
          <div className="rounded-anthropic-lg border border-anthropic-hairline bg-anthropic-surface-soft p-4">
            <div className="flex items-center gap-2 text-[13px] font-medium text-anthropic-coral">
              <Brain className="h-3.5 w-3.5" />
              正在深度思考...
              {message.thinkingDuration ? (
                <span className="anthropic-mono rounded-anthropic-xs bg-anthropic-canvas px-1.5 py-[1px] text-[11px] text-anthropic-coral">
                  {message.thinkingDuration}s
                </span>
              ) : null}
            </div>
            <p className="mt-3 whitespace-pre-wrap text-[13.5px] leading-[1.6] text-anthropic-body-strong">
              {message.thinking || ""}
              <span className="anthropic-stream-cursor" />
            </p>
          </div>
        ) : null}

        {/* Thinking collapsed (after done) */}
        {!isThinking && hasThinking ? (
          <div className="overflow-hidden rounded-anthropic-lg border border-anthropic-hairline bg-anthropic-canvas">
            <button
              type="button"
              onClick={() => setThinkingExpanded((p) => !p)}
              className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-anthropic-surface-soft"
            >
              <span className="flex h-7 w-7 items-center justify-center rounded-anthropic-md bg-anthropic-surface-card text-anthropic-coral">
                <Brain className="h-3.5 w-3.5" />
              </span>
              <span className="flex-1">
                <span className="text-[13px] font-medium text-anthropic-ink">深度思考过程</span>
                {message.thinkingDuration ? (
                  <span className="ml-2 anthropic-mono rounded-anthropic-xs bg-anthropic-surface-card px-1.5 py-[1px] text-[11px] text-anthropic-coral">
                    {message.thinkingDuration}s
                  </span>
                ) : null}
              </span>
              <ChevronDown
                className={cn(
                  "h-4 w-4 text-anthropic-muted transition-transform",
                  thinkingExpanded && "rotate-180"
                )}
              />
            </button>
            {thinkingExpanded ? (
              <div className="border-t border-anthropic-hairline px-4 py-3">
                <p className="whitespace-pre-wrap text-[13px] leading-[1.65] text-anthropic-body-strong">
                  {message.thinking}
                </p>
              </div>
            ) : null}
          </div>
        ) : null}

        {/* Waiting dots */}
        {isWaiting ? (
          <div className="flex items-center gap-1.5 text-anthropic-muted">
            <span className="ai-wait-dots" aria-hidden="true">
              <span className="ai-wait-dot" />
              <span className="ai-wait-dot" />
              <span className="ai-wait-dot" />
            </span>
            <span className="anthropic-mono text-[11.5px]">检索证据中</span>
          </div>
        ) : null}

        {/* Answer body */}
        {hasContent ? (
          <div className="relative">
            <AnthropicMarkdown content={message.content} />
            {isStreamingThis ? <span className="anthropic-stream-cursor" /> : null}
          </div>
        ) : null}

        {/* Citations */}
        <AnthropicCitations citations={message.citations} />

        {/* Error */}
        {message.status === "error" ? (
          <p className="text-[12.5px] text-anthropic-error">生成已中断。</p>
        ) : null}

        {/* Feedback toolbar */}
        {showFeedback ? (
          <div className="flex items-center gap-1 border-t border-anthropic-hairline-soft pt-3">
            <FeedbackButtons
              messageId={message.id}
              feedback={message.feedback ?? null}
              content={message.content}
              alwaysVisible={Boolean(isLast)}
            />
            <span className="ml-auto anthropic-mono text-[11px] text-anthropic-muted-soft">
              trace · {message.id.slice(0, 10)}
            </span>
          </div>
        ) : null}
      </div>
    </div>
  );
});