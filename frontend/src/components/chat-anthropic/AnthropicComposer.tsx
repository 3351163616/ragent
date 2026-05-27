import * as React from "react";
import { Brain, Layers, Send, Square, X } from "lucide-react";

import { ModelSelector } from "@/components/chat/ModelSelector";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores/chatStore";

export function AnthropicComposer() {
  const [value, setValue] = React.useState("");
  const [isFocused, setIsFocused] = React.useState(false);
  const isComposingRef = React.useRef(false);
  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const cancelGeneration = useChatStore((s) => s.cancelGeneration);
  const deepThinkingEnabled = useChatStore((s) => s.deepThinkingEnabled);
  const setDeepThinkingEnabled = useChatStore((s) => s.setDeepThinkingEnabled);
  const inputFocusKey = useChatStore((s) => s.inputFocusKey);
  const editingMessageId = useChatStore((s) => s.editingMessageId);
  const editingMessageContent = useChatStore((s) => s.editingMessageContent);
  const clearEditingMessage = useChatStore((s) => s.clearEditingMessage);
  const resendEditedMessage = useChatStore((s) => s.resendEditedMessage);

  const focusInput = React.useCallback(() => {
    textareaRef.current?.focus({ preventScroll: true });
  }, []);

  const adjustHeight = React.useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  React.useEffect(() => {
    adjustHeight();
  }, [value, adjustHeight]);

  React.useEffect(() => {
    if (!inputFocusKey) return;
    focusInput();
  }, [inputFocusKey, focusInput]);

  React.useEffect(() => {
    if (!editingMessageId) return;
    setValue(editingMessageContent);
    window.requestAnimationFrame(() => {
      adjustHeight();
      focusInput();
    });
  }, [editingMessageId, editingMessageContent, adjustHeight, focusInput]);

  const hasContent = value.trim().length > 0;
  const isEditing = Boolean(editingMessageId);

  const handleSubmit = async () => {
    if (isStreaming) {
      cancelGeneration();
      focusInput();
      return;
    }
    if (!value.trim()) return;
    const next = value;
    setValue("");
    focusInput();
    if (editingMessageId) {
      const ok = await resendEditedMessage(next);
      if (!ok) setValue(next);
    } else {
      await sendMessage(next);
    }
    focusInput();
  };

  return (
    <div className="pointer-events-auto flex flex-col gap-2">
      <div
        className={cn(
          "rounded-anthropic-lg border bg-anthropic-canvas px-4 pt-3 pb-2.5 transition-all duration-200",
          isFocused
            ? "border-anthropic-coral shadow-anthropic-coral-focus"
            : "border-anthropic-hairline shadow-anthropic-card hover:border-anthropic-hairline-dark/30"
        )}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={
            deepThinkingEnabled ? "输入需要深度分析的问题..." : "继续提问，Ragent 会引用文档回答..."
          }
          rows={1}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onCompositionStart={() => {
            isComposingRef.current = true;
          }}
          onCompositionEnd={() => {
            isComposingRef.current = false;
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              const native = e.nativeEvent as KeyboardEvent;
              if (native.isComposing || isComposingRef.current || native.keyCode === 229) return;
              e.preventDefault();
              handleSubmit();
            }
          }}
          className="max-h-[200px] min-h-[28px] w-full resize-none border-0 bg-transparent px-0 py-1.5 text-[15px] leading-[1.6] text-anthropic-ink outline-none placeholder:text-anthropic-muted-soft"
          aria-label="聊天输入框"
        />

        {isEditing ? (
          <div className="mt-1.5 mb-1 flex items-center gap-2 rounded-anthropic-md bg-anthropic-surface-card px-3 py-1.5 text-[12px] text-anthropic-coral">
            <span className="min-w-0 flex-1 truncate">正在编辑历史消息</span>
            <button
              type="button"
              onClick={() => {
                clearEditingMessage();
                setValue("");
                focusInput();
              }}
              className="rounded-full p-1 text-anthropic-coral hover:bg-anthropic-canvas"
              aria-label="取消编辑"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        ) : null}

        <div className="mt-1 flex items-center justify-between">
          <div className="flex flex-wrap items-center gap-1">
            <button
              type="button"
              onClick={() => setDeepThinkingEnabled(!deepThinkingEnabled)}
              disabled={isStreaming || isEditing}
              aria-pressed={deepThinkingEnabled}
              className={cn(
                "flex items-center gap-1.5 rounded-anthropic-md px-2.5 py-1.5 text-[12.5px] font-medium transition-colors",
                deepThinkingEnabled
                  ? "bg-anthropic-surface-card text-anthropic-ink"
                  : "text-anthropic-muted hover:bg-anthropic-surface-soft hover:text-anthropic-ink",
                (isStreaming || isEditing) && "cursor-not-allowed opacity-60"
              )}
            >
              <Brain
                className={cn(
                  "h-3.5 w-3.5",
                  deepThinkingEnabled ? "text-anthropic-coral" : "text-anthropic-muted"
                )}
              />
              深度思考
              {deepThinkingEnabled ? (
                <span className="ml-0.5 h-1.5 w-1.5 animate-pulse rounded-full bg-anthropic-coral" />
              ) : null}
            </button>
            <span className="flex items-center gap-1 rounded-anthropic-md px-2.5 py-1.5 text-[12.5px] font-medium text-anthropic-muted">
              <Layers className="h-3.5 w-3.5" />
              <ModelSelector disabled={isStreaming || isEditing} />
            </span>
          </div>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!hasContent && !isStreaming}
            aria-label={isStreaming ? "停止生成" : "发送消息"}
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-anthropic-md transition-all duration-200",
              isStreaming
                ? "bg-anthropic-error/15 text-anthropic-error hover:bg-anthropic-error/25"
                : hasContent
                  ? "bg-anthropic-coral text-white hover:bg-anthropic-coral-active"
                  : "cursor-not-allowed bg-anthropic-surface-card text-anthropic-muted-soft"
            )}
          >
            {isStreaming ? <Square className="h-4 w-4" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
      </div>
      <p className="text-center anthropic-mono text-[11px] text-anthropic-muted-soft max-sm:hidden">
        Enter 发送 · Shift + Enter 换行 · ⌘ J 切深度思考
        {isStreaming ? (
          <span className="ml-2 animate-pulse text-anthropic-coral">生成中…</span>
        ) : null}
      </p>
    </div>
  );
}