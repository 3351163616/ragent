import * as React from "react";
import { Virtuoso, type VirtuosoHandle } from "react-virtuoso";

import { AnthropicMessageItem } from "@/components/chat-anthropic/AnthropicMessageItem";
import { cn } from "@/lib/utils";
import type { Message } from "@/types";

interface AnthropicMessageThreadProps {
  messages: Message[];
  isLoading: boolean;
  isStreaming: boolean;
  sessionKey?: string | null;
}

/**
 * Anthropic 版消息流容器 — 复用 legacy 的滚动行为（Virtuoso + auto-follow + 三阶段稳定）。
 */
export function AnthropicMessageThread({
  messages,
  isLoading,
  isStreaming,
  sessionKey
}: AnthropicMessageThreadProps) {
  const virtuosoRef = React.useRef<VirtuosoHandle | null>(null);
  const scrollerRef = React.useRef<HTMLElement | null>(null);
  const lastSessionRef = React.useRef<string | null>(null);
  const autoFollowRef = React.useRef(true);
  const pendingScrollRef = React.useRef(true);
  const settleTimerRef = React.useRef<number | null>(null);
  const heightScrollRafRef = React.useRef<number | null>(null);
  const prevStreamingRef = React.useRef(false);
  const initialTopMostItemIndex = React.useMemo(
    () => ({ index: "LAST" as const, align: "end" as const }),
    []
  );

  const scrollToBottom = React.useCallback(() => {
    virtuosoRef.current?.scrollToIndex({ index: "LAST", align: "end", behavior: "auto" });
    const scroller = scrollerRef.current;
    if (scroller) scroller.scrollTop = scroller.scrollHeight;
    autoFollowRef.current = true;
  }, []);

  const stickToBottom = React.useCallback(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    scroller.scrollTop = scroller.scrollHeight;
    autoFollowRef.current = true;
  }, []);

  const updateAutoFollow = React.useCallback(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const distance = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    const atBottom = distance < 80;
    autoFollowRef.current = atBottom;
    if (!atBottom) pendingScrollRef.current = false;
  }, []);

  React.useEffect(() => {
    const nextKey = sessionKey ?? "empty";
    if (lastSessionRef.current !== nextKey) {
      lastSessionRef.current = nextKey;
      autoFollowRef.current = true;
      pendingScrollRef.current = true;
      if (settleTimerRef.current) {
        window.clearTimeout(settleTimerRef.current);
        settleTimerRef.current = null;
      }
    }
  }, [sessionKey]);

  React.useEffect(() => {
    const wasStreaming = prevStreamingRef.current;
    prevStreamingRef.current = isStreaming;
    if (!wasStreaming && isStreaming) {
      if (!autoFollowRef.current && !pendingScrollRef.current) return;
      stickToBottom();
      const t = window.setTimeout(stickToBottom, 120);
      return () => window.clearTimeout(t);
    }
    if (wasStreaming && !isStreaming) {
      if (!autoFollowRef.current && !pendingScrollRef.current) return;
      scrollToBottom();
      const t1 = window.setTimeout(scrollToBottom, 120);
      const t2 = window.setTimeout(scrollToBottom, 360);
      return () => {
        window.clearTimeout(t1);
        window.clearTimeout(t2);
      };
    }
    return;
  }, [isStreaming, scrollToBottom, stickToBottom]);

  React.useLayoutEffect(() => {
    if (!pendingScrollRef.current || isStreaming || isLoading || messages.length === 0) return;
    let attempts = 0;
    let raf = 0;
    let active = true;
    const run = () => {
      scrollToBottom();
      attempts += 1;
      if (attempts < 3) raf = window.requestAnimationFrame(run);
    };
    run();
    const t1 = window.setTimeout(scrollToBottom, 240);
    const t2 = window.setTimeout(scrollToBottom, 900);
    const handleLoad = () => {
      if (active) scrollToBottom();
    };
    if (document.readyState === "complete") handleLoad();
    else window.addEventListener("load", handleLoad, { once: true });
    if (document.fonts?.ready) {
      document.fonts.ready.then(() => {
        if (active) scrollToBottom();
      });
    }
    if (settleTimerRef.current) window.clearTimeout(settleTimerRef.current);
    settleTimerRef.current = window.setTimeout(() => {
      pendingScrollRef.current = false;
      settleTimerRef.current = null;
    }, 1500);
    return () => {
      active = false;
      window.cancelAnimationFrame(raf);
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      if (settleTimerRef.current) {
        window.clearTimeout(settleTimerRef.current);
        settleTimerRef.current = null;
      }
      window.removeEventListener("load", handleLoad);
    };
  }, [messages.length, isStreaming, isLoading, sessionKey, scrollToBottom]);

  React.useEffect(() => {
    return () => {
      scrollerRef.current?.removeEventListener("scroll", updateAutoFollow);
      if (heightScrollRafRef.current) {
        window.cancelAnimationFrame(heightScrollRafRef.current);
        heightScrollRafRef.current = null;
      }
      if (settleTimerRef.current) {
        window.clearTimeout(settleTimerRef.current);
        settleTimerRef.current = null;
      }
    };
  }, [updateAutoFollow]);

  const handleTotalListHeightChanged = React.useCallback(() => {
    if (isLoading) return;
    const shouldStick = pendingScrollRef.current || autoFollowRef.current;
    if (!shouldStick) return;
    if (heightScrollRafRef.current) return;
    heightScrollRafRef.current = window.requestAnimationFrame(() => {
      heightScrollRafRef.current = null;
      if (isStreaming) {
        if (autoFollowRef.current || pendingScrollRef.current) stickToBottom();
        return;
      }
      scrollToBottom();
    });
  }, [isStreaming, isLoading, scrollToBottom, stickToBottom]);

  // 三击防扩展选区（与 legacy 同款）
  const handleTripleClickDown = React.useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (e.detail < 3) return;
    e.preventDefault();
    const target = e.target as HTMLElement;
    const block = target.closest("p, li, h1, h2, h3, h4, h5, h6, pre, blockquote, td, th");
    const container = block && e.currentTarget.contains(block) ? block : e.currentTarget;
    const sel = window.getSelection();
    if (sel) {
      const range = document.createRange();
      range.selectNodeContents(container);
      sel.removeAllRanges();
      sel.addRange(range);
    }
  }, []);

  const List = React.useMemo(() => {
    const Comp = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
      ({ className, ...props }, ref) => (
        <div
          ref={ref}
          className={cn(
            "mx-auto flex max-w-[760px] flex-col gap-12 px-6 pb-24 pt-12 md:px-8",
            className
          )}
          {...props}
        />
      )
    );
    Comp.displayName = "AnthropicThreadList";
    return Comp;
  }, []);

  const Footer = React.useMemo(() => {
    const Comp = () => <div aria-hidden="true" className="h-32" />;
    Comp.displayName = "AnthropicThreadFooter";
    return Comp;
  }, []);

  if (messages.length === 0) return <div className="h-full" />;

  return (
    <Virtuoso
      key={sessionKey ?? "empty"}
      ref={virtuosoRef}
      data={messages}
      initialTopMostItemIndex={initialTopMostItemIndex}
      followOutput={(atBottom) => {
        if (isStreaming) return false;
        return atBottom ? "auto" : false;
      }}
      atBottomStateChange={(atBottom) => {
        if (atBottom) autoFollowRef.current = true;
      }}
      scrollerRef={(node) => {
        const scroller = node as HTMLElement | null;
        if (scrollerRef.current && scrollerRef.current !== scroller) {
          scrollerRef.current.removeEventListener("scroll", updateAutoFollow);
        }
        if (scroller && scrollerRef.current !== scroller) {
          scroller.addEventListener("scroll", updateAutoFollow, { passive: true });
          scrollerRef.current = scroller;
          updateAutoFollow();
          return;
        }
        scrollerRef.current = scroller;
      }}
      totalListHeightChanged={handleTotalListHeightChanged}
      className="h-full anthropic-scroll"
      components={{ List, Footer }}
      itemContent={(index, message) => (
        <div
          className={cn(
            "transition-opacity",
            index === messages.length - 1 && "animate-anthropic-rise"
          )}
          onMouseDown={handleTripleClickDown}
        >
          <AnthropicMessageItem message={message} isLast={index === messages.length - 1} />
        </div>
      )}
    />
  );
}