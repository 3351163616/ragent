import * as React from "react";

import { AnthropicComposer } from "@/components/chat-anthropic/AnthropicComposer";
import { AnthropicMessageThread } from "@/components/chat-anthropic/AnthropicMessageThread";
import { AnthropicSidebar } from "@/components/chat-anthropic/AnthropicSidebar";
import { AnthropicTopNav } from "@/components/chat-anthropic/AnthropicTopNav";
import { AnthropicWelcome } from "@/components/chat-anthropic/AnthropicWelcome";
import { useChatStore } from "@/stores/chatStore";

/**
 * Anthropic Edition 顶层布局（双栏）：
 *   ┌─────────────────── topnav (64px) ───────────────────┐
 *   │ sidebar (264px) │ main (1fr)                        │
 *   └─────────────────────────────────────────────────────┘
 *
 * < 768px 时 sidebar 变为 overlay drawer。
 */
export function AnthropicChatView() {
  const messages = useChatStore((s) => s.messages);
  const isLoading = useChatStore((s) => s.isLoading);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const currentSessionId = useChatStore((s) => s.currentSessionId);

  const showWelcome = messages.length === 0 && !isLoading;
  const [sidebarOpen, setSidebarOpen] = React.useState(false);
  const [isMobile, setIsMobile] = React.useState(false);

  React.useEffect(() => {
    const mq = window.matchMedia("(max-width: 767px)");
    const apply = () => setIsMobile(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  const sidebarVisible = isMobile ? sidebarOpen : true;

  return (
    <div
      className="anthropic-edition grid h-screen w-full overflow-hidden bg-anthropic-canvas text-anthropic-ink"
      style={{
        gridTemplateColumns: isMobile ? "minmax(0,1fr)" : "264px minmax(0,1fr)",
        gridTemplateRows: "64px minmax(0,1fr)"
      }}
    >
      <div className="col-span-full row-start-1">
        <AnthropicTopNav
          onToggleSidebar={() => setSidebarOpen((p) => !p)}
          isMobile={isMobile}
        />
      </div>

      {/* Desktop sidebar */}
      {!isMobile ? (
        <div className="row-start-2 min-h-0 overflow-hidden border-r border-anthropic-hairline-soft bg-anthropic-canvas">
          <AnthropicSidebar />
        </div>
      ) : null}

      {/* Mobile overlay */}
      {isMobile && sidebarOpen ? (
        <>
          <div
            className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="fixed left-0 top-0 z-50 h-full w-[280px] border-r border-anthropic-hairline-soft bg-anthropic-canvas shadow-anthropic-card">
            <div className="h-full pt-[64px]">
              <AnthropicSidebar onNavigate={() => setSidebarOpen(false)} />
            </div>
          </div>
        </>
      ) : null}

      <div className="relative row-start-2 flex min-h-0 flex-col overflow-hidden bg-anthropic-canvas">
        <div className="flex-1 min-h-0 overflow-hidden">
          {showWelcome ? (
            <AnthropicWelcome />
          ) : (
            <AnthropicMessageThread
              messages={messages}
              isLoading={isLoading}
              isStreaming={isStreaming}
              sessionKey={currentSessionId}
            />
          )}
        </div>
        {showWelcome ? null : (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 z-30 bg-gradient-to-t from-anthropic-canvas via-anthropic-canvas/90 to-transparent px-6 pb-6 pt-12">
            <div className="mx-auto w-full max-w-[760px]">
              <AnthropicComposer />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
