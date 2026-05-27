import { Menu } from "lucide-react";

import { Spike } from "@/components/chat-anthropic/Spike";
import { useAuthStore } from "@/stores/authStore";
import { useChatStore } from "@/stores/chatStore";

interface AnthropicTopNavProps {
  onToggleSidebar: () => void;
  isMobile: boolean;
}

export function AnthropicTopNav({ onToggleSidebar, isMobile }: AnthropicTopNavProps) {
  const isStreaming = useChatStore((s) => s.isStreaming);
  const selectedModelId = useChatStore((s) => s.selectedModelId);
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);

  const dotColor = isStreaming ? "#e8a55a" : "#5db872";
  const dotShadow = isStreaming
    ? "0 0 0 3px rgba(232, 165, 90, 0.18)"
    : "0 0 0 3px rgba(93, 184, 114, 0.18)";

  return (
    <header className="flex h-16 items-center gap-6 border-b border-anthropic-hairline-soft bg-anthropic-canvas px-6">
      {isMobile ? (
        <button
          type="button"
          onClick={onToggleSidebar}
          className="flex h-9 w-9 items-center justify-center rounded-anthropic-md text-anthropic-ink transition-colors hover:bg-anthropic-surface-card"
          aria-label="打开侧边栏"
        >
          <Menu className="h-5 w-5" />
        </button>
      ) : null}

      <a className="flex items-center gap-2.5 text-anthropic-ink" href="/chat">
        <Spike size={18} />
        <span className="anthropic-display text-[20px] leading-none">Ragent</span>
      </a>

      {!isMobile ? (
        <nav className="ml-2 flex items-center gap-0.5">
          <button
            type="button"
            className="rounded-anthropic-md bg-anthropic-surface-card px-3.5 py-2 text-[13.5px] font-medium text-anthropic-ink"
          >
            Chat
          </button>
        </nav>
      ) : null}

      <div className="ml-auto flex items-center gap-4">
        <span className="flex items-center gap-2 text-[12.5px] font-medium text-anthropic-muted">
          <span
            className="h-[7px] w-[7px] rounded-full"
            style={{ backgroundColor: dotColor, boxShadow: dotShadow }}
          />
          <span className="anthropic-mono hidden text-[11.5px] text-anthropic-muted sm:inline">
            {selectedModelId || "claude-sonnet-4-6"} ·{" "}
            {isStreaming ? "streaming" : "routing healthy"}
          </span>
        </span>

        {user ? (
          <button
            type="button"
            onClick={() => logout()}
            className="flex h-9 items-center gap-2 rounded-anthropic-md px-3 text-[13px] font-medium text-anthropic-ink transition-colors hover:bg-anthropic-surface-card"
          >
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-br from-[#cc785c] to-[#a9583e] text-[12px] font-semibold text-white">
              {(user.username || user.userId || "U").slice(0, 1).toUpperCase()}
            </span>
            <span className="hidden md:inline">{user.username || "Sign out"}</span>
          </button>
        ) : null}
      </div>
    </header>
  );
}
