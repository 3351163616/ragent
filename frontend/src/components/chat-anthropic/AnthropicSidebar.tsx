import * as React from "react";
import { differenceInCalendarDays, isValid } from "date-fns";
import { MessageSquare, MoreHorizontal, Pencil, Plus, Search, Settings, Trash2 } from "lucide-react";
import { useNavigate } from "react-router-dom";

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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { Loading } from "@/components/common/Loading";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/stores/authStore";
import { useChatStore } from "@/stores/chatStore";

interface AnthropicSidebarProps {
  onNavigate?: () => void;
}

export function AnthropicSidebar({ onNavigate }: AnthropicSidebarProps) {
  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const sessions = useChatStore((s) => s.sessions);
  const currentSessionId = useChatStore((s) => s.currentSessionId);
  const isLoading = useChatStore((s) => s.isLoading);
  const sessionsLoaded = useChatStore((s) => s.sessionsLoaded);
  const createSession = useChatStore((s) => s.createSession);
  const deleteSession = useChatStore((s) => s.deleteSession);
  const renameSession = useChatStore((s) => s.renameSession);
  const selectSession = useChatStore((s) => s.selectSession);
  const fetchSessions = useChatStore((s) => s.fetchSessions);

  const [query, setQuery] = React.useState("");
  const [renamingId, setRenamingId] = React.useState<string | null>(null);
  const [renameValue, setRenameValue] = React.useState("");
  const [deleteTarget, setDeleteTarget] = React.useState<{ id: string; title: string } | null>(
    null
  );
  const renameInputRef = React.useRef<HTMLInputElement | null>(null);

  React.useEffect(() => {
    if (sessions.length === 0) {
      fetchSessions().catch(() => null);
    }
  }, [fetchSessions, sessions.length]);

  React.useEffect(() => {
    if (renamingId) {
      renameInputRef.current?.focus();
      renameInputRef.current?.select();
    }
  }, [renamingId]);

  const filtered = React.useMemo(() => {
    const kw = query.trim().toLowerCase();
    if (!kw) return sessions;
    return sessions.filter((s) => (s.title || "").toLowerCase().includes(kw));
  }, [query, sessions]);

  const grouped = React.useMemo(() => {
    const now = new Date();
    const map = new Map<string, typeof filtered>();
    const order: string[] = [];
    const label = (value?: string) => {
      const d = value ? new Date(value) : now;
      const day = isValid(d) ? d : now;
      const diff = Math.max(0, differenceInCalendarDays(now, day));
      if (diff === 0) return "Today";
      if (diff <= 7) return "Past 7 days";
      if (diff <= 30) return "Past 30 days";
      return "Earlier";
    };
    filtered.forEach((s) => {
      const l = label(s.lastTime);
      if (!map.has(l)) {
        map.set(l, []);
        order.push(l);
      }
      map.get(l)?.push(s);
    });
    return order.map((l) => ({ label: l, items: map.get(l) || [] }));
  }, [filtered]);

  const startRename = (id: string, title: string) => {
    setRenamingId(id);
    setRenameValue(title || "新对话");
  };
  const cancelRename = () => {
    setRenamingId(null);
    setRenameValue("");
  };
  const commitRename = async () => {
    if (!renamingId) return;
    const next = renameValue.trim();
    if (!next) {
      cancelRename();
      return;
    }
    const current = sessions.find((s) => s.id === renamingId)?.title || "新对话";
    if (next === current) {
      cancelRename();
      return;
    }
    await renameSession(renamingId, next);
    cancelRename();
  };

  return (
    <aside className="flex h-full min-h-0 flex-col gap-5 overflow-hidden px-4 py-5">
      <button
        type="button"
        onClick={() => {
          createSession().catch(() => null);
          navigate("/chat");
          onNavigate?.();
        }}
        className="flex items-center justify-between rounded-anthropic-md bg-anthropic-surface-card px-3.5 py-3 text-left text-[13.5px] font-medium text-anthropic-ink transition-colors hover:bg-anthropic-surface-strong"
      >
        <span className="flex items-center gap-2.5">
          <Plus className="h-4 w-4" />
          新建对话
        </span>
        <span className="anthropic-mono rounded-anthropic-xs border border-anthropic-hairline bg-anthropic-canvas px-1.5 py-0.5 text-[10.5px] text-anthropic-muted">
          ⌘ K
        </span>
      </button>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-anthropic-muted-soft" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="搜索对话"
          className="h-9 w-full rounded-anthropic-md border border-anthropic-hairline-soft bg-anthropic-canvas pl-9 pr-3 text-[13px] text-anthropic-ink placeholder:text-anthropic-muted-soft focus:border-anthropic-coral focus:outline-none"
        />
      </div>

      <div className="anthropic-scroll flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto pr-1">
        {sessions.length === 0 && (!sessionsLoaded || isLoading) ? (
          <div className="flex h-32 items-center justify-center text-anthropic-muted">
            <Loading label="加载会话中" />
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex h-32 flex-col items-center justify-center gap-2 text-anthropic-muted">
            <MessageSquare className="h-9 w-9 opacity-60" />
            <span className="text-[12.5px]">暂无对话记录</span>
          </div>
        ) : (
          grouped.map((g) => (
            <div key={g.label} className="mb-2 flex flex-col gap-0.5">
              <div className="px-2.5 pb-1.5 anthropic-mono text-[10.5px] uppercase tracking-[1.5px] text-anthropic-muted-soft">
                {g.label}
              </div>
              {g.items.map((s) => {
                const active = currentSessionId === s.id;
                return (
                  <div
                    key={s.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => {
                      if (renamingId === s.id) return;
                      if (renamingId) cancelRename();
                      selectSession(s.id).catch(() => null);
                      navigate(`/chat/${s.id}`);
                      onNavigate?.();
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        selectSession(s.id).catch(() => null);
                        navigate(`/chat/${s.id}`);
                        onNavigate?.();
                      }
                    }}
                    className={cn(
                      "group flex cursor-pointer items-center justify-between gap-2 rounded-anthropic-md px-3 py-2.5 transition-colors duration-150",
                      active
                        ? "bg-anthropic-surface-card text-anthropic-ink"
                        : "text-anthropic-body hover:bg-anthropic-surface-soft hover:text-anthropic-ink"
                    )}
                  >
                    {renamingId === s.id ? (
                      <input
                        ref={renameInputRef}
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onClick={(e) => e.stopPropagation()}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            commitRename().catch(() => null);
                          }
                          if (e.key === "Escape") {
                            e.preventDefault();
                            cancelRename();
                          }
                        }}
                        onBlur={() => commitRename().catch(() => null)}
                        className="h-6 flex-1 rounded-anthropic-sm border border-anthropic-hairline bg-anthropic-canvas px-2 text-[13px] text-anthropic-ink focus:border-anthropic-coral focus:outline-none"
                      />
                    ) : (
                      <span className="min-w-0 flex-1 truncate text-[13.5px] font-medium">
                        {s.title || "新对话"}
                      </span>
                    )}
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <button
                          type="button"
                          className={cn(
                            "flex h-6 w-6 items-center justify-center rounded text-anthropic-muted opacity-0 transition-opacity hover:bg-black/5 group-hover:opacity-100",
                            active && "opacity-100"
                          )}
                          onClick={(e) => e.stopPropagation()}
                          aria-label="会话操作"
                        >
                          <MoreHorizontal className="h-3.5 w-3.5" />
                        </button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent
                        align="start"
                        className="min-w-[120px] rounded-anthropic-md border border-anthropic-hairline bg-anthropic-canvas p-1 shadow-anthropic-card"
                      >
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            startRename(s.id, s.title || "新对话");
                          }}
                          className="rounded-anthropic-sm px-2.5 py-1.5 text-[13px] text-anthropic-ink data-[highlighted]:bg-anthropic-surface-card data-[highlighted]:text-anthropic-ink"
                        >
                          <Pencil className="mr-2 h-3.5 w-3.5" />
                          重命名
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            setDeleteTarget({ id: s.id, title: s.title || "新对话" });
                          }}
                          className="rounded-anthropic-sm px-2.5 py-1.5 text-[13px] text-anthropic-error data-[highlighted]:bg-anthropic-surface-card data-[highlighted]:text-anthropic-error"
                        >
                          <Trash2 className="mr-2 h-3.5 w-3.5" />
                          删除
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                );
              })}
            </div>
          ))
        )}
      </div>

      <div className="flex flex-none flex-col gap-4 border-t border-anthropic-hairline-soft pt-4">
        {user?.role === "admin" ? (
          <button
            type="button"
            onClick={() => window.open("/admin", "_blank")}
            className="flex items-center gap-2.5 rounded-anthropic-md border border-anthropic-hairline bg-anthropic-canvas px-3.5 py-2.5 text-[13px] font-medium text-anthropic-ink transition-colors hover:bg-anthropic-surface-card"
          >
            <Settings className="h-4 w-4 text-anthropic-coral" />
            管理后台
          </button>
        ) : null}

        <div className="flex items-center gap-2.5 rounded-anthropic-md border border-anthropic-hairline bg-anthropic-canvas p-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-[#cc785c] to-[#a9583e] text-[13px] font-semibold text-white">
            {(user?.username || user?.userId || "U").slice(0, 1).toUpperCase()}
          </div>
          <div className="flex min-w-0 flex-1 flex-col leading-tight">
            <span className="truncate text-[13px] font-medium text-anthropic-ink">
              {(() => {
                const fallback = user?.username || user?.userId || "用户";
                return /^\d+$/.test(fallback) ? "用户" : fallback;
              })()}
            </span>
            <span className="text-[11px] text-anthropic-muted">
              {user?.role === "admin" ? "Admin" : "Member"}
            </span>
          </div>
        </div>
      </div>

      <AlertDialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除该会话？</AlertDialogTitle>
            <AlertDialogDescription>
              [{deleteTarget?.title || "该会话"}] 将被永久删除，无法恢复。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (!deleteTarget) return;
                const target = deleteTarget;
                const isCurrent = currentSessionId === target.id;
                setDeleteTarget(null);
                deleteSession(target.id)
                  .then(() => {
                    if (isCurrent) navigate("/chat");
                  })
                  .catch(() => null);
              }}
            >
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </aside>
  );
}
