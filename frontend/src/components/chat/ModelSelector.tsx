import * as React from "react";
import { ChevronDown } from "lucide-react";

import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores/chatStore";
import { useModelOptions } from "@/hooks/useModelOptions";

interface ModelSelectorProps {
  disabled?: boolean;
  variant?: "rounded" | "pill";
}

export function ModelSelector({ disabled, variant = "rounded" }: ModelSelectorProps) {
  const options = useModelOptions();
  const selectedModelId = useChatStore((s) => s.selectedModelId);
  const setSelectedModelId = useChatStore((s) => s.setSelectedModelId);
  const [open, setOpen] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);

  const selected = options.find((o) => o.id === selectedModelId) ?? options[0];

  React.useEffect(() => {
    if (options.length === 0) return;
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [options.length]);

  if (options.length <= 1) return null;

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        disabled={disabled}
        className={cn(
          "inline-flex items-center gap-1.5 text-xs font-medium transition-all",
          variant === "pill"
            ? cn(
                "rounded-full border px-3 py-1.5",
                selectedModelId
                  ? "border-[#BFDBFE] bg-[#DBEAFE] text-[#2563EB]"
                  : "border-transparent bg-[#F5F5F5] text-[#6B7280] hover:bg-[#EEEEEE]"
              )
            : cn(
                "rounded-lg border px-3 py-1.5",
                selectedModelId
                  ? "border-[#BFDBFE] bg-[#DBEAFE] text-[#2563EB]"
                  : "border-transparent bg-[#F5F5F5] text-[#999999] hover:bg-[#EEEEEE]"
              ),
          disabled && "cursor-not-allowed opacity-60"
        )}
      >
        {selected?.label ?? "自动选择"}
        <ChevronDown className={cn("h-3 w-3 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div
          className={cn(
            "absolute bottom-full left-0 z-50 mb-1 min-w-[160px] overflow-hidden rounded-lg border bg-white py-1 shadow-lg",
            variant === "pill" && "left-0"
          )}
        >
          {options.map((option) => (
            <button
              key={option.id}
              type="button"
              onClick={() => {
                setSelectedModelId(option.id === "auto" ? null : option.id);
                setOpen(false);
              }}
              className={cn(
                "flex w-full items-center gap-2 px-3 py-1.5 text-xs transition-colors hover:bg-[#F5F5F5]",
                (option.id === selectedModelId || (!selectedModelId && option.id === "auto"))
                  ? "bg-[#EFF6FF] text-[#2563EB] font-medium"
                  : "text-[#333333]"
              )}
            >
              <span className="flex-1 text-left">{option.label}</span>
              {option.provider && (
                <span className="text-[10px] text-[#999999] uppercase">{option.provider}</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
