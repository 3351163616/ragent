import * as React from "react";
import { Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";
import { useEditionStore, type ChatEdition } from "@/stores/editionStore";

interface EditionSwitcherProps {
  className?: string;
}

const OPTIONS: { value: ChatEdition; label: string; hint: string }[] = [
  { value: "legacy", label: "Legacy", hint: "原版蓝调界面" },
  { value: "anthropic", label: "Anthropic Edition", hint: "暖奶油 · 珊瑚 · 衬线" }
];

export function EditionSwitcher({ className }: EditionSwitcherProps) {
  const edition = useEditionStore((s) => s.edition);
  const setEdition = useEditionStore((s) => s.setEdition);

  return (
    <div
      className={cn(
        "fixed bottom-5 right-5 z-[60] flex items-center gap-1 rounded-full border border-black/5 bg-white/90 p-1 shadow-[0_18px_40px_-22px_rgba(20,20,19,0.45)] backdrop-blur",
        className
      )}
      role="group"
      aria-label="切换前端版本"
    >
      <Sparkles className="ml-2 mr-0.5 h-3.5 w-3.5 text-[#cc785c]" />
      {OPTIONS.map((option) => {
        const active = edition === option.value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => setEdition(option.value)}
            title={option.hint}
            aria-pressed={active}
            className={cn(
              "rounded-full px-3 py-1.5 text-xs font-medium transition-all duration-200",
              active
                ? option.value === "anthropic"
                  ? "bg-[#cc785c] text-white shadow-sm"
                  : "bg-[#1f2937] text-white shadow-sm"
                : "text-slate-500 hover:text-slate-900"
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
