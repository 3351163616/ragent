import * as React from "react";
import { ArrowUpRight, BookOpen, Check, Lightbulb } from "lucide-react";

import { AnthropicComposer } from "@/components/chat-anthropic/AnthropicComposer";
import { Spike } from "@/components/chat-anthropic/Spike";
import { cn } from "@/lib/utils";
import { listSampleQuestions } from "@/services/sampleQuestionService";
import { useChatStore } from "@/stores/chatStore";

type PromptPreset = {
  id?: string;
  title: string;
  description: string;
  prompt: string;
  icon: React.ComponentType<{ className?: string }>;
};

const PRESET_ICONS = [BookOpen, Check, Lightbulb];

const DEFAULT_PRESETS: PromptPreset[] = [
  {
    title: "内容总结",
    description: "提炼 3-5 条关键信息与行动点",
    prompt: "请帮我总结以下内容，并列出 3-5 条要点：",
    icon: BookOpen
  },
  {
    title: "任务拆解",
    description: "把目标拆成可执行步骤与优先级",
    prompt: "请把下面需求拆解为步骤，并给出优先级和里程碑：",
    icon: Check
  },
  {
    title: "灵感扩展",
    description: "给出多个方案并比较优缺点",
    prompt: "围绕以下主题给出 5-8 个方案，并注明优缺点：",
    icon: Lightbulb
  }
];

/**
 * Welcome 屏 — Anthropic 编辑风：
 *   1. Eyebrow tag (spike-mark + RAG 标签)
 *   2. Hero serif headline
 *   3. Lead paragraph
 *   4. Composer (sticky 上方居中)
 *   5. Preset 卡片网格 (3-up，深色 hover-out 风)
 */
export function AnthropicWelcome() {
  const sendMessage = useChatStore((s) => s.sendMessage);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const [presets, setPresets] = React.useState<PromptPreset[]>(DEFAULT_PRESETS);

  React.useEffect(() => {
    let active = true;
    listSampleQuestions()
      .then((data) => {
        if (!active || !data || data.length === 0) return;
        const mapped = data
          .filter((it) => it.question && it.question.trim())
          .slice(0, 3)
          .map((it, i) => {
            const question = it.question.trim();
            const title =
              it.title?.trim() ||
              (question.length > 12 ? `${question.slice(0, 12)}...` : question) ||
              `推荐问法 ${i + 1}`;
            const description = it.description?.trim() || "点击即可开始对话";
            return {
              id: it.id,
              title,
              description,
              prompt: question,
              icon: PRESET_ICONS[i % PRESET_ICONS.length]
            };
          });
        if (mapped.length > 0) setPresets(mapped);
      })
      .catch(() => null);
    return () => {
      active = false;
    };
  }, []);

  const applyPreset = (prompt: string) => {
    if (isStreaming) return;
    sendMessage(prompt).catch(() => null);
  };

  return (
    <div className="relative flex h-full w-full items-center justify-center overflow-y-auto px-6 py-16 anthropic-scroll">
      <div className="relative w-full max-w-[760px]">
        <div
          className="flex flex-col gap-6 text-left"
          style={{ animation: "rise 0.7s cubic-bezier(0.16,1,0.3,1) both" }}
        >
          <span className="inline-flex w-fit items-center gap-2 rounded-full border border-anthropic-hairline bg-anthropic-canvas px-3 py-1 text-[11px] font-medium uppercase tracking-[1.5px] text-anthropic-muted">
            <Spike size={12} className="text-anthropic-coral" />
            RAG 智能问答
          </span>
          <h1 className="anthropic-display text-[44px] leading-[1.08] text-anthropic-ink md:text-[52px]">
            把问题变成清晰答案
          </h1>
          <p className="max-w-[600px] text-[17px] leading-[1.6] text-anthropic-body">
            结构化提问、知识检索与深度思考，一次对话给出可执行方案
          </p>
        </div>

        <div
          className="mt-10"
          style={{ animation: "rise 0.7s 0.1s cubic-bezier(0.16,1,0.3,1) both" }}
        >
          <AnthropicComposer />
        </div>

        <div
          className="mt-12"
          style={{ animation: "rise 0.7s 0.18s cubic-bezier(0.16,1,0.3,1) both" }}
        >
          <div className="mb-5 flex items-center gap-3 anthropic-mono text-[11px] uppercase tracking-[1.5px] text-anthropic-muted-soft">
            <span className="h-px flex-1 bg-anthropic-hairline" />
            试试这些开场
            <span className="h-px flex-1 bg-anthropic-hairline" />
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            {presets.map((preset) => {
              const Icon = preset.icon;
              return (
                <button
                  key={preset.id ?? preset.title}
                  type="button"
                  onClick={() => applyPreset(preset.prompt)}
                  disabled={isStreaming}
                  className={cn(
                    "group flex flex-col gap-3 rounded-anthropic-lg border border-anthropic-hairline bg-anthropic-canvas p-5 text-left transition-all duration-200 hover:-translate-y-0.5 hover:border-anthropic-coral/40 hover:shadow-anthropic-card",
                    isStreaming && "cursor-not-allowed opacity-60"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="flex h-9 w-9 items-center justify-center rounded-anthropic-md bg-anthropic-surface-card text-anthropic-coral">
                      <Icon className="h-4 w-4" />
                    </span>
                    <ArrowUpRight className="h-4 w-4 text-anthropic-muted-soft transition-colors group-hover:text-anthropic-coral" />
                  </div>
                  <div>
                    <p className="text-[14.5px] font-medium text-anthropic-ink">{preset.title}</p>
                    <p className="mt-1 text-[12.5px] leading-[1.5] text-anthropic-muted">
                      {preset.description}
                    </p>
                  </div>
                  <p className="mt-auto line-clamp-2 anthropic-mono text-[11.5px] leading-[1.5] text-anthropic-muted-soft">
                    {preset.prompt}
                  </p>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}