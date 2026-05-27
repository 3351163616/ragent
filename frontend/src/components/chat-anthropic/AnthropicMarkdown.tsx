// @ts-nocheck
/* eslint-disable */

import * as React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy, ImageIcon } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface AnthropicMarkdownProps {
  content: string;
}

/**
 * Anthropic 版 Markdown 渲染器：
 *   - 衬线 H1-H4（在 globals.css 的 .anthropic-prose 中定义）
 *   - 代码块使用深海军蓝 oneDark 主题
 *   - 内联代码用 surface-card 浅色卡片
 *   - 中英混排自动 palt
 */
export function AnthropicMarkdown({ content }: AnthropicMarkdownProps) {
  return (
    <div className="anthropic-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ inline, className, children, node, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const language = match?.[1] || "text";
            const value = String(children).replace(/\n$/, "");

            if (inline || !value.includes("\n")) {
              return (
                <code className={cn(className)} {...props}>
                  {children}
                </code>
              );
            }

            return (
              <div className="my-4 overflow-hidden rounded-anthropic-lg border border-[#2d2b27] bg-[#181715]">
                <div className="flex items-center justify-between border-b border-[#2d2b27] px-4 py-2">
                  <span className="anthropic-mono text-[11px] font-medium uppercase tracking-wider text-[#a09d96]">
                    {language}
                  </span>
                  <CopyButton value={value} />
                </div>
                <div className="overflow-x-auto">
                  <SyntaxHighlighter
                    language={language}
                    style={oneDark}
                    PreTag="div"
                    customStyle={{
                      margin: 0,
                      padding: "16px 18px",
                      background: "transparent",
                      fontSize: "13px",
                      lineHeight: "1.65",
                      fontFamily:
                        "'JetBrains Mono', 'Sarasa Mono SC', ui-monospace, monospace"
                    }}
                    showLineNumbers={false}
                    wrapLines={true}
                  >
                    {value}
                  </SyntaxHighlighter>
                </div>
              </div>
            );
          },
          img({ src, alt, ...props }) {
            const [hasError, setHasError] = React.useState(false);
            if (hasError) {
              return (
                <span className="my-3 inline-flex items-center gap-2 text-[13px] text-anthropic-muted">
                  <ImageIcon className="h-4 w-4" />
                  图片加载失败
                </span>
              );
            }
            return (
              <img
                src={src}
                alt={alt || ""}
                className="my-4 max-w-full rounded-anthropic-lg border border-anthropic-hairline"
                onError={() => setHasError(true)}
                loading="lazy"
                {...props}
              />
            );
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function CopyButton({ value }: { value: string }) {
  const [copied, setCopied] = React.useState(false);
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={handleCopy}
      aria-label="复制代码"
      className="h-7 w-7 text-[#a09d96] hover:bg-[#252320] hover:text-anthropic-on-dark"
    >
      {copied ? (
        <Check className="h-3.5 w-3.5 text-[#5db872]" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
    </Button>
  );
}