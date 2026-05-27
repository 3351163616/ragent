import * as React from "react";

import { cn } from "@/lib/utils";

interface SpikeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  className?: string;
}

/**
 * Anthropic 风格的 4-spoke 星号 brand mark。
 * 用于 logo、avatar 与内联强调。
 */
export function Spike({ size = 18, className, ...props }: SpikeProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      className={cn(className)}
      {...props}
    >
      <path
        fill="currentColor"
        d="M12 1 13.4 9.2 21 6.5l-5.5 6.1L22 16.4l-8.1-1.1L13.5 23 12 15.2 10.5 23l-.4-7.7L2 16.4l6.5-3.8L3 6.5l7.6 2.7L12 1Z"
      />
    </svg>
  );
}