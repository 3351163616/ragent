/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: "hsl(var(--card))",
        "card-foreground": "hsl(var(--card-foreground))",
        popover: "hsl(var(--popover))",
        "popover-foreground": "hsl(var(--popover-foreground))",
        primary: "hsl(var(--primary))",
        "primary-foreground": "hsl(var(--primary-foreground))",
        secondary: "hsl(var(--secondary))",
        "secondary-foreground": "hsl(var(--secondary-foreground))",
        muted: "hsl(var(--muted))",
        "muted-foreground": "hsl(var(--muted-foreground))",
        accent: "hsl(var(--accent))",
        "accent-foreground": "hsl(var(--accent-foreground))",
        destructive: "hsl(var(--destructive))",
        "destructive-foreground": "hsl(var(--destructive-foreground))",
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        "chat-user": "hsl(var(--chat-user))",
        "chat-assistant": "hsl(var(--chat-assistant))",
        anthropic: {
          canvas: "#faf9f5",
          "surface-soft": "#f5f0e8",
          "surface-card": "#efe9de",
          "surface-strong": "#e8e0d2",
          "surface-dark": "#181715",
          "surface-dark-elevated": "#252320",
          "surface-dark-soft": "#1f1e1b",
          ink: "#141413",
          "body-strong": "#252523",
          body: "#3d3d3a",
          muted: "#6c6a64",
          "muted-soft": "#8e8b82",
          hairline: "#e6dfd8",
          "hairline-soft": "#ebe6df",
          "hairline-dark": "#2d2b27",
          coral: "#cc785c",
          "coral-active": "#a9583e",
          "coral-disabled": "#e6dfd8",
          teal: "#5db8a6",
          amber: "#e8a55a",
          success: "#5db872",
          warning: "#d4a017",
          error: "#c64545",
          "on-primary": "#ffffff",
          "on-dark": "#faf9f5",
          "on-dark-soft": "#a09d96"
        }
      },
      fontFamily: {
        display: ["'Space Grotesk'", "ui-sans-serif", "system-ui"],
        body: ["'DM Sans'", "ui-sans-serif", "system-ui"],
        mono: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular"],
        "anthropic-display": [
          "'EB Garamond'",
          "'Tiempos Headline'",
          "'Cormorant Garamond'",
          "'Noto Serif SC'",
          "'Source Han Serif SC'",
          "'Songti SC'",
          "'Times New Roman'",
          "serif"
        ],
        "anthropic-body": [
          "'Inter'",
          "-apple-system",
          "BlinkMacSystemFont",
          "'Segoe UI'",
          "'PingFang SC'",
          "'Noto Sans SC'",
          "'Microsoft YaHei'",
          "Roboto",
          "system-ui",
          "sans-serif"
        ],
        "anthropic-mono": [
          "'JetBrains Mono'",
          "'Sarasa Mono SC'",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace"
        ]
      },
      borderRadius: {
        "anthropic-xs": "4px",
        "anthropic-sm": "6px",
        "anthropic-md": "8px",
        "anthropic-lg": "12px",
        "anthropic-xl": "16px"
      },
      spacing: {
        "anthropic-section": "96px"
      },
      boxShadow: {
        soft: "0 24px 60px -30px rgba(10, 10, 15, 0.65)",
        glow: "0 0 0 1px rgba(59, 130, 246, 0.2), 0 16px 40px rgba(59, 130, 246, 0.25)",
        neon: "0 0 30px rgba(59, 130, 246, 0.35)",
        "anthropic-card":
          "0 12px 40px -20px rgba(20, 20, 19, 0.18), 0 2px 8px -2px rgba(20, 20, 19, 0.06)",
        "anthropic-coral-focus":
          "0 0 0 3px rgba(204, 120, 92, 0.16), 0 12px 40px -20px rgba(20, 20, 19, 0.18)",
        "anthropic-dark-lift": "0 18px 40px -22px rgba(20, 20, 19, 0.55)"
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: 0, transform: "translateY(10px)" },
          "100%": { opacity: 1, transform: "translateY(0)" }
        },
        "pulse-soft": {
          "0%, 100%": { opacity: 1 },
          "50%": { opacity: 0.5 }
        },
        "blink": {
          "0%, 100%": { opacity: 1 },
          "50%": { opacity: 0 }
        },
        "spin-slow": {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" }
        },
        "glow": {
          "0%, 100%": { opacity: 0.5 },
          "50%": { opacity: 1 }
        },
        "float": {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-6px)" }
        },
        "rise": {
          "0%": { opacity: 0, transform: "translateY(12px)" },
          "100%": { opacity: 1, transform: "translateY(0)" }
        }
      },
      animation: {
        "fade-up": "fade-up 0.35s ease-out",
        "pulse-soft": "pulse-soft 1.4s ease-in-out infinite",
        "blink": "blink 1s step-end infinite",
        "spin-slow": "spin-slow 4s linear infinite",
        "glow": "glow 2.6s ease-in-out infinite",
        "float": "float 6s ease-in-out infinite",
        "anthropic-rise": "rise 0.6s cubic-bezier(0.16, 1, 0.3, 1) both"
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "grid-pattern":
          "linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px)"
      }
    }
  },
  plugins: [require("@tailwindcss/typography")]
};
