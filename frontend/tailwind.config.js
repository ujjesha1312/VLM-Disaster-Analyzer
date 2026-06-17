/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // ── Backgrounds ──────────────────────────────────────────────────────
        "bg-main":     "#000000",
        "bg-surface":  "#0F0F0F",
        "bg-elevated": "#161616",
        "bg-high":     "#1C1C1C",

        // ── Brand ────────────────────────────────────────────────────────────
        "primary":       "#853953",
        "primary-hover": "#612D53",
        "accent":        "#612D53",

        // ── Text ─────────────────────────────────────────────────────────────
        "text-base":  "#F3F4F4",
        "text-muted": "rgba(243,244,244,0.60)",
        "text-faint": "rgba(243,244,244,0.35)",

        // ── Borders ──────────────────────────────────────────────────────────
        "border-base":   "rgba(243,244,244,0.08)",
        "border-mid":    "rgba(243,244,244,0.14)",
        "border-strong": "rgba(243,244,244,0.24)",

        // ── Status (replaces Tailwind built-in emerald/red/amber) ────────────
        "success": "#34D399",
        "error":   "#F87171",
        "warning": "#FBBF24",

        // ── Severity chips ───────────────────────────────────────────────────
        "sev-critical-bg":   "#F3F4F4",
        "sev-critical-text": "#000000",
        "sev-high-bg":       "rgba(133,57,83,0.22)",
        "sev-high-text":     "#D4869F",
        "sev-moderate-bg":   "rgba(133,57,83,0.14)",
        "sev-moderate-text": "#C07090",
        "sev-low-bg":        "rgba(243,244,244,0.06)",
        "sev-low-text":      "rgba(243,244,244,0.55)",
      },
      spacing: {
        "margin-desktop": "48px",
        "margin-mobile":  "16px",
        "gutter":         "24px",
        "stack-sm":       "8px",
        "stack-md":       "16px",
        "stack-lg":       "32px",
      },
      fontFamily: {
        "display":     ["Hanken Grotesk", "sans-serif"],
        "headline-lg": ["Hanken Grotesk", "sans-serif"],
        "headline-md": ["Hanken Grotesk", "sans-serif"],
        "body-lg":     ["Inter", "sans-serif"],
        "body-md":     ["Inter", "sans-serif"],
        "body-sm":     ["Inter", "sans-serif"],
        "label-lg":    ["Inter", "sans-serif"],
        "label-sm":    ["Inter", "sans-serif"],
        "mono-sm":     ["JetBrains Mono", "monospace"],
      },
      fontSize: {
        "display":     ["40px", { lineHeight: "48px", letterSpacing: "-0.02em", fontWeight: "600" }],
        "headline-lg": ["32px", { lineHeight: "40px", letterSpacing: "-0.01em", fontWeight: "600" }],
        "headline-md": ["24px", { lineHeight: "32px", fontWeight: "500" }],
        "body-lg":     ["18px", { lineHeight: "28px", fontWeight: "400" }],
        "body-md":     ["16px", { lineHeight: "24px", fontWeight: "400" }],
        "body-sm":     ["14px", { lineHeight: "20px", fontWeight: "400" }],
        "label-lg":    ["14px", { lineHeight: "20px", letterSpacing: "0.05em", fontWeight: "600" }],
        "label-sm":    ["12px", { lineHeight: "16px", letterSpacing: "0.05em", fontWeight: "500" }],
        "mono-sm":     ["13px", { lineHeight: "18px", fontWeight: "400" }],
      },
      boxShadow: {
        // Base elevation
        "elev-sm":    "0 1px 4px rgba(0,0,0,0.70)",
        "elev-md":    "0 4px 20px rgba(0,0,0,0.75)",
        "elev-lg":    "0 8px 40px rgba(0,0,0,0.85)",
        // Surface chrome
        "card":       "0 1px 3px rgba(0,0,0,0.70), 0 0 0 1px rgba(243,244,244,0.06)",
        "nav":        "0 1px 0 rgba(243,244,244,0.06)",
        "input":      "0 0 0 1px rgba(243,244,244,0.10), 0 4px 24px rgba(0,0,0,0.80)",
        // Glow family
        "glow-sm":    "0 0 16px rgba(133,57,83,0.25)",
        "glow-md":    "0 0 20px rgba(133,57,83,0.35)",
        "glow-lg":    "0 0 30px rgba(133,57,83,0.50)",
        "glow-xl":    "0 0 40px rgba(133,57,83,0.22), 0 0 80px rgba(97,45,83,0.12)",
        "glow-primary": "0 0 30px rgba(133,57,83,0.25), 0 0 60px rgba(97,45,83,0.12)",
        "accent-sm":  "0 0 20px rgba(133,57,83,0.10)",
      },
    },
  },
  plugins: [],
}
