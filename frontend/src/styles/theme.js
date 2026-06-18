// ─────────────────────────────────────────────────────────────────────────────
// VLM Disaster Analyzer — Design System  v4
// Theme: Warm Light  ·  Disaster Intelligence Platform
//
// Single source of truth for all visual tokens.
// Mirrors tailwind.config.js — both must stay in sync.
// ─────────────────────────────────────────────────────────────────────────────

export const colors = {
  // Backgrounds
  bg:              "#FFF8F0",
  surface:         "#FFFFFF",
  surfaceElevated: "#FDF5EE",
  surfaceHigh:     "#FDF5EE",

  // Brand
  primary:   "#C08552",
  secondary: "#8C5A3C",

  // Text
  text:       "#2B211F",
  textMuted:  "#6B5A53",
  textFaint:  "#A08878",
  textGhost:  "#A08878",

  // Borders
  border:       "#E8DDD4",
  borderMid:    "#D4C4B8",
  borderStrong: "#D4C4B8",

  // Glows
  primaryGlow:   "rgba(192,133,82,0.15)",
  secondaryGlow: "rgba(140,90,60,0.10)",

  // Severity
  critical: { bg: "#FDECEA",  text: "#C0392B",  border: "#E74C3C" },
  high:     { bg: "#FEF5E7",  text: "#C08552",  border: "rgba(192,133,82,0.40)" },
  moderate: { bg: "#EBF5FB",  text: "#2980B9",  border: "rgba(93,173,226,0.40)" },
  low:      { bg: "#F4F6F7",  text: "#7F8C8D",  border: "#BDC3C7" },

  // Status
  success: "#27AE60",
  error:   "#E74C3C",
  warning: "#F39C12",
};

export const fonts = {
  display: "'Hanken Grotesk', sans-serif",
  body:    "'Inter', sans-serif",
  mono:    "'JetBrains Mono', monospace",
};

export const typography = {
  display:    { size: "40px", lineHeight: "48px", weight: 600, tracking: "-0.02em" },
  headlineLg: { size: "32px", lineHeight: "40px", weight: 600, tracking: "-0.01em" },
  headlineMd: { size: "24px", lineHeight: "32px", weight: 500 },
  bodyLg:     { size: "18px", lineHeight: "28px", weight: 400 },
  bodyMd:     { size: "16px", lineHeight: "24px", weight: 400 },
  bodySm:     { size: "14px", lineHeight: "20px", weight: 400 },
  labelLg:    { size: "14px", lineHeight: "20px", weight: 600, tracking: "0.05em" },
  labelSm:    { size: "12px", lineHeight: "16px", weight: 500, tracking: "0.05em" },
  monoSm:     { size: "13px", lineHeight: "18px", weight: 400 },
};

export const shadows = {
  sm:       "0 1px 4px rgba(0,0,0,0.08)",
  md:       "0 4px 20px rgba(0,0,0,0.10)",
  lg:       "0 8px 40px rgba(0,0,0,0.12)",
  nav:      "0 1px 0 rgba(43,33,31,0.06)",
  card:     "0 1px 3px rgba(0,0,0,0.08), 0 0 0 1px rgba(43,33,31,0.06)",
  input:    "0 0 0 1px rgba(43,33,31,0.10), 0 4px 24px rgba(0,0,0,0.06)",
  glowSm:   "0 0 16px rgba(192,133,82,0.20)",
  glowMd:   "0 0 20px rgba(192,133,82,0.30)",
  glowLg:   "0 0 30px rgba(192,133,82,0.40)",
  glowXl:   "0 0 40px rgba(192,133,82,0.20), 0 0 80px rgba(140,90,60,0.10)",
  accentSm: "0 0 20px rgba(192,133,82,0.10)",
};

export const borderRadius = {
  xs:   "6px",
  sm:   "8px",
  md:   "12px",
  lg:   "16px",
  xl:   "20px",
  full: "9999px",
};

export const transitions = {
  fast:   "120ms ease",
  normal: "220ms ease",
  slow:   "380ms ease",
  theme:  "550ms cubic-bezier(0.4, 0, 0.2, 1)",
};

export const spacing = {
  navHeight:    "64px",
  sidebarWidth: "240px",
  maxContent:   "800px",
  maxPage:      "1200px",
};

export const gradients = {
  ambient: "radial-gradient(ellipse at 40% 45%, rgba(192,133,82,0.06) 0%, #FFF8F0 55%, #FFF8F0 100%)",
  primary: "linear-gradient(135deg, #C08552 0%, #8C5A3C 100%)",
  streamH: (color) => `linear-gradient(to right, transparent 0%, ${color} 55%, #ffffff 100%)`,
  streamV: (color) => `linear-gradient(to bottom, transparent 0%, ${color} 55%, #ffffff 100%)`,
};

export default { colors, fonts, typography, shadows, borderRadius, transitions, spacing, gradients };
