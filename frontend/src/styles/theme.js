// ─────────────────────────────────────────────────────────────────────────────
// VLM Disaster Analyzer — Design System  v2
// Theme: Obsidian Dark  ·  Enterprise Intelligence Platform
//
// Single source of truth for all visual tokens.
// Mirrors tailwind.config.js — both must stay in sync.
// ─────────────────────────────────────────────────────────────────────────────

// ── Color palette ─────────────────────────────────────────────────────────────
export const colors = {
  // Backgrounds
  bg:              "#000000",            // true black — main canvas
  surface:         "#0F0F0F",            // card / panel background
  surfaceElevated: "#161616",            // nested surfaces, inner panels
  surfaceHigh:     "#1C1C1C",            // highest elevation, popovers

  // Brand
  primary:   "#853953",                  // crimson rose — core accent
  secondary: "#612D53",                  // deep plum — hover / secondary

  // Text
  text:       "#F3F4F4",
  textMuted:  "rgba(243,244,244,0.60)",
  textFaint:  "rgba(243,244,244,0.35)",
  textGhost:  "rgba(243,244,244,0.18)",

  // Borders
  border:       "rgba(243,244,244,0.08)",
  borderMid:    "rgba(243,244,244,0.14)",
  borderStrong: "rgba(243,244,244,0.24)",

  // Glows
  primaryGlow:   "rgba(133,57,83,0.22)",
  secondaryGlow: "rgba(97,45,83,0.15)",

  // Severity
  critical: { bg: "#F3F4F4",                 text: "#000000",                border: "rgba(243,244,244,0.60)" },
  high:     { bg: "rgba(133,57,83,0.22)",     text: "#D4869F",                border: "rgba(133,57,83,0.45)"   },
  moderate: { bg: "rgba(133,57,83,0.14)",     text: "#C07090",                border: "rgba(133,57,83,0.28)"   },
  low:      { bg: "rgba(243,244,244,0.06)",   text: "rgba(243,244,244,0.55)", border: "rgba(243,244,244,0.14)" },

  // Status — use these instead of Tailwind built-in emerald/red/amber
  success: "#34D399",
  error:   "#F87171",
  warning: "#FBBF24",
};

// ── Typography ────────────────────────────────────────────────────────────────
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

// ── Shadows ───────────────────────────────────────────────────────────────────
// Named tokens map 1-to-1 with tailwind.config.js boxShadow extensions.
export const shadows = {
  sm:       "0 1px 4px rgba(0,0,0,0.70)",
  md:       "0 4px 20px rgba(0,0,0,0.75)",
  lg:       "0 8px 40px rgba(0,0,0,0.85)",
  nav:      "0 1px 0 rgba(243,244,244,0.06)",
  card:     "0 1px 3px rgba(0,0,0,0.70), 0 0 0 1px rgba(243,244,244,0.06)",
  input:    "0 0 0 1px rgba(243,244,244,0.10), 0 4px 24px rgba(0,0,0,0.80)",
  // Glow family — primary brand color at increasing intensities
  glowSm:   "0 0 16px rgba(133,57,83,0.25)",
  glowMd:   "0 0 20px rgba(133,57,83,0.35)",
  glowLg:   "0 0 30px rgba(133,57,83,0.50)",
  glowXl:   "0 0 40px rgba(133,57,83,0.22), 0 0 80px rgba(97,45,83,0.12)",
  accentSm: "0 0 20px rgba(133,57,83,0.10)",
};

// ── Border radius ─────────────────────────────────────────────────────────────
export const borderRadius = {
  xs:   "6px",
  sm:   "8px",
  md:   "12px",
  lg:   "16px",
  xl:   "20px",
  full: "9999px",
};

// ── Transitions ───────────────────────────────────────────────────────────────
export const transitions = {
  fast:   "120ms ease",
  normal: "220ms ease",
  slow:   "380ms ease",
  theme:  "550ms cubic-bezier(0.4, 0, 0.2, 1)",
};

// ── Spacing / layout ──────────────────────────────────────────────────────────
export const spacing = {
  navHeight:    "64px",
  sidebarWidth: "240px",
  maxContent:   "800px",
  maxPage:      "1200px",
};

// ── Gradient presets ──────────────────────────────────────────────────────────
export const gradients = {
  // Subtle brand radial used as page ambient glow
  ambient: "radial-gradient(ellipse at 40% 45%, rgba(133,57,83,0.06) 0%, #000000 55%, #000000 100%)",
  // Full brand linear for filled surfaces
  primary: "linear-gradient(135deg, #853953 0%, #612D53 100%)",
  // Horizontal / vertical stream variants (used in IntroAnimation)
  streamH: (color) => `linear-gradient(to right, transparent 0%, ${color} 55%, #ffffff 100%)`,
  streamV: (color) => `linear-gradient(to bottom, transparent 0%, ${color} 55%, #ffffff 100%)`,
};

export default { colors, fonts, typography, shadows, borderRadius, transitions, spacing, gradients };
