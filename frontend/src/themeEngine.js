// ---------------------------------------------------------------------------
// Disaster Atmosphere Theme Engine — Warm Light Edition
// ---------------------------------------------------------------------------
// Warm ivory base (#FFF8F0) with disaster-type accent shifts.
// All surfaces are light; glows are subtle warm tones.
// ---------------------------------------------------------------------------

export const DISASTER_THEMES = {
  Default: {
    bg:       '#FFF8F0',
    surface:  '#FFFFFF',
    surface2: '#FDF5EE',
    bubble:   'rgba(192,133,82,0.08)',
    chip:     'rgba(192,133,82,0.12)',
    primary:  '#C08552',
    accent:   '#5DADE2',
    muted:    'rgba(43,33,31,0.50)',
    glow1:    'rgba(192,133,82,0.15)',
    glow2:    'rgba(140,90,60,0.10)',
    label:    '',
  },

  Flood: {
    bg:       '#F0F7FF',
    surface:  '#FFFFFF',
    surface2: '#EAF4FF',
    bubble:   'rgba(41,128,185,0.08)',
    chip:     'rgba(41,128,185,0.12)',
    primary:  '#2980B9',
    accent:   '#5DADE2',
    muted:    'rgba(20,50,80,0.50)',
    glow1:    'rgba(41,128,185,0.15)',
    glow2:    'rgba(93,173,226,0.10)',
    label:    'Hydrological Assessment Active',
  },

  Fire: {
    bg:       '#FFF5F0',
    surface:  '#FFFFFF',
    surface2: '#FFF0E8',
    bubble:   'rgba(192,57,43,0.08)',
    chip:     'rgba(192,57,43,0.12)',
    primary:  '#C0392B',
    accent:   '#E74C3C',
    muted:    'rgba(80,20,15,0.50)',
    glow1:    'rgba(192,57,43,0.15)',
    glow2:    'rgba(231,76,60,0.10)',
    label:    'Wildfire Response Context Active',
  },

  Earthquake: {
    bg:       '#FDFAF5',
    surface:  '#FFFFFF',
    surface2: '#F9F3E8',
    bubble:   'rgba(192,133,82,0.10)',
    chip:     'rgba(192,133,82,0.14)',
    primary:  '#C08552',
    accent:   '#8C5A3C',
    muted:    'rgba(43,33,31,0.50)',
    glow1:    'rgba(192,133,82,0.20)',
    glow2:    'rgba(140,90,60,0.12)',
    label:    'Structural Damage Context Active',
  },

  Cyclone: {
    bg:       '#F0F7FF',
    surface:  '#FFFFFF',
    surface2: '#E8F3FC',
    bubble:   'rgba(52,152,219,0.08)',
    chip:     'rgba(52,152,219,0.12)',
    primary:  '#2980B9',
    accent:   '#5DADE2',
    muted:    'rgba(20,50,80,0.50)',
    glow1:    'rgba(52,152,219,0.15)',
    glow2:    'rgba(93,173,226,0.10)',
    label:    'Storm Impact Context Active',
  },

  Landslide: {
    bg:       '#FDFAF5',
    surface:  '#FFFFFF',
    surface2: '#F5EEE0',
    bubble:   'rgba(160,110,50,0.10)',
    chip:     'rgba(160,110,50,0.14)',
    primary:  '#A07032',
    accent:   '#8C5A3C',
    muted:    'rgba(43,33,31,0.50)',
    glow1:    'rgba(160,110,50,0.18)',
    glow2:    'rgba(140,90,60,0.10)',
    label:    'Geological Event Context Active',
  },
};

// ---------------------------------------------------------------------------
// Maps CLIP disaster_type labels → theme keys
// ---------------------------------------------------------------------------

const THEME_MAP = {
  'Flood':                 'Flood',
  'Water Disaster':        'Flood',
  'Wild Fire':             'Fire',
  'Urban Fire':            'Fire',
  'Earthquake':            'Earthquake',
  'Infrastructure Damage': 'Earthquake',
  'Human Damage':          'Earthquake',
  'Drought':               'Earthquake',
  'Landslide':             'Landslide',
  'Cyclone':               'Cyclone',
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function getTheme(eventType) {
  const key = THEME_MAP[eventType] ?? eventType;
  return DISASTER_THEMES[key] ?? DISASTER_THEMES.Default;
}

export function applyTheme(eventType) {
  const theme = eventType ? getTheme(eventType) : DISASTER_THEMES.Default;
  const r = document.documentElement;
  r.style.setProperty('--atm-bg',       theme.bg);
  r.style.setProperty('--atm-surface',  theme.surface);
  r.style.setProperty('--atm-surface2', theme.surface2);
  r.style.setProperty('--atm-bubble',   theme.bubble);
  r.style.setProperty('--atm-chip',     theme.chip);
  r.style.setProperty('--atm-primary',  theme.primary);
  r.style.setProperty('--atm-accent',   theme.accent);
  r.style.setProperty('--atm-muted',    theme.muted);
  r.style.setProperty('--atm-glow-1',   theme.glow1);
  r.style.setProperty('--atm-glow-2',   theme.glow2);
}

export function resetTheme() {
  applyTheme(null);
}
