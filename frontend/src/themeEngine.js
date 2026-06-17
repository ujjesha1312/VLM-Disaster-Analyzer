// ---------------------------------------------------------------------------
// Disaster Atmosphere Theme Engine — Obsidian Dark Edition
// ---------------------------------------------------------------------------
// True-black base (#000000) with disaster-type accent shifts.
// All glows are stronger than the light edition to read on dark backgrounds.
// ---------------------------------------------------------------------------

export const DISASTER_THEMES = {
  Default: {
    bg:       '#000000',
    surface:  '#0F0F0F',
    surface2: '#161616',
    bubble:   'rgba(133,57,83,0.10)',
    chip:     'rgba(133,57,83,0.14)',
    primary:  '#853953',
    accent:   '#612D53',
    muted:    'rgba(243,244,244,0.45)',
    glow1:    'rgba(133,57,83,0.22)',
    glow2:    'rgba(97,45,83,0.14)',
    label:    '',
  },

  Flood: {
    bg:       '#000000',
    surface:  '#060F14',
    surface2: '#0A141C',
    bubble:   'rgba(59,111,160,0.10)',
    chip:     'rgba(59,111,160,0.14)',
    primary:  '#4A7FA8',
    accent:   '#2A5280',
    muted:    'rgba(243,244,244,0.45)',
    glow1:    'rgba(59,111,160,0.22)',
    glow2:    'rgba(42,82,128,0.14)',
    label:    'Hydrological Assessment Active',
  },

  Fire: {
    bg:       '#000000',
    surface:  '#130800',
    surface2: '#1A0C00',
    bubble:   'rgba(168,64,50,0.10)',
    chip:     'rgba(168,64,50,0.14)',
    primary:  '#C05040',
    accent:   '#8C3828',
    muted:    'rgba(243,244,244,0.45)',
    glow1:    'rgba(168,64,50,0.25)',
    glow2:    'rgba(124,46,37,0.15)',
    label:    'Wildfire Response Context Active',
  },

  Earthquake: {
    bg:       '#000000',
    surface:  '#0E0900',
    surface2: '#150E00',
    bubble:   'rgba(140,96,32,0.10)',
    chip:     'rgba(140,96,32,0.14)',
    primary:  '#A07030',
    accent:   '#6A4818',
    muted:    'rgba(243,244,244,0.45)',
    glow1:    'rgba(140,96,32,0.22)',
    glow2:    'rgba(106,72,24,0.14)',
    label:    'Structural Damage Context Active',
  },

  Cyclone: {
    bg:       '#000000',
    surface:  '#050B10',
    surface2: '#08101A',
    bubble:   'rgba(74,106,138,0.10)',
    chip:     'rgba(74,106,138,0.14)',
    primary:  '#5A80A0',
    accent:   '#355070',
    muted:    'rgba(243,244,244,0.45)',
    glow1:    'rgba(74,106,138,0.22)',
    glow2:    'rgba(53,80,112,0.14)',
    label:    'Storm Impact Context Active',
  },

  Landslide: {
    bg:       '#000000',
    surface:  '#0C0800',
    surface2: '#130E00',
    bubble:   'rgba(122,88,40,0.10)',
    chip:     'rgba(122,88,40,0.14)',
    primary:  '#9A7040',
    accent:   '#5C4018',
    muted:    'rgba(243,244,244,0.45)',
    glow1:    'rgba(122,88,40,0.22)',
    glow2:    'rgba(92,64,24,0.14)',
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
