// ---------------------------------------------------------------------------
// Disaster Atmosphere Theme Engine
// ---------------------------------------------------------------------------
// Warm humanitarian-intelligence base palette. Disaster themes shift hue and
// tone to subconsciously signal the event type while keeping the UI readable.
// ---------------------------------------------------------------------------

export const DISASTER_THEMES = {
  Default: {
    bg:        '#131010',
    surface:   '#543A14',
    surface2:  '#0D0B0B',
    bubble:    'rgba(255,240,220,0.07)',
    chip:      '#543A14',
    primary:   '#F0BB78',
    accent:    '#FFF0DC',
    muted:     'rgba(255,240,220,0.50)',
    glow1:     'rgba(240, 187, 120, 0.12)',
    glow2:     'rgba(19,  16,  16, 0.06)',
    label:     '',
  },

  Flood: {
    bg:        '#08121E',
    surface:   '#0E2A48',
    surface2:  '#050C16',
    bubble:    'rgba(100,190,255,0.08)',
    chip:      '#0E2A48',
    primary:   '#5BAAD4',
    accent:    '#A8D8F0',
    muted:     'rgba(220,240,255,0.50)',
    glow1:     'rgba(70, 160, 230, 0.18)',
    glow2:     'rgba(20,  70, 140, 0.10)',
    label:     'Hydrological Assessment Active',
  },

  Fire: {
    bg:        '#1A0804',
    surface:   '#3E1A08',
    surface2:  '#110600',
    bubble:    'rgba(255,110,30,0.08)',
    chip:      '#3E1A08',
    primary:   '#E86030',
    accent:    '#FFAB70',
    muted:     'rgba(255,235,210,0.50)',
    glow1:     'rgba(255, 80, 20, 0.20)',
    glow2:     'rgba(180, 40,  0, 0.12)',
    label:     'Wildfire Response Context Active',
  },

  Earthquake: {
    bg:        '#131008',
    surface:   '#302A14',
    surface2:  '#0D0C06',
    bubble:    'rgba(210,190,130,0.08)',
    chip:      '#302A14',
    primary:   '#C8A856',
    accent:    '#E8D090',
    muted:     'rgba(250,240,210,0.50)',
    glow1:     'rgba(200, 170,  80, 0.16)',
    glow2:     'rgba(100,  80,  30, 0.10)',
    label:     'Structural Damage Context Active',
  },

  Cyclone: {
    bg:        '#080C14',
    surface:   '#14243A',
    surface2:  '#050810',
    bubble:    'rgba(130,170,210,0.08)',
    chip:      '#14243A',
    primary:   '#7098B8',
    accent:    '#B0C8D8',
    muted:     'rgba(220,235,250,0.50)',
    glow1:     'rgba(90, 140, 200, 0.16)',
    glow2:     'rgba(40,  80, 140, 0.10)',
    label:     'Storm Impact Context Active',
  },

  Landslide: {
    bg:        '#120D06',
    surface:   '#2E1E0A',
    surface2:  '#0D0A04',
    bubble:    'rgba(195,160,100,0.08)',
    chip:      '#2E1E0A',
    primary:   '#A88040',
    accent:    '#D0B070',
    muted:     'rgba(245,235,210,0.50)',
    glow1:     'rgba(170, 128,  60, 0.18)',
    glow2:     'rgba( 90,  60,  20, 0.10)',
    label:     'Geological Event Context Active',
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
