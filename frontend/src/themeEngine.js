// ---------------------------------------------------------------------------
// Disaster Atmosphere Theme Engine — Light Palette Edition
// ---------------------------------------------------------------------------
// Light humanitarian-intelligence base palette. Disaster themes shift the
// primary/accent hue to signal event type while keeping all backgrounds
// light for readability.
// ---------------------------------------------------------------------------

export const DISASTER_THEMES = {
  Default: {
    bg:       '#F3F4F4',
    surface:  '#FFFFFF',
    surface2: '#ECEDEF',
    bubble:   'rgba(133,57,83,0.06)',
    chip:     'rgba(133,57,83,0.10)',
    primary:  '#853953',
    accent:   '#612D53',
    muted:    'rgba(44,44,44,0.45)',
    glow1:    'rgba(133,57,83,0.06)',
    glow2:    'rgba(97,45,83,0.04)',
    label:    '',
  },

  Flood: {
    bg:       '#F3F4F4',
    surface:  '#FFFFFF',
    surface2: '#ECEDEF',
    bubble:   'rgba(59,111,160,0.06)',
    chip:     'rgba(59,111,160,0.10)',
    primary:  '#3B6FA0',
    accent:   '#2A5280',
    muted:    'rgba(44,44,44,0.45)',
    glow1:    'rgba(59,111,160,0.08)',
    glow2:    'rgba(42,82,128,0.05)',
    label:    'Hydrological Assessment Active',
  },

  Fire: {
    bg:       '#F3F4F4',
    surface:  '#FFFFFF',
    surface2: '#ECEDEF',
    bubble:   'rgba(168,64,50,0.06)',
    chip:     'rgba(168,64,50,0.10)',
    primary:  '#A84032',
    accent:   '#7C2E25',
    muted:    'rgba(44,44,44,0.45)',
    glow1:    'rgba(168,64,50,0.08)',
    glow2:    'rgba(124,46,37,0.05)',
    label:    'Wildfire Response Context Active',
  },

  Earthquake: {
    bg:       '#F3F4F4',
    surface:  '#FFFFFF',
    surface2: '#ECEDEF',
    bubble:   'rgba(140,96,32,0.06)',
    chip:     'rgba(140,96,32,0.10)',
    primary:  '#8C6020',
    accent:   '#6A4818',
    muted:    'rgba(44,44,44,0.45)',
    glow1:    'rgba(140,96,32,0.08)',
    glow2:    'rgba(106,72,24,0.05)',
    label:    'Structural Damage Context Active',
  },

  Cyclone: {
    bg:       '#F3F4F4',
    surface:  '#FFFFFF',
    surface2: '#ECEDEF',
    bubble:   'rgba(74,106,138,0.06)',
    chip:     'rgba(74,106,138,0.10)',
    primary:  '#4A6A8A',
    accent:   '#355070',
    muted:    'rgba(44,44,44,0.45)',
    glow1:    'rgba(74,106,138,0.08)',
    glow2:    'rgba(53,80,112,0.05)',
    label:    'Storm Impact Context Active',
  },

  Landslide: {
    bg:       '#F3F4F4',
    surface:  '#FFFFFF',
    surface2: '#ECEDEF',
    bubble:   'rgba(122,88,40,0.06)',
    chip:     'rgba(122,88,40,0.10)',
    primary:  '#7A5828',
    accent:   '#5C4018',
    muted:    'rgba(44,44,44,0.45)',
    glow1:    'rgba(122,88,40,0.08)',
    glow2:    'rgba(92,64,24,0.05)',
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
