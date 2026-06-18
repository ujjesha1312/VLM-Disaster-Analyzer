import { useState, useRef, useCallback, useEffect, Fragment } from "react";
import { applyTheme, resetTheme, getTheme } from "./themeEngine";
import IntroAnimation from "./components/IntroAnimation";

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

const API_BASE_URL =import.meta.env.VITE_API_URL || "https://providing-earthy-phonebook.ngrok-free.dev";
const MODEL_TIMEOUT_MS = 180_000;
const CHAT_TIMEOUT_MS  =  60_000;
const MAX_FILE_SIZE_MB  = 10;
const MAX_FILE_SIZE     = MAX_FILE_SIZE_MB * 1024 * 1024;
const MAX_VIDEO_FILE_MB = 500;
const MAX_VIDEO_FILE    = MAX_VIDEO_FILE_MB * 1024 * 1024;

const UNIFIED_LOADING_MSGS = [
  "Examining the scene...",
  "Assessing disaster severity...",
  "Preparing response recommendations...",
  "Generating intelligence report...",
];
const VIDEO_EXTENSIONS  = new Set(["mp4", "avi", "mov", "mkv", "webm"]);

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

const MODELS = [
  { key: "clip",  name: "CLIP",     label: "CLIP-ViT-L/14",     endpoint: "/predict/clip",  gradient: "from-blue-500 to-cyan-500"    },
  { key: "blip2", name: "BLIP-2",   label: "BLIP-2 Caption",    endpoint: "/predict/blip2", gradient: "from-violet-500 to-purple-500" },
  { key: "llava", name: "LLaVA",    label: "LLaVA Reasoning",   endpoint: "/predict/llava", gradient: "from-emerald-500 to-teal-500"  },
  { key: "qwen",  name: "Qwen2-VL", label: "Qwen2-VL Analysis", endpoint: "/predict/qwen",  gradient: "from-orange-500 to-amber-500"  },
];

const MODEL_TIMELINE_LABEL = {
  clip:  (data) => data?.metrics?.disaster_type
    ? `Scene classified: ${data.metrics.disaster_type}`
    : "Initial scene classification complete",
  blip2: (_d) => "Scene description gathered",
  llava: (_d) => "Damage indicators and structural analysis complete",
  qwen:  (_d) => "Detailed field analysis complete",
};

// ---------------------------------------------------------------------------
// Disaster knowledge base
// ---------------------------------------------------------------------------

const IMPACTS = {
  Flood:      ["Road and transportation infrastructure damage", "Residential and commercial property flooding", "Contamination of water supply systems", "Population displacement and potential casualties"],
  Fire:       ["Widespread vegetation and forest destruction", "Air quality degradation from smoke and particulates", "Wildlife habitat and biodiversity loss", "Structural damage to nearby buildings"],
  Earthquake: ["Structural collapse of buildings and bridges", "Utility disruption — power, gas, water", "Secondary landslide and aftershock risk", "Population displacement and potential casualties"],
  Landslide:  ["Transportation corridor blockage", "Burial or structural damage to nearby structures", "Drainage and water system disruption", "Secondary flooding risk in downstream areas"],
  Cyclone:    ["Wind and storm surge damage to coastal infrastructure", "Widespread power and communications outages", "Low-lying area flooding", "Agricultural destruction across the affected region"],
};

const ACTIONS = {
  Flood:      ["Deploy water rescue teams and inflatable vessels", "Establish elevated evacuation routes and shelters", "Coordinate drainage authorities for water level management", "Activate emergency broadcast for affected communities"],
  Fire:       ["Mobilise aerial and ground fire suppression units", "Establish firebreaks to contain spread", "Order evacuation within the defined fire perimeter", "Deploy medical units for respiratory and burn treatment"],
  Earthquake: ["Activate urban search-and-rescue operations", "Deploy structural engineers for building safety assessment", "Establish emergency medical triage centres", "Issue aftershock warnings to the public"],
  Landslide:  ["Close all roads within the slide corridor", "Conduct geotechnical assessment for further slide risk", "Evacuate settlements in the debris flow path", "Deploy heavy machinery for access route clearance"],
  Cyclone:    ["Activate coastal evacuation protocols for vulnerable zones", "Pre-position emergency shelters and backup power units", "Suspend maritime and aviation operations", "Conduct rapid damage assessment after the storm passes"],
};

const INFRASTRUCTURE = {
  Flood:      ["Roads, bridges, and drainage networks", "Residential and commercial buildings in low-lying zones", "Water treatment and supply facilities", "Rail and surface transport corridors"],
  Fire:       ["Power lines and electrical grid infrastructure", "Rural road access and forest management corridors", "Agricultural structures and rural properties", "Telecommunications relay equipment"],
  Earthquake: ["Buildings, bridges, overpasses, and retaining walls", "Gas, water, and sewage pipelines", "Power and communications grid", "Rail and highway transport networks"],
  Landslide:  ["Mountain and hillside road networks", "Rail transport and freight corridors", "Hillside residential settlements", "Irrigation channels and water supply systems"],
  Cyclone:    ["Coastal buildings, piers, and port facilities", "Power and telecommunications lines", "Airport and maritime operations infrastructure", "Agricultural land and coastal assets"],
};

const HUMAN_IMPACTS = {
  Flood:      ["Displacement of residents from inundated zones", "Waterborne disease risk from contaminated water", "Drowning and injury risk in fast-moving currents", "Loss of access to healthcare, food, and shelter"],
  Fire:       ["Respiratory illness from prolonged smoke and particulate exposure", "Burn injuries for those within the fire perimeter", "Mass displacement from the active evacuation zone", "Long-term psychological impact on affected communities"],
  Earthquake: ["Crush injuries from structural collapse", "Survivors trapped in rubble requiring extraction", "Population displacement from condemned structures", "Disruption of emergency medical services"],
  Landslide:  ["Burial risk for residents in the debris flow path", "Community isolation from blocked access roads", "Injury from high-velocity debris impact", "Displacement and loss of agricultural livelihoods"],
  Cyclone:    ["Storm surge drowning risk in low-lying coastal zones", "Wind injury from airborne structural debris", "Mass displacement of coastal communities", "Loss of housing, food security, and essential services"],
};

const ENVIRONMENTAL_IMPACTS = {
  Flood:      ["Soil erosion and downstream sediment displacement", "Groundwater contamination from surface runoff", "Loss of riparian and wetland vegetation", "Disruption of aquatic and terrestrial ecosystems"],
  Fire:       ["Atmospheric carbon release and regional air quality degradation", "Long-term soil and watershed damage", "Loss of biodiversity and critical wildlife habitat", "Elevated erosion and flash flood risk post-fire"],
  Earthquake: ["Ground deformation and surface fault rupture", "Secondary landslide and rockfall hazard zones", "Contamination from ruptured utility infrastructure", "Disruption of natural drainage and groundwater flow"],
  Landslide:  ["Catchment sedimentation and potential river blockage", "Loss of topsoil and slope vegetation cover", "Secondary flooding from landslide debris dams", "Habitat destruction within the full slide corridor"],
  Cyclone:    ["Coastal erosion and mangrove system destruction", "Saltwater inundation of freshwater aquifers", "Marine ecosystem disruption from storm surge", "Widespread crop, vegetation, and topsoil loss"],
};

const SUGGESTED_BY_TYPE = {
  Flood: [
    "What resources are required for this flood response?",
    "What infrastructure is at immediate risk?",
    "What should first responders prioritise?",
    "What are the environmental contamination risks?",
    "What actions are needed in the next 24 hours?",
  ],
  Fire: [
    "What aerial and ground resources are required?",
    "What infrastructure is in the fire's path?",
    "What should incident commanders prioritise?",
    "What are the air quality and watershed risks?",
    "What actions are needed in the next 24 hours?",
  ],
  Earthquake: [
    "What search and rescue resources are required?",
    "Which critical infrastructure has been compromised?",
    "What should first responders prioritise?",
    "What are the secondary hazard risks?",
    "What actions are needed in the next 24 hours?",
  ],
  Landslide: [
    "What recovery and clearance resources are required?",
    "Which transport routes are blocked or at risk?",
    "What should response teams prioritise?",
    "What is the risk of secondary slides?",
    "What actions are needed in the next 24 hours?",
  ],
  Cyclone: [
    "What resources are required for this cyclone response?",
    "What coastal infrastructure faces the greatest risk?",
    "What should emergency coordinators prioritise?",
    "What are the storm surge and flooding risks?",
    "What actions are needed in the next 24 hours?",
  ],
};

const DEFAULT_SUGGESTED = [
  "What resources are required?",
  "What infrastructure is affected?",
  "What should first responders prioritise?",
  "What are the environmental risks?",
  "What actions are needed in the next 24 hours?",
];

function getSuggestedQuestions(eventType) {
  return SUGGESTED_BY_TYPE[eventType] ?? DEFAULT_SUGGESTED;
}

// ---------------------------------------------------------------------------
// Persistent memory (localStorage — all data stays in the browser)
// ---------------------------------------------------------------------------

const MEMORY_KEY      = "dia_memory";
const MAX_ASSESSMENTS = 10;
const DEFAULT_MEMORY  = { totalIncidents: 0, lastVisit: null, assessments: [] };

function loadMemory() {
  try {
    const raw = localStorage.getItem(MEMORY_KEY);
    return raw ? { ...DEFAULT_MEMORY, ...JSON.parse(raw) } : { ...DEFAULT_MEMORY };
  } catch { return { ...DEFAULT_MEMORY }; }
}

function saveMemory(m) {
  try { localStorage.setItem(MEMORY_KEY, JSON.stringify(m)); } catch { /* storage full — ignore */ }
}

function formatTimeAgo(iso) {
  const diff  = Date.now() - new Date(iso).getTime();
  const mins  = Math.floor(diff / 60_000);
  const hours = Math.floor(diff / 3_600_000);
  const days  = Math.floor(diff / 86_400_000);
  if (mins  <  2) return "just now";
  if (mins  < 60) return `${mins} min ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days  <  7) return `${days} day${days !== 1 ? "s" : ""} ago`;
  return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric" });
}

async function generateThumb(file) {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const S = 80;
      const canvas = document.createElement("canvas");
      canvas.width  = S;
      canvas.height = S;
      const c = canvas.getContext("2d");
      const scale = Math.min(S / img.width, S / img.height);
      const w = img.width * scale, h = img.height * scale;
      c.fillStyle = "#FFFFFF";
      c.fillRect(0, 0, S, S);
      c.drawImage(img, (S - w) / 2, (S - h) / 2, w, h);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/jpeg", 0.65));
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
    img.src = url;
  });
}

const INITIAL_MEMORY = loadMemory();

// ---------------------------------------------------------------------------
// Name personalization
// ---------------------------------------------------------------------------

const NAME_KEY = "dia_username";

function getTimeGreeting() {
  const h = new Date().getHours();
  return h >= 5  && h < 12 ? "Good morning" :
         h >= 12 && h < 17 ? "Good afternoon" :
         h >= 17 && h < 22 ? "Good evening" : "Hello";
}

const IDLE_STATUS_MESSAGES = [
  "Intelligence systems online",
  "Ready for analysis",
  "Awaiting imagery",
  "Situational awareness active",
];

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function callModel(endpoint, file) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), MODEL_TIMEOUT_MS);
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      body:   fd,
      signal:  controller.signal,
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    return res.json();
  } catch (err) {
    if (err.name === "AbortError") throw new Error("Timed out — model took too long");
    if (err.message.toLowerCase().includes("failed to fetch")) throw new Error("Network unreachable");
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function callChat(question, context, history) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE_URL}/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        context: {
          eventType:     context.eventType,
          confidence:    context.confidence,
          caption:       context.caption,
          reasoning:     context.reasoning,
          sceneAnalysis: context.sceneAnalysis,
          severity:      context.severity,
        },
        history: history.slice(-8).map((m) => ({ role: m.role, content: m.content })),
      }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    return data.response;
  } catch (err) {
    if (err.name === "AbortError") throw new Error("Request timed out");
    if (err.message.toLowerCase().includes("failed to fetch")) throw new Error("Network unreachable");
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function callVideoAnalysis(file) {
  const fd = new FormData();
  fd.append("video", file);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120_000);
  try {
    const res = await fetch(`${API_BASE_URL}/predict/video/analyze`, {
      method: "POST",
      body:   fd,
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Video analysis failed (HTTP ${res.status})`);
    }
    return res.json();
  } catch (err) {
    if (err.name === "AbortError") throw new Error("Request timed out after 2 minutes");
    if (err.message.toLowerCase().includes("failed to fetch")) throw new Error("Network unreachable");
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// DisasterContext builder
// ---------------------------------------------------------------------------

function buildDisasterContext(outputs) {
  const eventType     = outputs.clip?.metrics?.disaster_type    ?? "Unknown Event";
  const confidence    = outputs.clip?.metrics?.confidence_score ?? 0;
  const severity      = outputs.clip?.metrics?.confidence_level ?? "Low";
  const top3          = outputs.clip?.metrics?.top_3_predictions ?? [];
  const caption       = outputs.blip2?.metrics?.scene_description ?? "";
  const keywords      = outputs.blip2?.metrics?.keywords          ?? [];
  const reasoning     = outputs.llava?.metrics?.raw_assessment    ?? "";
  const sceneAnalysis = outputs.qwen?.metrics?.raw_analysis       ?? "";

  const llavaMetrics = {
    disaster_type:         outputs.llava?.metrics?.disaster_type          ?? "",
    severity:              outputs.llava?.metrics?.severity               ?? "",
    affected_areas:        outputs.llava?.metrics?.affected_areas         ?? "",
    infrastructure_damage: outputs.llava?.metrics?.infrastructure_damage  ?? "",
    recommended_action:    outputs.llava?.metrics?.recommended_action     ?? "",
  };

  const qwenMetrics = {
    disaster_type:         outputs.qwen?.metrics?.disaster_type          ?? "",
    severity:              outputs.qwen?.metrics?.severity               ?? "",
    affected_population:   outputs.qwen?.metrics?.affected_population    ?? "",
    infrastructure_status: outputs.qwen?.metrics?.infrastructure_status  ?? "",
    environmental_impact:  outputs.qwen?.metrics?.environmental_impact   ?? "",
  };

  const impacts             = IMPACTS[eventType]             ?? ["Environmental and infrastructure damage", "Potential civilian impact", "Service disruption"];
  const actions             = ACTIONS[eventType]             ?? ["Deploy emergency response teams", "Establish incident command", "Prioritise civilian evacuation"];
  const infrastructure      = INFRASTRUCTURE[eventType]      ?? ["Critical infrastructure under assessment", "Utility systems at risk"];
  const humanImpact         = HUMAN_IMPACTS[eventType]       ?? ["Civilian safety under assessment", "Evacuation and medical staging recommended"];
  const environmentalImpact = ENVIRONMENTAL_IMPACTS[eventType] ?? ["Environmental assessment in progress", "Contamination risk under evaluation"];

  return { eventType, confidence, severity, top3, caption, keywords, reasoning, sceneAnalysis, llavaMetrics, qwenMetrics, impacts, actions, infrastructure, humanImpact, environmentalImpact };
}

// ---------------------------------------------------------------------------
// buildClipReport — fast operational assessment from CLIP only
// Maps CLIP output + existing knowledge-base tables into the unified result
// schema expected by UnifiedReportPanel. No Qwen required.
// ---------------------------------------------------------------------------

const _CLIP_KB_KEY = {
  "Earthquake":           "Earthquake",
  "Wild Fire":            "Fire",
  "Urban Fire":           "Fire",
  "Water Disaster":       "Flood",
  "Landslide":            "Landslide",
  "Infrastructure Damage":"Earthquake",
};

function buildClipReport(clipData, elapsedMs) {
  const metrics = clipData?.metrics ?? {};
  const type    = metrics.disaster_type    ?? "Unknown";
  const conf    = metrics.confidence_score ?? 0;
  const level   = metrics.confidence_level ?? "Low";
  const top3    = metrics.top_3_predictions ?? [];

  const key = _CLIP_KB_KEY[type];

  const visibleDamage = top3.length
    ? top3.map((p) => `${p.label} (${p.score}%)`).join("  ·  ")
    : `Visual indicators consistent with ${type.toLowerCase()} conditions`;

  return {
    category:                  type,
    classification_confidence: Math.round(conf * 100) / 100,
    severity:                  level,
    visible_damage:            visibleDamage,
    affected_area:   (INFRASTRUCTURE[key]       ?? ["Area-specific assessment pending"]).slice(0, 2).join(". ") + ".",
    environmental_impact: (ENVIRONMENTAL_IMPACTS[key] ?? ["Environmental assessment pending"]).slice(0, 2).join(". ") + ".",
    recommendations: (ACTIONS[key]              ?? ["Deploy emergency assessment teams", "Establish incident command", "Activate emergency protocols"]).slice(0, 3).join(". ") + ".",
    active_models:   ["Disaster Intelligence Engine"],
    processing_time_ms: elapsedMs,
  };
}

function buildDescription(ctx) {
  const { eventType, severity, caption, reasoning } = ctx;

  const observation = caption
    ? caption.charAt(0).toUpperCase() + caption.slice(1).replace(/\.$/, "")
    : reasoning
      ? reasoning.split(/[.!?]/)[0].trim()
      : `visual signatures consistent with ${eventType.toLowerCase()} conditions`;

  const urgencyLine =
    severity === "Critical"
      ? "This incident requires immediate coordinated emergency response — all operational windows are active."
      : severity === "High"
      ? "Prompt deployment of response resources is warranted. Pre-position teams and equipment now."
      : severity === "Moderate"
      ? "Active monitoring and precautionary resource staging are recommended."
      : "Situation is under assessment. Standby protocols apply.";

  return (
    `I've finished reviewing the scene you shared. I'm reading this as a ${severity.toLowerCase()}-severity ${eventType.toLowerCase()} event. ` +
    `${observation}. ` +
    `${urgencyLine} ` +
    `I've outlined key risks, recommended actions, and impact areas below — ask me anything about this incident.`
  );
}

// ---------------------------------------------------------------------------
// Client-side fallback for /chat
// ---------------------------------------------------------------------------

function buildFallbackResponse(question, ctx) {
  const q       = question.toLowerCase();
  const event   = ctx?.eventType      ?? "the incident";
  const conf    = ctx?.confidence     ?? 0;
  const sev     = ctx?.severity       ?? "unknown";
  const reasoning = ctx?.reasoning    ?? "";
  const scene     = ctx?.sceneAnalysis ?? "";

  if (/sever|how bad|intensity|danger/i.test(q))
    return `Looking at the ${event.toLowerCase()} scene you uploaded, I assess this as ${sev.toLowerCase()} severity. ${reasoning.split(".")[0] || "The visible damage indicators are consistent with substantial impact conditions"}.`;

  if (/resource|equip|personnel|deploy/i.test(q)) {
    const list = (ACTIONS[event] ?? []).slice(0, 4).map((a) => `• ${a}`).join("\n");
    return `Based on what I can see in this ${event.toLowerCase()} scene, here are my deployment priorities:\n\n${list}`;
  }

  if (/action|response|protocol|24 hour|next step|priorit|what should/i.test(q)) {
    const list = (ACTIONS[event] ?? []).slice(0, 4).map((a) => `• ${a}`).join("\n");
    return `Looking at this ${event.toLowerCase()} scene, here are my recommended actions:\n\n${list}`;
  }

  if (/people|casualt|human|injur|civilian|evacuati/i.test(q))
    return `From the image, civilian exposure in this ${event.toLowerCase()} scene appears to be ${conf > 80 ? "HIGH" : "MODERATE"} risk. Immediate evacuation and medical staging should be initiated.`;

  if (/infrastructure|structure|building|road|bridge|damage/i.test(q)) {
    const list = (INFRASTRUCTURE[event] ?? IMPACTS[event] ?? []).slice(0, 4).map((i) => `• ${i}`).join("\n");
    return `Looking at the visible infrastructure in this ${event.toLowerCase()} scene, the following is at risk:\n\n${list}`;
  }

  if (/environment|ecology|water|soil|vegetation|contamin/i.test(q)) {
    const list = (ENVIRONMENTAL_IMPACTS[event] ?? []).slice(0, 4).map((i) => `• ${i}`).join("\n");
    return `From what I can see, here's the environmental impact from this ${event.toLowerCase()} scene:\n\n${list}`;
  }

  const context = scene.split(".")[0] || reasoning.split(".")[0] || "";
  return `Based on the ${event.toLowerCase()} scene you uploaded (${sev.toLowerCase()} severity)${context ? `: ${context}.` : "."} What specific aspect of the situation would you like to understand better?`;
}

// ---------------------------------------------------------------------------
// Model consensus calculation
// ---------------------------------------------------------------------------

function computeConsensus(modelOutputs, eventType) {
  const synonyms = {
    Flood:      ["flood", "water", "inundated", "submerged", "overflow", "flooded"],
    Fire:       ["fire", "wildfire", "burning", "blaze", "smoke", "flames"],
    Earthquake: ["earthquake", "seismic", "collapse", "collapsed", "rubble", "debris", "tremor"],
    Landslide:  ["landslide", "landslip", "debris", "slope", "mudslide", "mud"],
    Cyclone:    ["cyclone", "hurricane", "typhoon", "storm", "wind"],
  };

  const terms = synonyms[eventType] ?? [eventType?.toLowerCase() ?? ""];
  const matches = (text) => !!text && terms.some((t) => text.toLowerCase().includes(t));

  const checks = {
    clip:  !!modelOutputs.clip  && !modelOutputs.clip?.error,
    blip2: matches(modelOutputs.blip2?.metrics?.scene_description),
    llava: matches(modelOutputs.llava?.metrics?.raw_assessment),
    qwen:  matches(modelOutputs.qwen?.metrics?.raw_analysis),
  };

  const count = Object.values(checks).filter(Boolean).length;
  const level = count === 4 ? "Strong" : count === 3 ? "Moderate" : count >= 2 ? "Mixed" : "Low";

  return { level, count, total: 4, checks };
}

// ---------------------------------------------------------------------------
// Severity chip styling
// ---------------------------------------------------------------------------

function severityChipClass(severity) {
  switch (severity) {
    case "Critical": return "bg-[#FDECEA] text-[#C0392B] border-[#E74C3C]/60";
    case "High":     return "bg-[#FEF5E7] text-[#C08552] border-[#C08552]/40";
    case "Moderate": return "bg-[#EBF5FB] text-[#2980B9] border-[#5DADE2]/40";
    default:         return "bg-[#F4F6F7] text-[#7F8C8D] border-[#BDC3C7]";
  }
}

// ---------------------------------------------------------------------------
// EvidencePanel — collapsible model output details inside the briefing
// ---------------------------------------------------------------------------

function LevelBadge({ level }) {
  const cls =
    level === "Critical" ? "bg-[#FDECEA] text-[#C0392B] border-[#E74C3C]/60" :
    level === "High"     ? "bg-[#FEF5E7] text-[#C08552] border-[#C08552]/40" :
    level === "Moderate" ? "bg-[#EBF5FB] text-[#2980B9] border-[#5DADE2]/40" :
                           "bg-[#F4F6F7] text-[#7F8C8D] border-[#BDC3C7]";
  return (
    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded border ${cls}`}>{level}</span>
  );
}

function KVRow({ label, value }) {
  if (!value) return null;
  return (
    <div className="flex gap-2">
      <span className="text-[#6B5A53] text-xs shrink-0 w-36">{label}</span>
      <span className="text-[#2B211F] text-xs">{value}</span>
    </div>
  );
}

function EvidencePanel({ modelOutputs }) {
  const [open,     setOpen]     = useState(false);
  const [llavOpen, setLlavOpen] = useState(false);
  const [qwenOpen, setQwenOpen] = useState(false);

  const clipM  = modelOutputs.clip?.metrics  ?? {};
  const blip2M = modelOutputs.blip2?.metrics ?? {};
  const llavaM = modelOutputs.llava?.metrics ?? {};
  const qwenM  = modelOutputs.qwen?.metrics  ?? {};

  return (
    <div className="mt-6 border border-[#E8DDD4] rounded-xl overflow-hidden">
      <button
        className="w-full flex items-center justify-between p-4 bg-[#FDF5EE] atm-bubble hover:bg-[#E8DDD4]/50 transition-colors"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-[#2B211F] text-[18px]">database</span>
          <span className="text-xs font-semibold uppercase tracking-widest text-[#2B211F]">
            How I reached this conclusion
          </span>
        </div>
        <span
          className="material-symbols-outlined text-[#2B211F] transition-transform duration-200"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        >
          expand_more
        </span>
      </button>

      {open && (
        <div className="bg-white atm-surface2 p-4 space-y-5 divide-y divide-[#E8DDD4] max-h-[520px] overflow-y-auto">

          {/* ── CLIP ─────────────────────────────────────────────────────────── */}
          <div className="pt-2 space-y-3">
            <p className="text-[#2B211F] text-xs uppercase font-semibold tracking-wider" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
              CLIP-ViT-L/14
            </p>
            {modelOutputs.clip?.error ? (
              <p className="text-[#E74C3C] text-sm italic">Failed — {modelOutputs.clip.error}</p>
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-[#2B211F] text-base font-semibold">{clipM.disaster_type ?? "—"}</span>
                  {clipM.confidence_level && <LevelBadge level={clipM.confidence_level} />}
                </div>
                {clipM.top_3_predictions?.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-[10px] text-[#6B5A53] uppercase tracking-wider">Top predictions</p>
                    {clipM.top_3_predictions.map((p) => (
                      <div key={p.label} className="flex items-center gap-2">
                        <span className="text-[#2B211F] text-xs w-36 shrink-0 truncate">{p.label}</span>
                        <div className="flex-1 h-1.5 bg-[#E8DDD4] rounded-full overflow-hidden">
                          <div className="h-full bg-[#C08552]/70 rounded-full" style={{ width: `${Math.min(p.score, 100)}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>

          {/* ── BLIP-2 ───────────────────────────────────────────────────────── */}
          <div className="pt-4 space-y-3">
            <p className="text-[#2B211F] text-xs uppercase font-semibold tracking-wider" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
              BLIP-2 Caption
            </p>
            {modelOutputs.blip2?.error ? (
              <p className="text-[#E74C3C] text-sm italic">Failed — {modelOutputs.blip2.error}</p>
            ) : (
              <>
                <p className="text-[#2B211F] text-sm italic">
                  {blip2M.scene_description ? `"${blip2M.scene_description}"` : "—"}
                </p>
                {blip2M.keywords?.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {blip2M.keywords.map((kw) => (
                      <span key={kw} className="px-2 py-0.5 rounded-full bg-[#FDF5EE] border border-[#E8DDD4] text-[#2B211F] text-[11px]">
                        {kw}
                      </span>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>

          {/* ── LLaVA ────────────────────────────────────────────────────────── */}
          <div className="pt-4 space-y-3">
            <p className="text-[#2B211F] text-xs uppercase font-semibold tracking-wider" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
              LLaVA Reasoning
            </p>
            {modelOutputs.llava?.error ? (
              <p className="text-[#E74C3C] text-sm italic">Failed — {modelOutputs.llava.error}</p>
            ) : (
              <>
                <KVRow label="Disaster Type"         value={llavaM.disaster_type} />
                <KVRow label="Severity"              value={llavaM.severity} />
                <KVRow label="Affected Area"         value={llavaM.affected_areas} />
                <KVRow label="Infrastructure Damage" value={llavaM.infrastructure_damage} />
                <KVRow label="Recommended Action"    value={llavaM.recommended_action} />
                {llavaM.raw_assessment && (
                  <div>
                    <button
                      onClick={() => setLlavOpen((v) => !v)}
                      className="text-[10px] text-[#6B5A53] hover:text-[#2B211F] flex items-center gap-1 uppercase tracking-wider"
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "12px" }}>
                        {llavOpen ? "expand_less" : "expand_more"}
                      </span>
                      Full Assessment
                    </button>
                    {llavOpen && (
                      <p className="mt-2 text-[#6B5A53] text-xs leading-relaxed border-l-2 border-[#E8DDD4] pl-3">
                        {llavaM.raw_assessment}
                      </p>
                    )}
                  </div>
                )}
              </>
            )}
          </div>

          {/* ── Qwen ─────────────────────────────────────────────────────────── */}
          <div className="pt-4 space-y-3">
            <p className="text-[#2B211F] text-xs uppercase font-semibold tracking-wider" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
              Qwen2-VL Analysis
            </p>
            {modelOutputs.qwen?.error ? (
              <p className="text-[#E74C3C] text-sm italic">Failed — {modelOutputs.qwen.error}</p>
            ) : (
              <>
                <KVRow label="Disaster Type"         value={qwenM.disaster_type} />
                <KVRow label="Severity"              value={qwenM.severity} />
                <KVRow label="Affected Population"   value={qwenM.affected_population} />
                <KVRow label="Infrastructure Status" value={qwenM.infrastructure_status} />
                <KVRow label="Environmental Impact"  value={qwenM.environmental_impact} />
                {qwenM.raw_analysis && (
                  <div>
                    <button
                      onClick={() => setQwenOpen((v) => !v)}
                      className="text-[10px] text-[#6B5A53] hover:text-[#2B211F] flex items-center gap-1 uppercase tracking-wider"
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "12px" }}>
                        {qwenOpen ? "expand_less" : "expand_more"}
                      </span>
                      Full Analysis
                    </button>
                    {qwenOpen && (
                      <p className="mt-2 text-[#6B5A53] text-xs leading-relaxed border-l-2 border-[#E8DDD4] pl-3">
                        {qwenM.raw_analysis}
                      </p>
                    )}
                  </div>
                )}
              </>
            )}
          </div>

        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ReportCard — reusable cell inside UnifiedReportPanel
// ---------------------------------------------------------------------------

function ReportCard({ title, icon, content, accent = false }) {
  return (
    <div className={`p-4 rounded-xl border ${
      accent
        ? "bg-[#C08552]/10 border-[#C08552]/30 shadow-accent-sm"
        : "bg-white border-[#E8DDD4]"
    }`}>
      <div className="flex items-center gap-2 mb-3">
        <span
          className="material-symbols-outlined text-[#2B211F]"
          style={{ fontSize: "15px", fontVariationSettings: "'FILL' 1" }}
        >
          {icon}
        </span>
        <h4 className="text-[#2B211F] text-xs font-semibold uppercase tracking-widest">{title}</h4>
      </div>
      <p className="text-[#6B5A53] text-sm leading-relaxed">{content || "Not assessed."}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// UnifiedReportPanel — rendered as a chat message for unified-briefing type
// ---------------------------------------------------------------------------

function UnifiedReportPanel({ msg, unifiedResult, disasterCtx, onSuggestedQuery, chatLength }) {
  const d = unifiedResult;
  if (!d) return null;

  const conf    = d.classification_confidence ?? 0;
  const models  = d.active_models ?? ["CLIP", "Qwen2-VL"];
  const procMs  = d.processing_time_ms ?? null;

  return (
    <article key={msg.id} className="flex gap-4 items-start message-enter">
      <div className="w-8 h-8 rounded-full bg-[#C08552]/15 flex items-center justify-center shrink-0 mt-1">
        <span
          className="material-symbols-outlined text-[#2B211F]"
          style={{ fontSize: "18px", fontVariationSettings: "'FILL' 1" }}
        >
          analytics
        </span>
      </div>

      <div className="flex-1 space-y-5 pt-1">

        {/* ── Header ── */}
        <div className="space-y-3">
          <div className="flex flex-wrap items-start gap-x-3 gap-y-2">
            <h2
              className="text-[#2B211F] text-3xl font-bold leading-tight"
              style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}
            >
              {d.category}
            </h2>
            <div className="flex items-center gap-2 pt-1.5">
              <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${severityChipClass(d.severity)}`}>
                {d.severity}
              </span>
              <span className="flex items-center gap-1 text-[#27AE60] text-xs font-semibold">
                <span
                  className="material-symbols-outlined"
                  style={{ fontSize: "13px", fontVariationSettings: "'FILL' 1" }}
                >
                  check_circle
                </span>
                Report Ready
              </span>
            </div>
          </div>

          {/* Professional subtitle */}
          <p className="text-[#6B5A53] text-sm">
            Assessment generated using the Disaster Intelligence Engine.
          </p>

          {/* Active model chips */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[#A08878] text-xs">Models used:</span>
            {models.map((m) => (
              <span
                key={m}
                className="px-2 py-0.5 rounded bg-[#C08552]/12 border border-[#C08552]/25 text-[#C08552] text-[10px] font-semibold"
              >
                {m}
              </span>
            ))}
          </div>
        </div>

        {/* ── Report grid ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ReportCard
            title="Visible Damage"
            icon="warning"
            content={d.visible_damage}
          />
          <ReportCard
            title="Affected Area"
            icon="location_on"
            content={d.affected_area}
          />
          <ReportCard
            title="Environmental Impact"
            icon="eco"
            content={d.environmental_impact}
          />
          <ReportCard
            title="Recommendations"
            icon="emergency"
            content={d.recommendations}
            accent
          />
        </div>

        {/* ── Similar historical events (populated when FAISS index is built) ── */}
        <SimilarEventsCard events={d.similar_events} />

        {/* ── Suggested queries (only when conversation is fresh) ── */}
        {chatLength <= 2 && disasterCtx && (
          <div className="pt-1 space-y-2">
            <p className="text-[10px] text-[#6B5A53] uppercase tracking-widest font-semibold">
              Suggested queries
            </p>
            <div className="flex flex-wrap gap-2">
              {getSuggestedQuestions(d.category).map((q) => (
                <button
                  key={q}
                  onClick={() => onSuggestedQuery(q)}
                  className="atm-chip hover:bg-[#C08552]/20 transition-all text-[#2B211F] hover:text-[#2B211F] px-4 py-2 rounded-full text-xs border border-[#E8DDD4] hover:border-[#C08552]/40"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </article>
  );
}

// ---------------------------------------------------------------------------
// SimilarEventsCard — rendered inside UnifiedReportPanel when retrieval data exists
// ---------------------------------------------------------------------------

function SimilarEventsCard({ events }) {
  if (!events || events.length === 0) return null;

  const barColor = (sim) => {
    if (sim >= 80) return "bg-[#27AE60]";
    if (sim >= 65) return "bg-[#C08552]";
    return "bg-[#D4C4B8]";
  };

  const simLabel = (sim) => {
    if (sim >= 80) return "text-[#27AE60]";
    if (sim >= 65) return "text-[#C08552]";
    return "text-[#A08878]";
  };

  return (
    <div className="rounded-xl border border-[#E8DDD4] bg-white p-4 space-y-3">
      <div className="flex items-center gap-2">
        <span
          className="material-symbols-outlined text-[#C08552]"
          style={{ fontSize: "17px", fontVariationSettings: "'FILL' 1" }}
        >
          history
        </span>
        <p className="text-[10px] text-[#6B5A53] uppercase tracking-widest font-semibold">
          Similar Historical Events
        </p>
      </div>

      <div className="space-y-2">
        {events.map((ev, i) => (
          <div
            key={i}
            className="flex items-start gap-3 p-3 rounded-lg bg-[#FDF5EE] border border-[#E8DDD4] hover:border-[#D4C4B8] transition-colors"
          >
            {/* Rank badge */}
            <span className="shrink-0 w-5 h-5 rounded-full bg-[#C08552]/15 border border-[#C08552]/30 flex items-center justify-center text-[#C08552] text-[9px] font-bold mt-0.5">
              {i + 1}
            </span>

            {/* Event info */}
            <div className="flex-1 min-w-0 space-y-1">
              <div className="flex items-baseline gap-2 flex-wrap">
                <span className="text-[#2B211F] text-sm font-semibold leading-tight">
                  {ev.event}
                </span>
                <span className="text-[#8C7B73] text-xs">{ev.year}</span>
              </div>
              <p className="text-[#6B5A53] text-[11px] leading-snug">
                {ev.location}
              </p>
              {ev.description && (
                <p className="text-[#A08878] text-[10px] leading-snug line-clamp-2">
                  {ev.description}
                </p>
              )}

              {/* Similarity bar */}
              <div className="flex items-center gap-2 pt-1">
                <div className="flex-1 h-1 rounded-full bg-[#E8DDD4] overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${barColor(ev.similarity)}`}
                    style={{ width: `${Math.min(ev.similarity, 100)}%` }}
                  />
                </div>
                <span className={`text-[10px] font-bold tabular-nums shrink-0 ${simLabel(ev.similarity)}`}>
                  {ev.similarity.toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <p className="text-[9px] text-[#A08878] pt-1">
        Similarity computed via CLIP cosine distance against {events.length} indexed events.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// VideoAssessmentPanel — rendered as a chat message for video-briefing type.
// When unifiedResult is provided the full disaster report is shown alongside
// stream metadata. When absent, falls back to the metadata-only view.
// ---------------------------------------------------------------------------

function VideoAssessmentPanel({ msg, videoAnalysis, unifiedResult }) {
  // Support both response schemas: new (video_metadata) and legacy (file_info)
  const fi    = videoAnalysis?.video_metadata ?? videoAnalysis?.file_info;
  const an    = videoAnalysis?.analysis;
  const thumb = videoAnalysis?.thumbnail_b64;

  function fmtDuration(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m ? `${m}m ${sec.toString().padStart(2, "0")}s` : `${sec}s`;
  }

  const SEV_COLOR = {
    Critical: "text-[#C0392B] border-[#E74C3C]/60 bg-[#FDECEA]",
    High:     "text-[#C08552] border-[#C08552]/40 bg-[#FEF5E7]",
    Moderate: "text-[#2980B9] border-[#5DADE2]/40 bg-[#EBF5FB]",
    Low:      "text-[#7F8C8D] border-[#BDC3C7] bg-[#F4F6F7]",
  };
  const sevColor = SEV_COLOR[unifiedResult?.severity] ?? "text-[#7F8C8D] border-[#BDC3C7] bg-[#F4F6F7]";

  const StreamCard = () => {
    if (!fi) return null;
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-xl overflow-hidden border border-[#8C5A3C]/30 bg-[#FDF5EE]">
          {thumb ? (
            <img src={thumb} alt="Video thumbnail" className="w-full aspect-video object-cover" />
          ) : (
            <div className="w-full aspect-video flex items-center justify-center">
              <span className="material-symbols-outlined text-[#8C5A3C] text-4xl">movie</span>
            </div>
          )}
          <div className="px-3 py-2 bg-[#FFF8F0]">
            <p className="text-[10px] text-[#A08878] uppercase tracking-wider font-semibold">
              Thumbnail · 1 s seek
            </p>
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-[#E8DDD4] space-y-2">
          <h4 className="text-[#2B211F] text-xs font-semibold uppercase tracking-widest mb-3">
            Stream Properties
          </h4>
          {[
            ["Duration",     fi.duration_s ? fmtDuration(fi.duration_s) : "—"],
            ["Frame rate",   fi.fps ? `${fi.fps.toFixed(1)} fps` : "—"],
            ["Resolution",   fi.resolution ?? "—"],
            ["Codec",        fi.codec ? fi.codec.toUpperCase() : "—"],
            ["Container",    fi.format ?? "—"],
            ["File size",    `${(fi.size_mb ?? 0).toFixed(1)} MB`],
            ["Total frames", fi.total_frames ? fi.total_frames.toLocaleString() : "—"],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between items-center border-b border-[#E8DDD4] pb-1.5 last:border-none last:pb-0">
              <span className="text-[#6B5A53] text-xs">{k}</span>
              <span className="text-[#2B211F] text-xs font-semibold"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}>{v}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ── Full disaster report ──────────────────────────────────────────────────
  if (unifiedResult) {
    const d             = unifiedResult;
    const votes         = videoAnalysis?.frame_votes        ?? {};
    const framesAnalyzed = videoAnalysis?.frames_analyzed   ?? 0;
    const similarEvents  = d.similar_events                  ?? [];

    return (
      <article className="flex gap-4 items-start message-enter">
        <div className="w-8 h-8 rounded-full bg-[#8C5A3C]/10 flex items-center justify-center shrink-0 mt-1">
          <span className="material-symbols-outlined text-[#8C5A3C]"
            style={{ fontSize: "18px", fontVariationSettings: "'FILL' 1" }}>
            videocam
          </span>
        </div>

        <div className="flex-1 space-y-5 pt-1">
          {/* Header */}
          <div className="space-y-2">
            <div className="flex flex-wrap items-start gap-x-3 gap-y-2">
              <h2 className="text-[#2B211F] text-3xl font-bold leading-tight"
                style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                {d.category}
              </h2>
              <div className="flex flex-wrap items-center gap-2 pt-1.5">
                <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${sevColor}`}>
                  {d.severity}
                </span>
                <span className="text-xs font-bold px-2.5 py-1 rounded-full border bg-[#8C5A3C]/10 text-[#8C5A3C] border-[#8C5A3C]/35">
                  Video · {framesAnalyzed} frames
                </span>
                <span className="flex items-center gap-1 text-[#27AE60] text-xs font-semibold">
                  <span className="material-symbols-outlined"
                    style={{ fontSize: "13px", fontVariationSettings: "'FILL' 1" }}>
                    check_circle
                  </span>
                  {Math.round(d.classification_confidence)}% confidence
                </span>
              </div>
            </div>
            {d.visible_damage && (
              <p className="text-[#6B5A53] text-[15px] leading-relaxed">{d.visible_damage}</p>
            )}
          </div>

          {/* Thumbnail + stream properties */}
          <StreamCard />

          {/* Disaster assessment fields */}
          <div className="space-y-3">
            {[
              ["Affected Area",        d.affected_area],
              ["Environmental Impact", d.environmental_impact],
              ["Recommendations",      d.recommendations],
            ].filter(([, v]) => v).map(([label, value]) => (
              <div key={label} className="bg-white rounded-xl border border-[#E8DDD4] p-4">
                <p className="text-[#A08878] text-[10px] font-semibold uppercase tracking-widest mb-1.5">
                  {label}
                </p>
                <p className="text-[#2B211F] text-sm leading-relaxed">{value}</p>
              </div>
            ))}
          </div>

          {/* Similar historical events */}
          {similarEvents.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-[#6B5A53] text-[10px] font-semibold uppercase tracking-widest">
                Similar Historical Events
              </h4>
              <div className="space-y-2">
                {similarEvents.slice(0, 3).map((ev, i) => (
                  <div key={i}
                    className="bg-[#FDF5EE] rounded-xl border border-[#E8DDD4] p-3 flex gap-3 items-start">
                    <span className="text-[#A08878] text-xs font-bold shrink-0 pt-0.5">#{i + 1}</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-[#2B211F] text-sm font-semibold">
                        {ev.event}{" "}
                        <span className="text-[#8C7B73] font-normal">({ev.year})</span>
                      </p>
                      <p className="text-[#6B5A53] text-xs">{ev.location}</p>
                      {ev.description && (
                        <p className="text-[#A08878] text-xs mt-1 leading-relaxed">{ev.description}</p>
                      )}
                    </div>
                    <span className={`text-xs font-bold shrink-0 ${
                      ev.similarity >= 80 ? "text-[#27AE60]" :
                      ev.similarity >= 65 ? "text-[#C08552]" : "text-[#A08878]"
                    }`}>
                      {ev.similarity?.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Frame vote breakdown */}
          {Object.keys(votes).length > 0 && (
            <p className="text-[#A08878] text-[9px]">
              Frame classification votes:{" "}
              {Object.entries(votes).map(([c, v]) => `${c} (${v})`).join(" · ")}
              {" · "}Best frame analyzed with CLIP + Qwen2-VL
            </p>
          )}
        </div>
      </article>
    );
  }

  // ── Metadata-only fallback (models disabled or frame extraction failed) ────
  if (!fi || !an) return null;

  return (
    <article key={msg.id} className="flex gap-4 items-start message-enter">
      <div className="w-8 h-8 rounded-full bg-[#8C5A3C]/10 flex items-center justify-center shrink-0 mt-1">
        <span className="material-symbols-outlined text-[#8C5A3C]"
          style={{ fontSize: "18px", fontVariationSettings: "'FILL' 1" }}>
          videocam
        </span>
      </div>

      <div className="flex-1 space-y-5 pt-1">
        {/* Header */}
        <div className="space-y-2">
          <div className="flex flex-wrap items-start gap-x-3 gap-y-2">
            <h2 className="text-[#2B211F] text-3xl font-bold leading-tight"
              style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
              Video Assessment
            </h2>
            <div className="flex items-center gap-2 pt-1.5">
              <span className="text-xs font-bold px-2.5 py-1 rounded-full border bg-[#8C5A3C]/12 text-[#8C5A3C] border-[#8C5A3C]/40">
                Metadata Only
              </span>
              <span className="flex items-center gap-1 text-[#27AE60] text-xs font-semibold">
                <span className="material-symbols-outlined"
                  style={{ fontSize: "13px", fontVariationSettings: "'FILL' 1" }}>
                  check_circle
                </span>
                Stream Analysis Complete
              </span>
            </div>
          </div>
          <p className="text-[#6B5A53] text-[15px] leading-relaxed whitespace-pre-line">{an.summary}</p>
        </div>

        {/* Thumbnail + stream properties */}
        <StreamCard />

        {/* Pending models */}
        <div className="bg-[#FDF5EE] rounded-xl border border-[#8C5A3C]/25 p-4 space-y-3">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[#8C5A3C]" style={{ fontSize: "16px" }}>
              model_training
            </span>
            <h4 className="text-[#8C5A3C] text-xs font-semibold uppercase tracking-widest">
              Video Intelligence Models — Pending Integration
            </h4>
          </div>
          <p className="text-[#A08878] text-xs">{an.assessment_note}</p>
          <div className="space-y-0.5">
            {an.pending_models.map((pm) => (
              <div key={pm.model} className="flex items-center gap-3 py-2 border-b border-[#E8DDD4] last:border-none">
                <div className="w-2 h-2 rounded-full bg-[#8C5A3C]/35 border border-[#8C5A3C]/55 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-[#8C5A3C] text-xs font-semibold">{pm.model}</p>
                  <p className="text-[#A08878] text-[11px]">{pm.description}</p>
                </div>
                <span className="hidden sm:inline text-[10px] text-[#8C5A3C]/50 shrink-0"
                  style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                  {pm.endpoint}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </article>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  const [showSplash,    setShowSplash]    = useState(true);
  const [phase,         setPhase]         = useState("upload");
  const [file,          setFile]          = useState(null);
  const [previewUrl,    setPreviewUrl]    = useState(null);
  const [isDragging,    setIsDragging]    = useState(false);
  const [fileError,     setFileError]     = useState(null);
  const [modelOutputs,  setModelOutputs]  = useState({});
  const [modelStatus,   setModelStatus]   = useState({});
  const [analysisError, setAnalysisError] = useState(null);
  const [timeline,      setTimeline]      = useState([]);
  const [disasterCtx,   setDisasterCtx]   = useState(null);
  const [chatHistory,   setChatHistory]   = useState([]);
  const [inputValue,    setInputValue]    = useState("");
  const [isTyping,      setIsTyping]      = useState(false);
  const [copiedId,          setCopiedId]          = useState(null);
  const [greetingVisible,   setGreetingVisible]   = useState(false);
  const [idleStatusIdx,     setIdleStatusIdx]     = useState(0);
  const [idleStatusVisible, setIdleStatusVisible] = useState(true);
  const [memory,            setMemory]            = useState(INITIAL_MEMORY);
  const [userName,          setUserName]          = useState(() => localStorage.getItem(NAME_KEY));
  const [nameInput,         setNameInput]         = useState("");
  const [timeGreeting]                            = useState(getTimeGreeting);
  const [fileMode,          setFileMode]          = useState("image"); // "image" | "video"
  const [videoAnalysis,     setVideoAnalysis]     = useState(null);
  const [analysisMode,      setAnalysisMode]      = useState("unified"); // "unified" | "research"
  const [unifiedResult,     setUnifiedResult]     = useState(null);
  const [rotatingMsgIdx,    setRotatingMsgIdx]    = useState(0);

  const fileInputRef = useRef(null);
  const chatEndRef   = useRef(null);
  const inputRef     = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, isTyping]);

  useEffect(() => {
    if (phase !== "upload") return;
    setGreetingVisible(false);
    setIdleStatusIdx(0);
    setIdleStatusVisible(true);
    const t = setTimeout(() => setGreetingVisible(true), 900);
    return () => clearTimeout(t);
  }, [phase]);

  useEffect(() => {
    if (phase !== "upload" || !greetingVisible) return;
    const timers = [];
    function step() {
      const wait = 5000 + Math.random() * 3000;
      const t1 = setTimeout(() => {
        setIdleStatusVisible(false);
        const t2 = setTimeout(() => {
          setIdleStatusIdx((i) => (i + 1) % IDLE_STATUS_MESSAGES.length);
          setIdleStatusVisible(true);
          step();
        }, 450);
        timers.push(t2);
      }, wait);
      timers.push(t1);
    }
    step();
    return () => timers.forEach(clearTimeout);
  }, [phase, greetingVisible]);

  // ── Atmosphere theming — fires on phase/disaster type change ───────────────
  useEffect(() => {
    if (phase === "ready" && disasterCtx?.eventType) {
      applyTheme(disasterCtx.eventType);
    } else if (phase === "upload") {
      resetTheme();
    }
  }, [phase, disasterCtx?.eventType]);

  // ── Rotate loading messages during unified analysis ─────────────────────────
  useEffect(() => {
    if (phase !== "analyzing" || analysisMode !== "unified") return;
    setRotatingMsgIdx(0);
    const id = setInterval(
      () => setRotatingMsgIdx((i) => (i + 1) % UNIFIED_LOADING_MSGS.length),
      2500,
    );
    return () => clearInterval(id);
  }, [phase, analysisMode]);

  // ── File handling ──────────────────────────────────────────────────────────

  const handleFile = useCallback((f) => {
    if (!f) return;
    const ext     = f.name.split(".").pop().toLowerCase();
    const isImage = f.type.startsWith("image/");
    const isVideo = f.type.startsWith("video/") || VIDEO_EXTENSIONS.has(ext);

    if (!isImage && !isVideo) {
      setFileError("Unsupported format — upload an image (JPEG, PNG, WebP) or video (MP4, MOV, AVI, MKV).");
      return;
    }
    const maxBytes = isVideo ? MAX_VIDEO_FILE : MAX_FILE_SIZE;
    const maxMB    = isVideo ? MAX_VIDEO_FILE_MB : MAX_FILE_SIZE_MB;
    if (f.size > maxBytes) {
      setFileError(`File too large — ${(f.size / 1024 / 1024).toFixed(1)} MB exceeds the ${maxMB} MB limit.`);
      return;
    }
    setFileError(null);
    setFile(f);
    setFileMode(isVideo ? "video" : "image");
    setPreviewUrl(URL.createObjectURL(f));
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const clearImage = useCallback((e) => {
    e.stopPropagation();
    setFile(null);
    setPreviewUrl(null);
    setFileError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const handleSaveName = useCallback((rawName) => {
    const name = rawName.trim();
    localStorage.setItem(NAME_KEY, name);
    setUserName(name);
    setNameInput("");
  }, []);

  // ── Reset ──────────────────────────────────────────────────────────────────

  const resetToUpload = () => {
    resetTheme();
    setPhase("upload");
    setFile(null);
    setPreviewUrl(null);
    setFileMode("image");
    setModelOutputs({});
    setModelStatus({});
    setDisasterCtx(null);
    setChatHistory([]);
    setInputValue("");
    setFileError(null);
    setAnalysisError(null);
    setTimeline([]);
    setVideoAnalysis(null);
    setUnifiedResult(null);
  };

  // ── Analysis ───────────────────────────────────────────────────────────────

  const handleUnifiedAnalyze = async () => {
    setPhase("analyzing");
    setModelOutputs({});
    setAnalysisError(null);
    setDisasterCtx(null);
    setChatHistory([]);
    setUnifiedResult(null);
    setTimeline([{ id: 1, text: "Submitting image to the analysis engine..." }]);
    setModelStatus({ clip: "running" });

    const t1 = setTimeout(() => {
      setTimeline((p) => [...p, { id: 2, text: "Examining the scene..." }]);
    }, 600);

    const t0 = performance.now();
    try {
      const data = await callModel("/predict/disaster", file);
      clearTimeout(t1);

      if (data.status === "disabled") {
        throw new Error(data.message || "Analysis engine is currently unavailable.");
      }

      const elapsed = Math.round(performance.now() - t0);
      const report = {
        category:                  data.category,
        classification_confidence: data.classification_confidence,
        severity:                  data.severity,
        visible_damage:            data.visible_damage,
        affected_area:             data.affected_area,
        environmental_impact:      data.environmental_impact,
        recommendations:           data.recommendations,
        similar_events:            data.similar_events ?? [],
        active_models:             data.active_models ?? ["CLIP", "Qwen2-VL"],
        processing_time_ms:        data.processing_time_ms ?? elapsed,
      };

      setModelStatus({ clip: "complete" });
      setTimeline([
        { id: 3, text: `CLIP classification → ${report.category} (${Math.round(report.classification_confidence)}%)` },
        { id: 4, text: `Qwen2-VL structured assessment complete` },
        { id: 5, text: `Severity: ${report.severity}` },
        { id: 6, text: "Intelligence report ready" },
      ]);

      setUnifiedResult(report);
      setDisasterCtx({
        eventType:     report.category,
        severity:      report.severity,
        confidence:    report.classification_confidence,
        caption:       report.visible_damage,
        reasoning:     report.environmental_impact,
        sceneAnalysis: report.recommendations,
        isUnified:     true,
      });

      setChatHistory([{
        id:      Date.now(),
        role:    "assistant",
        type:    "unified-briefing",
        content: `${report.category} — ${report.severity} severity.`,
        time:    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      }]);

      setPhase("ready");
      setTimeout(() => inputRef.current?.focus(), 300);

      generateThumb(file).then((thumb) => {
        setMemory((prev) => {
          const entry = {
            id:              Date.now().toString(),
            timestamp:       new Date().toISOString(),
            eventType:       report.category,
            severity:        report.severity,
            confidence:      report.classification_confidence,
            imageName:       file.name,
            imageThumb:      thumb,
            briefingSummary: report.visible_damage.slice(0, 250),
            briefingFull:    `${report.category} — ${report.severity}`,
            disasterCtx: {
              eventType:     report.category,
              confidence:    report.classification_confidence,
              severity:      report.severity,
              caption:       report.visible_damage,
              reasoning:     report.environmental_impact,
              sceneAnalysis: report.recommendations,
            },
          };
          const updated = {
            totalIncidents: prev.totalIncidents + 1,
            lastVisit:      new Date().toISOString(),
            assessments:    [entry, ...prev.assessments].slice(0, MAX_ASSESSMENTS),
          };
          saveMemory(updated);
          return updated;
        });
      });
    } catch (err) {
      clearTimeout(t1);
      const msg = err.name === "AbortError" ? "Timed out — model took too long" : err.message;
      setAnalysisError(msg);
      setModelStatus({ clip: "failed" });
      setTimeout(() => { setPhase("upload"); setAnalysisError(null); }, 5000);
    }
  };

  const handleVideoAnalyze = async () => {
    setPhase("analyzing");
    setModelOutputs({});
    setAnalysisError(null);
    setDisasterCtx(null);
    setChatHistory([]);
    setUnifiedResult(null);
    setTimeline([{ id: 0, text: "Receiving video and extracting stream properties..." }]);
    setModelStatus({ video: "running" });

    try {
      const data = await callVideoAnalysis(file);
      const hasFullReport = !!(data.category);

      setVideoAnalysis(data);
      setModelStatus({ video: "complete" });

      if (hasFullReport) {
        // Full disaster intelligence report (CLIP + Qwen2-VL + FAISS ran on extracted frames)
        const report = {
          category:                  data.category,
          classification_confidence: data.classification_confidence,
          severity:                  data.severity,
          visible_damage:            data.visible_damage,
          affected_area:             data.affected_area,
          environmental_impact:      data.environmental_impact,
          recommendations:           data.recommendations,
          similar_events:            data.similar_events ?? [],
          active_models:             data.active_models ?? ["CLIP", "Qwen2-VL"],
          processing_time_ms:        data.processing_time_ms,
        };
        setUnifiedResult(report);
        setTimeline([
          { id: 1, text: `${data.frames_analyzed} frames extracted at 25/50/75/90%` },
          { id: 2, text: `CLIP majority vote → ${data.category} (${Math.round(data.classification_confidence)}%)` },
          { id: 3, text: "Qwen2-VL structured assessment complete" },
          { id: 4, text: `Severity: ${data.severity}` },
        ]);
        setDisasterCtx({
          eventType:     data.category,
          severity:      data.severity,
          confidence:    data.classification_confidence,
          caption:       data.visible_damage,
          reasoning:     data.environmental_impact,
          sceneAnalysis: data.recommendations,
          isVideo:       true,
        });
        setChatHistory([{
          id:      Date.now(),
          role:    "assistant",
          type:    "video-briefing",
          content: `${data.category} — ${data.severity} severity.`,
          time:    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        }]);
      } else {
        // Metadata-only fallback (models disabled or frame extraction unavailable)
        const fi = data.file_info ?? {};
        setTimeline([
          { id: 1, text: fi.resolution ? `Stream: ${fi.resolution} · ${(fi.fps ?? 0).toFixed(1)} fps · ${(fi.duration_s ?? 0).toFixed(1)}s` : "Stream metadata extracted" },
          { id: 2, text: fi.codec ? `Codec: ${fi.codec.toUpperCase()} · ${(fi.size_mb ?? 0).toFixed(1)} MB` : "File metadata extracted" },
          { id: 3, text: data.thumbnail_b64 ? "Thumbnail frame extracted" : "Thumbnail unavailable — ffmpeg/opencv not found" },
          { id: 4, text: "Metadata analysis complete — video models pending integration" },
        ]);
        setDisasterCtx({ eventType: "Video Assessment", severity: "Pending", confidence: 0, isVideo: true });
        setChatHistory([{
          id:      Date.now(),
          role:    "assistant",
          type:    "video-briefing",
          content: data.analysis?.summary ?? "Video received. Stream metadata extracted.",
          time:    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        }]);
      }

      setPhase("ready");
      setTimeout(() => inputRef.current?.focus(), 300);

      setMemory((prev) => {
        const entry = {
          id:              Date.now().toString(),
          timestamp:       new Date().toISOString(),
          eventType:       hasFullReport ? data.category : "Video Assessment",
          severity:        hasFullReport ? data.severity : "Pending",
          confidence:      hasFullReport ? data.classification_confidence : 0,
          imageName:       file.name,
          imageThumb:      data.thumbnail_b64 ?? null,
          briefingSummary: hasFullReport
            ? (data.visible_damage ?? "").slice(0, 250)
            : (data.analysis?.summary ?? "").slice(0, 250),
          briefingFull:    hasFullReport
            ? `${data.category} — ${data.severity}`
            : (data.analysis?.summary ?? ""),
          disasterCtx: hasFullReport
            ? {
                eventType:     data.category,
                confidence:    data.classification_confidence,
                severity:      data.severity,
                caption:       data.visible_damage,
                reasoning:     data.environmental_impact,
                sceneAnalysis: data.recommendations,
              }
            : { eventType: "Video Assessment", confidence: 0, severity: "Pending", caption: "", reasoning: "", sceneAnalysis: "" },
        };
        const updated = {
          totalIncidents: prev.totalIncidents + 1,
          lastVisit:      new Date().toISOString(),
          assessments:    [entry, ...prev.assessments].slice(0, MAX_ASSESSMENTS),
        };
        saveMemory(updated);
        return updated;
      });
    } catch (err) {
      setAnalysisError(err.message);
      setModelStatus({ video: "failed" });
      setTimeout(() => { setPhase("upload"); setAnalysisError(null); }, 5000);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setFileError("Please select an image or video to analyze.");
      setTimeout(() => setFileError(null), 3000);
      return;
    }

    if (fileMode === "video") return handleVideoAnalyze();
    if (analysisMode === "unified") return handleUnifiedAnalyze();

    setPhase("analyzing");
    setModelOutputs({});
    setAnalysisError(null);
    setDisasterCtx(null);
    setChatHistory([]);
    setTimeline([{ id: 0, text: "I'm taking a look at what you've shared..." }]);
    setModelStatus(MODELS.reduce((a, m) => ({ ...a, [m.key]: "waiting" }), {}));

    const results = await Promise.all(
      MODELS.map(async (model, idx) => {
        await new Promise((r) => setTimeout(r, idx * 120));
        setModelStatus((prev) => ({ ...prev, [model.key]: "running" }));
        try {
          const data = await callModel(model.endpoint, file);
          setModelOutputs((prev) => ({ ...prev, [model.key]: data }));
          setModelStatus((prev) => ({ ...prev, [model.key]: "complete" }));
          setTimeline((prev) => [...prev, { id: Date.now() + Math.random(), text: MODEL_TIMELINE_LABEL[model.key](data) }]);
          return { key: model.key, data };
        } catch (err) {
          const errData = { error: err.message };
          setModelOutputs((prev) => ({ ...prev, [model.key]: errData }));
          setModelStatus((prev) => ({ ...prev, [model.key]: "failed" }));
          setTimeline((prev) => [...prev, { id: Date.now() + Math.random(), text: "One perspective is unavailable — continuing with the rest..." }]);
          return { key: model.key, data: errData };
        }
      })
    );

    const successCount = results.filter((r) => !r.data.error).length;
    if (successCount === 0) {
      setAnalysisError("Unable to reach the analysis pipeline. Verify your connection and confirm the backend is running.");
      setTimeline((prev) => [...prev, { id: Date.now(), text: "I couldn't complete the analysis — check your connection and try again" }]);
      setTimeout(() => { setPhase("upload"); setAnalysisError(null); }, 5000);
      return;
    }

    setTimeline((prev) => [...prev, { id: Date.now(), text: "I'm putting together my full assessment..." }]);

    const outputs     = Object.fromEntries(results.map((r) => [r.key, r.data]));
    const ctx         = buildDisasterContext(outputs);
    const description = buildDescription(ctx);

    setDisasterCtx(ctx);
    setChatHistory([{
      id:      Date.now(),
      role:    "assistant",
      type:    "briefing",
      content: description,
      time:    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }]);
    setPhase("ready");
    setTimeout(() => inputRef.current?.focus(), 300);

    // Persist to memory (async thumbnail generation, non-blocking)
    generateThumb(file).then((thumb) => {
      const newEntry = {
        id:              Date.now().toString(),
        timestamp:       new Date().toISOString(),
        eventType:       ctx.eventType,
        severity:        ctx.severity,
        confidence:      ctx.confidence,
        imageName:       file.name,
        imageThumb:      thumb,
        briefingSummary: description.slice(0, 250),
        briefingFull:    description,
        disasterCtx: {
          eventType:    ctx.eventType,
          confidence:   ctx.confidence,
          severity:     ctx.severity,
          caption:      ctx.caption,
          reasoning:    ctx.reasoning,
          sceneAnalysis: ctx.sceneAnalysis,
        },
      };
      setMemory((prev) => {
        const updated = {
          totalIncidents: prev.totalIncidents + 1,
          lastVisit:      new Date().toISOString(),
          assessments:    [newEntry, ...prev.assessments].slice(0, MAX_ASSESSMENTS),
        };
        saveMemory(updated);
        return updated;
      });
    });
  };

  // ── Chat ───────────────────────────────────────────────────────────────────

  const handleChat = async (question) => {
    const q = (question ?? inputValue).trim();
    if (!q || isTyping || phase !== "ready") return;
    if (!question) setInputValue("");

    const userMsg = {
      id:   Date.now(),
      role: "user",
      content: q,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };
    setChatHistory((prev) => [...prev, userMsg]);
    setIsTyping(true);

    try {
      const response = await callChat(q, disasterCtx, [...chatHistory, userMsg]);
      setChatHistory((prev) => [...prev, {
        id:      Date.now() + 1,
        role:    "assistant",
        content: response,
        time:    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      }]);
    } catch {
      const fallback = buildFallbackResponse(q, disasterCtx);
      setChatHistory((prev) => [...prev, {
        id:         Date.now() + 1,
        role:       "assistant",
        content:    fallback,
        isFallback: true,
        time:       new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      }]);
    } finally {
      setIsTyping(false);
      inputRef.current?.focus();
    }
  };

  // ── Regenerate last response ────────────────────────────────────────────────

  const handleRegenerate = async () => {
    if (isTyping) return;

    // Find last user message and last assistant (non-briefing) message
    let lastUserMsg  = null;
    let lastAsstIdx  = -1;
    for (let i = chatHistory.length - 1; i >= 0; i--) {
      if (lastAsstIdx === -1 && chatHistory[i].role === "assistant" && !chatHistory[i].type) lastAsstIdx = i;
      if (!lastUserMsg && chatHistory[i].role === "user") { lastUserMsg = chatHistory[i]; break; }
    }
    if (!lastUserMsg || lastAsstIdx === -1) return;

    const trimmedHistory = [...chatHistory.slice(0, lastAsstIdx), ...chatHistory.slice(lastAsstIdx + 1)];
    setChatHistory(trimmedHistory);
    setIsTyping(true);

    try {
      const contextHistory = trimmedHistory.filter((m) => m.type !== "briefing").slice(-8);
      const response = await callChat(lastUserMsg.content, disasterCtx, contextHistory);
      setChatHistory((prev) => [...prev, {
        id:      Date.now(),
        role:    "assistant",
        content: response,
        time:    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      }]);
    } catch {
      const fallback = buildFallbackResponse(lastUserMsg.content, disasterCtx);
      setChatHistory((prev) => [...prev, {
        id:         Date.now(),
        role:       "assistant",
        content:    fallback,
        isFallback: true,
        time:       new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      }]);
    } finally {
      setIsTyping(false);
      inputRef.current?.focus();
    }
  };

  // ── Copy to clipboard ──────────────────────────────────────────────────────

  const handleCopy = async (id, content) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      // Clipboard API unavailable — ignore silently
    }
  };

  // ── Restore a previous assessment from memory ─────────────────────────────

  const handleRestoreAssessment = (assessment) => {
    const ctx = {
      ...assessment.disasterCtx,
      top3:               [],
      keywords:           [],
      llavaMetrics:       {},
      qwenMetrics:        {},
      impacts:             IMPACTS[assessment.disasterCtx.eventType]              ?? [],
      actions:             ACTIONS[assessment.disasterCtx.eventType]              ?? [],
      infrastructure:      INFRASTRUCTURE[assessment.disasterCtx.eventType]       ?? [],
      humanImpact:         HUMAN_IMPACTS[assessment.disasterCtx.eventType]        ?? [],
      environmentalImpact: ENVIRONMENTAL_IMPACTS[assessment.disasterCtx.eventType] ?? [],
    };
    setDisasterCtx(ctx);
    setPreviewUrl(assessment.imageThumb ?? null);
    setModelOutputs({});
    setModelStatus({});
    setChatHistory([
      {
        id:          Date.now(),
        role:        "assistant",
        type:        "briefing",
        isRestored:  true,
        content:     assessment.briefingFull,
        time:        new Date(assessment.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
      {
        id:      Date.now() + 1,
        role:    "assistant",
        content: `I've restored your ${assessment.eventType.toLowerCase()} assessment from ${formatTimeAgo(assessment.timestamp)}. Detailed model evidence isn't available in a restored session, but I can still help you explore this incident. What would you like to know?`,
        time:    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
    setPhase("ready");
    setTimeout(() => inputRef.current?.focus(), 300);
  };

  // ── Personalized greeting (reactive to userName) ───────────────────────────
  const greeting = INITIAL_MEMORY.totalIncidents > 0
    ? (userName ? `Welcome back, ${userName}.` : "Welcome back.")
    : (userName ? `${timeGreeting}, ${userName}.` : `${timeGreeting}.`);

  // ── Atmosphere label (empty string = Default / no active atmosphere) ────────
  const atmosphereLabel = (phase === "ready" && disasterCtx?.eventType)
    ? getTheme(disasterCtx.eventType).label
    : "";

  // ── Shared top navigation ──────────────────────────────────────────────────

  const TopNav = (
    <header className="fixed top-0 left-0 right-0 z-50 bg-[#FFF8F0]/95 backdrop-blur-md border-b border-[#E8DDD4]">
      <div className="flex justify-between items-center w-full px-3 sm:px-6 md:px-12 py-3.5 max-w-[1200px] mx-auto">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-[#C08552]/20 rounded-lg flex items-center justify-center border border-[#C08552]/40 shadow-glow-sm">
            <span
              className="material-symbols-outlined text-[#C08552] text-[18px]"
              style={{ fontVariationSettings: "'FILL' 1" }}
            >
              radar
            </span>
          </div>
          <div className="flex flex-col leading-none gap-0.5">
            <h1 className="text-[15px] font-semibold text-[#2B211F] tracking-tight" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
              Disaster Intelligence
            </h1>
            <span className="hidden sm:inline text-[10px] text-[#A08878] uppercase tracking-[0.14em] font-medium" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
              VLM · Analyzer Platform
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {phase === "ready" && (
            <button
              onClick={resetToUpload}
              className="text-[11px] font-semibold text-[#6B5A53] hover:text-[#2B211F] border border-[#E8DDD4] hover:border-[#C08552]/45 rounded-lg px-3 py-1.5 transition-all hover:bg-[#C08552]/8"
            >
              ↩<span className="hidden sm:inline"> New Analysis</span>
            </button>
          )}
          <button className="hidden sm:flex text-[#A08878] hover:text-[#6B5A53] transition-colors p-2 rounded-lg hover:bg-[#E8DDD4]">
            <span className="material-symbols-outlined" style={{ fontSize: "19px" }}>help_outline</span>
          </button>
          <button className="hidden sm:flex text-[#A08878] hover:text-[#6B5A53] transition-colors p-2 rounded-lg hover:bg-[#E8DDD4]">
            <span className="material-symbols-outlined" style={{ fontSize: "19px" }}>settings</span>
          </button>
        </div>
      </div>
    </header>
  );

  // ---------------------------------------------------------------------------
  // PHASE: upload
  // ---------------------------------------------------------------------------

  if (showSplash) return <IntroAnimation onComplete={() => setShowSplash(false)} />;

  if (phase === "upload") {
    return (
      <div className="h-screen flex flex-col overflow-hidden atm-bg">
        {TopNav}

        <main className="flex-1 overflow-y-auto relative">
          {/* Atmospheric gradients */}
          <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
            <div className="absolute -top-[10%] -right-[5%] w-[50%] h-[50%] blur-[160px] rounded-full" style={{ background: 'var(--atm-glow-1)' }} />
            <div className="absolute -bottom-[10%] -left-[5%] w-[40%] h-[40%] blur-[140px] rounded-full" style={{ background: 'var(--atm-glow-2)' }} />
          </div>

          {/* Centering wrapper — vertically centers content in the available viewport, scrolls on overflow */}
          <div className="min-h-full flex items-center justify-center px-4 md:px-8 pt-16 pb-4">
            <div className="max-w-[560px] w-full flex flex-col z-10 gap-2.5">

              {/* Assistant greeting — heading style */}
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-1.5">
                  <div className="w-5 h-5 rounded-md bg-[#C08552]/15 flex items-center justify-center shrink-0">
                    <span
                      className="material-symbols-outlined text-[#C08552]"
                      style={{ fontSize: "12px", fontVariationSettings: "'FILL' 1" }}
                    >
                      radar
                    </span>
                  </div>
                  <span className="text-[10px] font-medium text-[#A08878] uppercase tracking-widest">
                    Disaster Intelligence
                  </span>
                </div>

                {!greetingVisible ? (
                  <div className="flex items-center gap-1.5 h-7">
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                  </div>
                ) : (
                  <div className="greeting-enter flex flex-col gap-1">
                    <h2 className="text-[#2B211F] text-[22px] font-semibold leading-tight">
                      {greeting}
                    </h2>

                    {userName !== null ? (
                      <div className="flex flex-col gap-0.5">
                        <p className="text-[#6B5A53] text-[13px]">
                          What would you like me to investigate today?
                        </p>
                        {memory.totalIncidents > 0 && (
                          <div className="greeting-enter-slow flex items-center gap-1.5 flex-wrap mt-1">
                            <span className="text-[#A08878] text-[11px]">Recent:</span>
                            {memory.assessments.slice(0, 2).map((a) => (
                              <button
                                key={a.id}
                                onClick={() => handleRestoreAssessment(a)}
                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white border border-[#E8DDD4] text-[#6B5A53] text-[11px] hover:border-[#C08552]/35 hover:text-[#2B211F] transition-all"
                              >
                                {a.eventType} · {formatTimeAgo(a.timestamp)}
                                <span className="material-symbols-outlined" style={{ fontSize: "10px" }}>arrow_forward</span>
                              </button>
                            ))}
                          </div>
                        )}
                        <p
                          className="idle-status text-[10px] text-[#A08878] mt-0.5"
                          style={{ opacity: idleStatusVisible ? 1 : 0 }}
                        >
                          {IDLE_STATUS_MESSAGES[idleStatusIdx]}
                        </p>
                      </div>
                    ) : (
                      <div className="greeting-enter-slow flex flex-col gap-2 mt-0.5">
                        <p className="text-[#6B5A53] text-[13px]">What should I call you?</p>
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={nameInput}
                            onChange={(e) => setNameInput(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleSaveName(nameInput)}
                            placeholder="Your name or callsign"
                            maxLength={32}
                            autoFocus
                            className="bg-white border border-[#E8DDD4] rounded-lg px-3 py-1.5 text-[#2B211F] text-sm placeholder-[#A08878] outline-none focus:border-[#C08552]/50 focus:ring-1 focus:ring-[#C08552]/20 transition-all w-44"
                          />
                          <button
                            onClick={() => handleSaveName(nameInput)}
                            disabled={!nameInput.trim()}
                            className="w-8 h-8 rounded-lg bg-[#C08552] text-white flex items-center justify-center hover:bg-[#8C5A3C] transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                          >
                            <span className="material-symbols-outlined" style={{ fontSize: "15px", fontVariationSettings: "'FILL' 1" }}>arrow_forward</span>
                          </button>
                          <button
                            onClick={() => handleSaveName("")}
                            className="text-[#A08878] text-xs hover:text-[#6B5A53] transition-colors"
                          >
                            skip
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Drop zone */}
              <div className="w-full group">
                <div
                  id="drop-zone"
                  role="button"
                  tabIndex={0}
                  aria-label="Upload disaster image"
                  onClick={() => fileInputRef.current?.click()}
                  onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
                  onDrop={onDrop}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  className={`upload-dashed rounded-xl min-h-[96px] flex flex-col items-center justify-center p-3
                    cursor-pointer transition-all duration-300
                    ${isDragging ? "upload-dashed-active bg-[#C08552]/6 scale-[1.01]" : "bg-[#FDFAF5] hover:bg-[#FDF5EE]"}`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*,video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,.avi,.mov,.mkv"
                    className="hidden"
                    onChange={(e) => handleFile(e.target.files[0])}
                  />

                  {previewUrl ? (
                    /* ── File selected ── */
                    <div className="space-y-2 w-full flex flex-col items-center" onClick={(e) => e.stopPropagation()}>
                      <div className="relative inline-block">
                        {fileMode === "video" ? (
                          <video
                            src={previewUrl}
                            className="max-h-40 max-w-full rounded-xl object-contain shadow-xl"
                            controls
                            muted
                            playsInline
                          />
                        ) : (
                          <img
                            src={previewUrl}
                            alt="preview"
                            className="max-h-40 max-w-full rounded-xl object-contain shadow-xl"
                            onClick={() => fileInputRef.current?.click()}
                            style={{ cursor: "pointer" }}
                          />
                        )}
                        <button
                          onClick={clearImage}
                          className="absolute top-2 right-2 w-6 h-6 rounded-full bg-[#C08552]/70 backdrop-blur-sm flex items-center justify-center hover:bg-[#8C5A3C] transition-colors border border-[#E8DDD4]"
                          title="Remove file"
                        >
                          <span className="material-symbols-outlined text-white" style={{ fontSize: "13px" }}>close</span>
                        </button>
                      </div>
                      <div className="flex items-center gap-2 flex-wrap text-xs text-[#2B211F]">
                        {/* Mode badge */}
                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px] font-semibold ${
                          fileMode === "video"
                            ? "bg-[#8C5A3C]/12 border-[#8C5A3C]/35 text-[#8C5A3C]"
                            : "bg-[#C08552]/15 border-[#C08552]/30 text-[#C08552]"
                        }`}>
                          <span className="material-symbols-outlined" style={{ fontSize: "11px", fontVariationSettings: "'FILL' 1" }}>
                            {fileMode === "video" ? "videocam" : "image"}
                          </span>
                          {fileMode === "video" ? "Video" : "Image"}
                        </span>
                        <span className="max-w-[120px] sm:max-w-[180px] truncate">{file?.name}</span>
                        <span className="text-[#8C7B73]">·</span>
                        <span className="text-[#6B5A53] shrink-0">{file ? (file.size / 1024 / 1024).toFixed(1) + " MB" : ""}</span>
                        <span className="text-[#8C7B73]">·</span>
                        <button
                          onClick={() => fileInputRef.current?.click()}
                          className="text-[#2B211F] hover:underline shrink-0"
                        >
                          replace
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* ── Empty state ── */
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-[#E8DDD4] flex items-center justify-center group-hover:scale-110 transition-transform duration-300 shrink-0">
                        <span
                          className="material-symbols-outlined text-[#6B5A53] text-xl"
                          style={{ fontVariationSettings: "'FILL' 0, 'wght' 200" }}
                        >
                          perm_media
                        </span>
                      </div>
                      <div className="text-left">
                        <p className="text-[14px] font-medium text-[#2B211F]" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                          Share imagery or footage to get started
                        </p>
                        <p className="text-[#A08878] text-[12px]">
                          Images (JPEG, PNG) or video (MP4, MOV, AVI, MKV)
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* File error / validation message */}
              {fileError && (
                <p className="text-[#E74C3C] text-sm flex items-center gap-2">
                  <span className="material-symbols-outlined text-[#E74C3C]" style={{ fontSize: "15px" }}>warning</span>
                  {fileError}
                </p>
              )}

              {/* Mode selector (image-only — video has no research mode) */}
              {fileMode === "image" && (
                <div className="flex justify-center">
                  <div className="flex items-center gap-0.5 bg-[#FDF5EE] rounded-xl p-1 border border-[#E8DDD4]">
                    <button
                      onClick={() => setAnalysisMode("unified")}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                        analysisMode === "unified"
                          ? "bg-[#C08552] text-white"
                          : "text-[#A08878] hover:text-[#6B5A53]"
                      }`}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "13px", fontVariationSettings: "'FILL' 1" }}>auto_awesome</span>
                      Unified Report
                    </button>
                    <button
                      onClick={() => setAnalysisMode("research")}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                        analysisMode === "research"
                          ? "bg-white text-[#2B211F] border border-[#E8DDD4]"
                          : "text-[#A08878] hover:text-[#6B5A53]"
                      }`}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>biotech</span>
                      Research Mode
                    </button>
                  </div>
                </div>
              )}

              {/* Research Mode notice */}
              {fileMode === "image" && analysisMode === "research" && (
                <div className="flex items-start gap-2.5 bg-[#FDF5EE] border border-[#C08552]/30 rounded-xl px-4 py-3">
                  <span
                    className="material-symbols-outlined text-[#C08552] shrink-0 mt-0.5"
                    style={{ fontSize: "15px", fontVariationSettings: "'FILL' 1" }}
                  >
                    schedule
                  </span>
                  <p className="text-[#6B5A53] text-xs leading-relaxed">
                    <span className="text-[#C08552] font-semibold">Research Mode</span> performs deeper multi-model analysis and may take 1–2 minutes.
                  </p>
                </div>
              )}

              {/* Analyze button + hints row */}
              <div className="flex flex-col items-center gap-2">
                <button
                  onClick={handleAnalyze}
                  className={`px-8 py-2.5 font-semibold text-sm rounded-xl shadow-lg flex items-center gap-2 transition-all duration-200
                    ${fileError && !file
                      ? "bg-[#E74C3C]/20 text-[#E74C3C] border border-[#E74C3C]/40"
                      : "bg-[#C08552] text-white font-bold hover:bg-[#8C5A3C] hover:scale-[1.02] active:scale-[0.98] shadow-glow-md hover:shadow-glow-lg"}`}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                    {fileMode === "video" ? "videocam" : "send"}
                  </span>
                  {fileMode === "video"
                    ? "Analyze Video"
                    : analysisMode === "unified"
                      ? "Generate Report"
                      : "Run All Models"}
                </button>
                <p className="text-[12px] text-[#6B5A53] flex items-center gap-1">
                  <span className="material-symbols-outlined" style={{ fontSize: "12px" }}>auto_awesome</span>
                  {fileMode === "video"
                    ? "Extract stream metadata · generate thumbnail"
                    : analysisMode === "unified"
                      ? "Automated scene analysis · unified intelligence report"
                      : "CLIP · BLIP-2 · LLaVA · Qwen — multi-model comparison"}
                </p>
              </div>

              {/* Pre-upload conversation hints */}
              {greetingVisible && (
                <div className="greeting-enter-late flex flex-col gap-1">
                  <p className="text-[9px] text-[#A08878] uppercase tracking-widest font-semibold">
                    After analysis
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {[
                      "What can you determine from this image?",
                      "Can you estimate the severity?",
                      "What are the immediate response priorities?",
                    ].map((q) => (
                      <span
                        key={q}
                        className="px-2 py-0.5 rounded-full atm-chip border border-[#E8DDD4] text-[11px] text-[#A08878] select-none"
                      >
                        {q}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // PHASE: analyzing
  // ---------------------------------------------------------------------------

  if (phase === "analyzing") {
    const totalModels = fileMode === "video" ? 1 : analysisMode === "unified" ? 1 : MODELS.length;
    const allDone = Object.values(modelStatus).length > 0 &&
      Object.values(modelStatus).every((s) => s === "complete" || s === "failed");

    return (
      <div className="min-h-screen flex flex-col atm-bg">
        {TopNav}

        <main className="flex-grow flex items-center justify-center px-4 pt-28 pb-12">
          <div className="max-w-[600px] w-full space-y-5">

            {/* File context strip */}
            {previewUrl && (
              <div className="flex items-center gap-4 bg-white p-4 rounded-xl border border-[#E8DDD4] shadow-sm">
                {fileMode === "video" ? (
                  <div className="w-20 h-16 rounded-xl shrink-0 bg-[#8C5A3C]/12 border border-[#8C5A3C]/30 flex items-center justify-center">
                    <span className="material-symbols-outlined text-[#8C5A3C] text-2xl">videocam</span>
                  </div>
                ) : (
                  <img src={previewUrl} alt="Analyzing" className="w-20 h-16 rounded-xl object-cover shrink-0" />
                )}
                <div>
                  <p className="text-[#2B211F] font-semibold" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                    {analysisError
                      ? "Analysis failed"
                      : fileMode === "video"
                        ? "Extracting video stream data..."
                        : analysisMode === "unified"
                          ? "Generating disaster intelligence report..."
                          : "Running multi-model analysis..."}
                  </p>
                  <p className="text-[#6B5A53] text-sm mt-0.5">
                    {analysisError
                      ? "Returning in 5 seconds"
                      : fileMode === "video"
                        ? "Running ffprobe and generating thumbnail..."
                        : analysisMode === "unified"
                          ? UNIFIED_LOADING_MSGS[rotatingMsgIdx]
                          : "Examining the scene from multiple angles..."}
                  </p>
                </div>
              </div>
            )}

            {/* Error banner */}
            {analysisError && (
              <div className="bg-[#E74C3C]/10 border border-[#E74C3C]/30 rounded-xl p-4 flex items-start gap-3">
                <span className="material-symbols-outlined text-[#E74C3C] shrink-0 mt-0.5">wifi_off</span>
                <p className="text-[#E74C3C] text-sm leading-relaxed">{analysisError}</p>
              </div>
            )}

            {/* Progress panel — video / unified / research */}
            {fileMode === "video" ? (
              <div className="bg-white p-5 rounded-xl border border-[#8C5A3C]/25 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-[#2B211F] text-sm font-semibold" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                    {analysisError ? "Analysis stopped" : "Extracting video metadata"}
                  </p>
                  <span className="text-[#8C5A3C] text-xs font-semibold" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                    {modelStatus.video === "complete" ? "1 / 1" : "0 / 1"}
                  </span>
                </div>
                <div className="h-1.5 bg-[#E8DDD4] rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      modelStatus.video === "complete" ? "bg-[#27AE60] w-full" :
                      modelStatus.video === "failed"   ? "bg-[#E74C3C] w-full" :
                                                         "bg-[#8C5A3C] animate-pulse w-2/3"
                    }`}
                  />
                </div>
                <p className="text-[#6B5A53] text-xs">
                  {modelStatus.video === "running"   ? "Running ffprobe · extracting thumbnail frame..." :
                   modelStatus.video === "complete"  ? "Stream data extracted — preparing assessment..." :
                   modelStatus.video === "failed"    ? "Analysis failed" :
                                                       "Initializing..."}
                </p>
              </div>
            ) : analysisMode === "unified" ? (
              /* ── Unified mode: single progress bar, no model names ── */
              <div className="bg-white p-5 rounded-xl border border-[#C08552]/25 shadow-sm space-y-4">
                <p className="text-[#2B211F] text-sm font-semibold" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                  {analysisError ? "Analysis stopped" : "Generating Disaster Intelligence Report"}
                </p>
                <div className="h-1.5 bg-[#E8DDD4] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#C08552] rounded-full transition-all duration-700 ease-out"
                    style={{
                      width: modelStatus.clip === "complete" ? "100%" :
                             modelStatus.clip === "running"  ? "30%"  : "0%",
                    }}
                  />
                </div>
                <p className="text-[#6B5A53] text-xs">
                  {analysisError                   ? "Analysis failed"              :
                   modelStatus.clip === "complete" ? "Compiling final report..."    :
                   modelStatus.clip === "running"  ? "Examining the scene..."       :
                                                     "Initializing analysis engine..."}
                </p>
              </div>
            ) : (
              /* ── Research mode: 4-model progress ── */
              <div className="bg-white p-5 rounded-xl border border-[#E8DDD4] shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-[#2B211F] text-sm font-semibold" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                    {analysisError ? "Analysis stopped" : "Running multi-perspective analysis"}
                  </p>
                  <span className="text-[#6B5A53] text-xs font-semibold" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                    {Object.values(modelStatus).filter((s) => s === "complete" || s === "failed").length} / {MODELS.length}
                  </span>
                </div>
                <div className="space-y-2">
                  <div className="h-1.5 bg-[#E8DDD4] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#C08552] rounded-full transition-all duration-500 ease-out"
                      style={{
                        width: `${(Object.values(modelStatus).filter((s) => s === "complete" || s === "failed").length / MODELS.length) * 100}%`,
                      }}
                    />
                  </div>
                  <div className="flex gap-1.5">
                    {MODELS.map((m) => {
                      const s = modelStatus[m.key] ?? "waiting";
                      return (
                        <div
                          key={m.key}
                          className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                            s === "complete" ? "bg-[#27AE60]/80" :
                            s === "failed"   ? "bg-[#E74C3C]/60"     :
                            s === "running"  ? "bg-[#C08552] animate-pulse" :
                                              "bg-[#E8DDD4]"
                          }`}
                        />
                      );
                    })}
                  </div>
                </div>
                <p className="text-[#6B5A53] text-xs">
                  {Object.values(modelStatus).some((s) => s === "running")
                    ? "Examining scene details..."
                    : Object.values(modelStatus).every((s) => s === "complete" || s === "failed")
                    ? "Preparing unified assessment..."
                    : "Initializing analysis pipeline..."}
                </p>
              </div>
            )}

            {/* Analysis timeline */}
            {timeline.length > 0 && (
              <div className="bg-white atm-surface2 rounded-xl border border-[#E8DDD4] p-4 space-y-2">
                {timeline.map((event) => (
                  <div key={event.id} className="flex items-center gap-2 text-xs text-[#6B5A53] timeline-enter">
                    <span
                      className="material-symbols-outlined text-[#27AE60]/80 shrink-0"
                      style={{ fontSize: "12px", fontVariationSettings: "'FILL' 1" }}
                    >
                      check_circle
                    </span>
                    <span>{event.text}</span>
                  </div>
                ))}
                {!analysisError && !allDone && (
                  <div className="flex items-center gap-2 text-xs text-[#A08878]">
                    <div className="w-3 h-3 shrink-0 flex items-center justify-center">
                      <div className="w-2 h-2 rounded-full bg-[#C08552]/50 animate-pulse" />
                    </div>
                    <span>Still examining the details...</span>
                  </div>
                )}
                {!analysisError && allDone && (
                  <div className="flex items-center gap-2 text-xs text-[#A08878]">
                    <div className="w-3 h-3 shrink-0 flex items-center justify-center">
                      <div className="w-2 h-2 rounded-full bg-[#C08552]/50 animate-pulse" />
                    </div>
                    <span>I'm preparing my assessment...</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // PHASE: ready — chat interface
  // ---------------------------------------------------------------------------

  // Pre-compute last assistant non-briefing index for regenerate button placement
  let lastAsstNonBriefingIdx = -1;
  for (let i = chatHistory.length - 1; i >= 0; i--) {
    if (chatHistory[i].role === "assistant" && !chatHistory[i].type) { lastAsstNonBriefingIdx = i; break; }
  }

  return (
    <div className="h-screen overflow-hidden flex flex-col atm-bg">
      {TopNav}

      <div className="flex-1 pt-20 flex overflow-hidden relative">

        {/* ── Left sidebar ─────────────────────────────────────────────────── */}
        <aside className="hidden md:flex fixed left-0 top-20 bottom-0 w-[240px] bg-white atm-surface border-r border-[#E8DDD4] p-4 flex-col gap-4 z-40">

          <h3
            className="text-xs font-semibold text-[#2B211F] uppercase tracking-widest"
            style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}
          >
            Reference Context
          </h3>
          <div className="-mt-2 flex flex-wrap gap-1.5">
            {atmosphereLabel && <span className="atm-label-badge">{atmosphereLabel}</span>}
            {fileMode === "image" && (
              <span className={`text-[10px] font-semibold px-2 py-0.5 rounded border ${
                analysisMode === "unified"
                  ? "bg-[#C08552]/15 border-[#C08552]/30 text-[#C08552]"
                  : "bg-white border-[#E8DDD4] text-[#6B5A53]"
              }`}>
                {analysisMode === "unified" ? "Unified" : "Research"}
              </span>
            )}
          </div>

          {/* Scene image / video */}
          <div className="rounded-xl overflow-hidden border border-[#E8DDD4] relative group">
            {previewUrl ? (
              <>
                {fileMode === "video" ? (
                  <div className="w-full aspect-video bg-[#FDF5EE] flex items-center justify-center">
                    {videoAnalysis?.thumbnail_b64 ? (
                      <img
                        src={videoAnalysis.thumbnail_b64}
                        alt="Video thumbnail"
                        className="w-full aspect-video object-cover grayscale brightness-75 group-hover:grayscale-0 group-hover:brightness-100 transition-all duration-500"
                      />
                    ) : (
                      <span className="material-symbols-outlined text-[#8C5A3C] text-4xl">movie</span>
                    )}
                  </div>
                ) : (
                  <img
                    src={previewUrl}
                    alt="Analysed scene"
                    className="w-full aspect-video object-cover grayscale brightness-75 group-hover:grayscale-0 group-hover:brightness-100 transition-all duration-500"
                  />
                )}
                <div
                  className="absolute bottom-2 left-2 bg-[#C08552]/80 backdrop-blur-sm px-2 py-1 rounded text-[10px] uppercase text-white"
                  style={{ fontFamily: "'JetBrains Mono', monospace" }}
                >
                  {fileMode === "video" ? "VIDEO" : `REF_ID: ${disasterCtx?.eventType?.slice(0, 3).toUpperCase() ?? "EVT"}`}
                </div>
              </>
            ) : (
              <div className="w-full aspect-video bg-[#FDF5EE] atm-surface2 flex items-center justify-center">
                <span className="material-symbols-outlined text-[#A08878] text-4xl">image</span>
              </div>
            )}
          </div>

          {/* Disaster metrics */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center border-b border-[#E8DDD4] pb-1.5">
              <span className="text-[#6B5A53] text-xs">Type</span>
              <span className="text-[#2B211F] text-xs font-semibold truncate max-w-[110px] text-right" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                {disasterCtx?.eventType ?? "—"}
              </span>
            </div>
            <div className="flex justify-between items-center border-b border-[#E8DDD4] pb-1.5">
              <span className="text-[#6B5A53] text-xs">Severity</span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase border ${severityChipClass(disasterCtx?.severity)}`}>
                {disasterCtx?.severity ?? "—"}
              </span>
            </div>
          </div>
        </aside>

        {/* ── Main chat area ────────────────────────────────────────────────── */}
        <section className="flex-1 flex flex-col md:ml-[240px] chat-container overflow-y-auto pb-32">
          {/* Mobile-only sticky context bar — shows disaster type and severity since sidebar is hidden */}
          <div className="md:hidden sticky top-0 z-30 bg-[#FFF8F0]/95 backdrop-blur-sm border-b border-[#E8DDD4] px-4 py-2 flex items-center gap-3 shrink-0">
            {previewUrl && (
              fileMode === "video" ? (
                <div className="w-7 h-7 rounded bg-[#FDF5EE] flex items-center justify-center shrink-0">
                  <span className="material-symbols-outlined text-[#8C5A3C]" style={{ fontSize: "14px" }}>videocam</span>
                </div>
              ) : (
                <img src={previewUrl} alt="scene" className="w-7 h-7 rounded object-cover shrink-0" />
              )
            )}
            <span className="text-[#2B211F] text-sm font-semibold truncate flex-1" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
              {disasterCtx?.eventType ?? "—"}
            </span>
            <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase border shrink-0 ${severityChipClass(disasterCtx?.severity)}`}>
              {disasterCtx?.severity ?? "—"}
            </span>
          </div>
          <div className="w-full max-w-[800px] mx-auto px-4 md:px-6 py-8 flex flex-col gap-8">

            {chatHistory.map((msg, msgIdx) => {

              /* ── Unified report message ── */
              if (msg.type === "unified-briefing" && unifiedResult) {
                return (
                  <UnifiedReportPanel
                    key={msg.id}
                    msg={msg}
                    unifiedResult={unifiedResult}
                    disasterCtx={disasterCtx}
                    onSuggestedQuery={handleChat}
                    chatLength={chatHistory.length}
                  />
                );
              }

              /* ── Video briefing message ── */
              if (msg.type === "video-briefing" && videoAnalysis) {
                return <VideoAssessmentPanel key={msg.id} msg={msg} videoAnalysis={videoAnalysis} unifiedResult={unifiedResult} />;
              }

              /* ── Briefing message ── */
              if (msg.type === "briefing" && disasterCtx) {
                return (
                  <article key={msg.id} className="flex gap-4 items-start message-enter">
                    <div className="w-8 h-8 rounded-full bg-[#C08552]/15 flex items-center justify-center shrink-0 mt-1">
                      <span
                        className="material-symbols-outlined text-[#C08552]"
                        style={{ fontSize: "18px", fontVariationSettings: "'FILL' 1" }}
                      >
                        analytics
                      </span>
                    </div>

                    <div className="flex-1 space-y-5 pt-1">

                      {/* ── Unified Incident Assessment Header ── */}
                      <div className="space-y-3">
                        <div className="flex flex-wrap items-start gap-x-3 gap-y-2">
                          <h2
                            className="text-[#2B211F] text-3xl font-bold leading-tight"
                            style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}
                          >
                            {disasterCtx.eventType}
                          </h2>
                          <div className="flex items-center gap-2 pt-1.5">
                            <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${severityChipClass(disasterCtx.severity)}`}>
                              {disasterCtx.severity}
                            </span>
                            <span className="flex items-center gap-1 text-[#27AE60] text-xs font-semibold">
                              <span
                                className="material-symbols-outlined"
                                style={{ fontSize: "13px", fontVariationSettings: "'FILL' 1" }}
                              >
                                check_circle
                              </span>
                              Assessment Complete
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center gap-3 flex-wrap">
                          <p className="text-[#6B5A53] text-sm">
                            Disaster indicators detected and assessed successfully.
                          </p>
                          {atmosphereLabel && <span className="atm-label-badge">{atmosphereLabel}</span>}
                        </div>
                        <p className="text-[#2B211F]/80 text-[17px] leading-[28px]">{msg.content}</p>
                      </div>

                      {/* Row 1: Risks | Actions */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-white p-4 rounded-xl border border-[#E8DDD4] shadow-sm">
                          <h4 className="text-[#2B211F] text-xs font-semibold uppercase tracking-widest mb-3">
                            Potential Risks
                          </h4>
                          <ul className="text-[#6B5A53] text-sm space-y-2">
                            {disasterCtx.impacts.map((item, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="w-1.5 h-1.5 bg-[#C08552] rounded-full mt-[5px] shrink-0" />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div className="bg-white p-4 rounded-xl border border-[#E8DDD4] shadow-sm">
                          <h4 className="text-[#2B211F] text-xs font-semibold uppercase tracking-widest mb-3">
                            Recommended Actions
                          </h4>
                          <ul className="text-[#6B5A53] text-sm space-y-2">
                            {disasterCtx.actions.map((item, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="w-1.5 h-1.5 bg-[#C08552] rounded-full mt-[5px] shrink-0" />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      {/* Row 2: Infrastructure | Human Impact */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-white p-4 rounded-xl border border-[#E8DDD4] shadow-sm">
                          <h4 className="text-[#6B5A53] text-xs font-semibold uppercase tracking-widest mb-3">
                            Affected Infrastructure
                          </h4>
                          <ul className="text-[#6B5A53] text-sm space-y-2">
                            {disasterCtx.infrastructure.map((item, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="w-1.5 h-1.5 bg-[#C08552]/25 rounded-full mt-[5px] shrink-0" />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div className="bg-white p-4 rounded-xl border border-[#E8DDD4] shadow-sm">
                          <h4 className="text-[#6B5A53] text-xs font-semibold uppercase tracking-widest mb-3">
                            Human Impact
                          </h4>
                          <ul className="text-[#6B5A53] text-sm space-y-2">
                            {disasterCtx.humanImpact.map((item, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="w-1.5 h-1.5 bg-[#C08552]/25 rounded-full mt-[5px] shrink-0" />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      {/* Row 3: Environmental Impact — full width */}
                      <div className="bg-white p-4 rounded-xl border border-[#E8DDD4] shadow-sm">
                        <h4 className="text-[#6B5A53] text-xs font-semibold uppercase tracking-widest mb-3">
                          Environmental Impact
                        </h4>
                        <ul className="text-[#6B5A53] text-sm grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2">
                          {disasterCtx.environmentalImpact.map((item, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="w-1.5 h-1.5 bg-[#C08552]/25 rounded-full mt-[5px] shrink-0" />
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Suggested operational queries */}
                      {chatHistory.length <= 2 && (
                        <div className="pt-2 space-y-2">
                          <p className="text-[10px] text-[#A08878] uppercase tracking-widest font-semibold">
                            Suggested queries
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {getSuggestedQuestions(disasterCtx.eventType).map((q) => (
                              <button
                                key={q}
                                onClick={() => handleChat(q)}
                                className="atm-chip hover:bg-[#C08552]/20 transition-all text-[#6B5A53] hover:text-[#2B211F] px-4 py-2 rounded-full text-xs border border-[#E8DDD4] hover:border-[#C08552]/40"
                              >
                                {q}
                              </button>
                            ))}
                            {INITIAL_MEMORY.assessments.length > 0 && !msg.isRestored && (
                              <button
                                onClick={() => handleChat(
                                  `How does this ${disasterCtx.eventType.toLowerCase()} compare to the ${INITIAL_MEMORY.assessments[0].eventType.toLowerCase()} incident from ${formatTimeAgo(INITIAL_MEMORY.assessments[0].timestamp)}?`
                                )}
                                className="bg-transparent hover:bg-[#8C5A3C]/20 hover:text-[#2B211F] transition-all text-[#6B5A53] px-4 py-2 rounded-full text-xs border border-[#E8DDD4] hover:border-[#8C5A3C]"
                              >
                                Compare with previous {INITIAL_MEMORY.assessments[0].eventType.toLowerCase()} incident
                              </button>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Model evidence */}
                      <EvidencePanel modelOutputs={modelOutputs} />
                    </div>
                  </article>
                );
              }

              /* ── User message ── */
              if (msg.role === "user") {
                return (
                  <div key={msg.id} className="flex justify-end message-enter">
                    <div className="max-w-[75%]">
                      <div className="px-4 py-3 rounded-2xl text-[#2B211F] text-sm leading-relaxed border border-[#C08552]/30" style={{ background: 'rgba(192,133,82,0.10)' }}>
                        {msg.content}
                      </div>
                      <p className="text-xs text-[#A08878] mt-1 text-right">{msg.time}</p>
                    </div>
                  </div>
                );
              }

              /* ── Assistant follow-up message ── */
              const isLastAsst = msgIdx === lastAsstNonBriefingIdx;
              return (
                <article key={msg.id} className="flex gap-4 items-start group message-enter">
                  <div className="w-8 h-8 rounded-full bg-[#C08552]/15 flex items-center justify-center shrink-0 mt-1">
                    <span
                      className="material-symbols-outlined text-[#C08552]"
                      style={{ fontSize: "18px", fontVariationSettings: "'FILL' 1" }}
                    >
                      analytics
                    </span>
                  </div>
                  <div className="flex-1 bg-[#FDF5EE] atm-bubble border border-[#E8DDD4] rounded-2xl rounded-tl-sm px-5 py-4">
                    <p className="text-[#2B211F] text-[16px] leading-[26px] whitespace-pre-line break-words">{msg.content}</p>
                    <div className="flex items-center gap-3 mt-3 flex-wrap">
                      <p className="text-xs text-[#A08878]">{msg.time}</p>
                      {msg.isFallback && (
                        <p className="text-xs text-[#A08878] flex items-center gap-1">
                          <span className="material-symbols-outlined" style={{ fontSize: "11px" }}>wifi_off</span>
                          Local response — backend unavailable
                        </p>
                      )}
                      {/* Copy button */}
                      <button
                        onClick={() => handleCopy(msg.id, msg.content)}
                        className="opacity-40 group-hover:opacity-100 transition-opacity duration-150 text-[#6B5A53] hover:text-[#2B211F] flex items-center gap-1"
                        title="Copy response"
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>
                          {copiedId === msg.id ? "check" : "content_copy"}
                        </span>
                        {copiedId === msg.id && <span className="text-xs text-[#27AE60]">Copied</span>}
                      </button>
                      {/* Regenerate button — only on last assistant message */}
                      {isLastAsst && !isTyping && (
                        <button
                          onClick={handleRegenerate}
                          className="opacity-40 group-hover:opacity-100 transition-opacity duration-150 text-[#6B5A53] hover:text-[#2B211F] flex items-center gap-1"
                          title="Regenerate response"
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>refresh</span>
                          <span className="text-xs">Regenerate</span>
                        </button>
                      )}
                    </div>
                  </div>
                </article>
              );
            })}

            {/* Typing indicator */}
            {isTyping && (
              <article className="flex gap-4 items-start message-enter">
                <div className="w-8 h-8 rounded-full bg-[#C08552]/15 flex items-center justify-center shrink-0 mt-1">
                  <span
                    className="material-symbols-outlined text-[#C08552]"
                    style={{ fontSize: "18px", fontVariationSettings: "'FILL' 1" }}
                  >
                    analytics
                  </span>
                </div>
                <div className="bg-[#FDF5EE] atm-bubble border border-[#E8DDD4] rounded-2xl rounded-tl-sm px-5 py-4 flex items-center gap-1.5">
                  {[0, 150, 300].map((d) => (
                    <span
                      key={d}
                      className="w-2 h-2 rounded-full bg-[#C08552]/50 animate-bounce"
                      style={{ animationDelay: `${d}ms` }}
                    />
                  ))}
                </div>
              </article>
            )}

            <div ref={chatEndRef} />
          </div>
        </section>
      </div>

      {/* ── Fixed chat input ────────────────────────────────────────────────── */}
      <div
        className="fixed bottom-0 left-0 right-0 md:left-[240px] flex justify-center px-4 md:px-6 pt-4 z-50"
        style={{ paddingBottom: "max(1.5rem, env(safe-area-inset-bottom))" }}
      >
        <div className="w-full max-w-[800px] bg-white atm-surface shadow-input rounded-full border border-[#E8DDD4] flex items-center px-4 md:px-6 py-2 focus-within:ring-2 focus-within:ring-[#C08552]/30 transition-all">
          <button className="text-[#A08878] hover:text-[#2B211F] p-2 transition-colors hidden sm:flex">
            <span className="material-symbols-outlined">attachment</span>
          </button>
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleChat()}
            placeholder="Ask me anything about this incident…"
            disabled={isTyping}
            className="flex-1 bg-transparent border-none focus:ring-0 text-[#2B211F] placeholder-[#A08878] px-3 md:px-4 py-2 text-sm outline-none disabled:cursor-not-allowed"
          />
          <div className="flex items-center gap-1 md:gap-2">
            <button className="text-[#A08878] hover:text-[#2B211F] p-2 transition-colors hidden sm:flex">
              <span className="material-symbols-outlined">mic</span>
            </button>
            <button
              onClick={() => handleChat()}
              disabled={!inputValue.trim() || isTyping}
              className="w-9 h-9 md:w-10 md:h-10 rounded-full atm-primary-bg text-white flex items-center justify-center hover:brightness-110 transition-all shadow-md disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>
                arrow_upward
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Atmospheric gradients */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <div className="absolute -top-[10%] -right-[5%] w-[50%] h-[50%] blur-[160px] rounded-full" style={{ background: 'var(--atm-glow-1)' }} />
        <div className="absolute -bottom-[10%] -left-[5%] w-[40%] h-[40%] blur-[140px] rounded-full" style={{ background: 'var(--atm-glow-2)' }} />
      </div>
    </div>
  );
}
