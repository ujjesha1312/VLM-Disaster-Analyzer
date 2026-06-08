import { useState, useRef, useCallback, useEffect } from "react";

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

const API_BASE_URL     = "https://providing-earthy-phonebook.ngrok-free.dev";
const MODEL_TIMEOUT_MS = 180_000;
const CHAT_TIMEOUT_MS  =  60_000;
const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE    = MAX_FILE_SIZE_MB * 1024 * 1024;

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
  clip:  "CLIP-ViT classification complete",
  blip2: "BLIP-2 caption generated",
  llava: "LLaVA reasoning complete",
  qwen:  "Qwen2-VL scene analysis complete",
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
    "How severe is this flood event?",
    "Are evacuation measures immediately needed?",
    "What infrastructure is most at risk?",
    "How long could floodwaters persist in this area?",
    "What rescue operations should be prioritised?",
  ],
  Fire: [
    "How fast might this fire spread?",
    "What aerial and ground resources should be deployed?",
    "Which communities or areas are most at risk?",
    "Are there immediate civilian evacuation needs?",
    "What containment strategy is most effective here?",
  ],
  Earthquake: [
    "How severe is the structural damage visible?",
    "Are there likely trapped survivors requiring extraction?",
    "What aftershock risk should be expected?",
    "Which critical infrastructure has been compromised?",
    "What medical and rescue resources are needed?",
  ],
  Landslide: [
    "What likely triggered this landslide?",
    "Is there risk of secondary slides in the area?",
    "Which roads or transport routes are blocked?",
    "What communities are within the impact zone?",
    "What heavy machinery or recovery operations are needed?",
  ],
  Cyclone: [
    "How intense are the winds in this event?",
    "Which coastal areas face the greatest storm surge risk?",
    "When should evacuation operations begin?",
    "What infrastructure is most vulnerable to wind damage?",
    "What is the projected duration and path of this system?",
  ],
};

const DEFAULT_SUGGESTED = [
  "How severe is this disaster?",
  "Are people at immediate risk?",
  "What infrastructure has been damaged?",
  "What emergency resources should be deployed?",
  "What response actions are recommended?",
];

function getSuggestedQuestions(eventType) {
  return SUGGESTED_BY_TYPE[eventType] ?? DEFAULT_SUGGESTED;
}

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
      method:  "POST",
      headers: { "ngrok-skip-browser-warning": "true" },
      body:    fd,
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
      headers: {
        "Content-Type":               "application/json",
        "ngrok-skip-browser-warning": "true",
      },
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

// ---------------------------------------------------------------------------
// DisasterContext builder
// ---------------------------------------------------------------------------

function buildDisasterContext(outputs) {
  const eventType     = outputs.clip?.prediction  ?? "Unknown Event";
  const confidence    = outputs.clip?.confidence  ?? 0;
  const caption       = outputs.blip2?.caption    ?? "";
  const reasoning     = outputs.llava?.response   ?? "";
  const sceneAnalysis = outputs.qwen?.response    ?? "";

  const severity =
    confidence > 88 ? "Critical" :
    confidence > 75 ? "High"     :
    confidence > 60 ? "Moderate" : "Low";

  const impacts            = IMPACTS[eventType]            ?? ["Environmental and infrastructure damage", "Potential civilian impact", "Service disruption"];
  const actions            = ACTIONS[eventType]            ?? ["Deploy emergency response teams", "Establish incident command", "Prioritise civilian evacuation"];
  const infrastructure     = INFRASTRUCTURE[eventType]     ?? ["Critical infrastructure under assessment", "Utility systems at risk"];
  const humanImpact        = HUMAN_IMPACTS[eventType]      ?? ["Civilian safety under assessment", "Evacuation and medical staging recommended"];
  const environmentalImpact = ENVIRONMENTAL_IMPACTS[eventType] ?? ["Environmental assessment in progress", "Contamination risk under evaluation"];

  return { eventType, confidence, caption, reasoning, sceneAnalysis, severity, impacts, actions, infrastructure, humanImpact, environmentalImpact };
}

function buildDescription(ctx) {
  const { eventType, confidence, severity, caption, reasoning } = ctx;
  const detail = caption
    ? caption.charAt(0).toUpperCase() + caption.slice(1).replace(/\.$/, "") + "."
    : reasoning
      ? reasoning.split(".")[0] + "."
      : `${eventType} conditions have been detected.`;
  const urgency = (severity === "Critical" || severity === "High")
    ? " Immediate response action is required."
    : " Ongoing monitoring and precautionary response are advised.";
  return (
    `The uploaded image depicts a ${eventType.toLowerCase()} event — ${detail} ` +
    `Assessed severity is ${severity.toLowerCase()} with ${confidence}% classification confidence.${urgency}`
  );
}

// ---------------------------------------------------------------------------
// Client-side fallback for /chat
// ---------------------------------------------------------------------------

function buildFallbackResponse(question, ctx) {
  const q        = question.toLowerCase();
  const event    = ctx?.eventType      ?? "disaster";
  const conf     = ctx?.confidence     ?? 0;
  const severity = ctx?.severity       ?? "Unknown";
  const reasoning = ctx?.reasoning     ?? "";
  const scene     = ctx?.sceneAnalysis ?? "";

  if (/sever|how bad|intensity|danger/i.test(q))
    return `This ${event} event is assessed as ${severity} severity (${conf}% CLIP confidence). ${reasoning.split(".")[0] + "." || ""}`;

  if (/emergency|response|protocol|action|help/i.test(q)) {
    const list = (ACTIONS[event] ?? []).slice(0, 4).map((a) => `• ${a}`).join("\n");
    return `Emergency response for ${severity.toLowerCase()} ${event}:\n\n${list}`;
  }

  if (/people|risk|casualt|human|injur/i.test(q))
    return `Civilian risk for this ${event} event is assessed as ${conf > 80 ? "HIGH" : "MODERATE"}. Immediate evacuation and medical staging are recommended.`;

  if (/impact|environment|damage|infrastructure/i.test(q)) {
    const list = (IMPACTS[event] ?? []).slice(0, 4).map((i) => `• ${i}`).join("\n");
    return `Identified impacts of this ${event}:\n\n${list}`;
  }

  return `Regarding the ${event} event (${conf}% confidence, ${severity} severity): ${scene.split(".")[0] + "." || reasoning.split(".")[0] + "." || ""} How else can I assist?`;
}

// ---------------------------------------------------------------------------
// Severity chip styling
// ---------------------------------------------------------------------------

function severityChipClass(severity) {
  switch (severity) {
    case "Critical": return "bg-red-500/10 text-red-400 border-red-500/30";
    case "High":     return "bg-orange-500/10 text-orange-400 border-orange-500/30";
    case "Moderate": return "bg-yellow-500/10 text-yellow-400 border-yellow-500/30";
    default:         return "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
  }
}

// ---------------------------------------------------------------------------
// EvidencePanel — collapsible model output details inside the briefing
// ---------------------------------------------------------------------------

function EvidencePanel({ modelOutputs }) {
  const [open, setOpen] = useState(false);
  const clip  = modelOutputs.clip  ?? {};
  const blip2 = modelOutputs.blip2 ?? {};
  const llava = modelOutputs.llava ?? {};
  const qwen  = modelOutputs.qwen  ?? {};

  return (
    <div className="mt-6 border border-[#4A7FA7]/30 rounded-xl overflow-hidden">
      <button
        className="w-full flex items-center justify-between p-4 bg-[#1A3D63] hover:bg-[#234d7a] transition-colors"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-[#4A7FA7] text-[18px]">database</span>
          <span className="text-xs font-semibold uppercase tracking-widest text-[#c2c7cc]">
            View Model Evidence
          </span>
        </div>
        <span
          className="material-symbols-outlined text-[#c2c7cc] transition-transform duration-200"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        >
          expand_more
        </span>
      </button>

      {open && (
        <div className="bg-[#050d1a] p-4 space-y-4 divide-y divide-[#4A7FA7]/20 max-h-[380px] overflow-y-auto">
          {/* CLIP */}
          <div className="flex justify-between items-start pt-2">
            <div className="space-y-1">
              <p className="text-[#4A7FA7] text-xs uppercase font-semibold tracking-wider" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                CLIP-ViT-L/14
              </p>
              {clip.error
                ? <p className="text-red-400 text-sm italic">Failed — {clip.error}</p>
                : <p className="text-[#c2c7cc] text-sm">{clip.prediction ?? "—"}</p>
              }
            </div>
            <p className="text-[#dfe3e6] text-xs shrink-0 ml-4" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
              {clip.confidence != null ? `${clip.confidence}% CONF` : "—"}
            </p>
          </div>

          {/* BLIP-2 */}
          <div className="flex flex-col gap-1 pt-4">
            <p className="text-[#4A7FA7] text-xs uppercase font-semibold tracking-wider" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
              BLIP-2 Caption
            </p>
            {blip2.error
              ? <p className="text-red-400 text-sm italic">Failed — {blip2.error}</p>
              : <p className="text-[#c2c7cc] text-sm italic">{blip2.caption ? `"${blip2.caption}"` : "—"}</p>
            }
          </div>

          {/* LLaVA */}
          <div className="flex flex-col gap-1 pt-4">
            <p className="text-[#4A7FA7] text-xs uppercase font-semibold tracking-wider" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
              LLaVA Reasoning
            </p>
            {llava.error
              ? <p className="text-red-400 text-sm italic">Failed — {llava.error}</p>
              : <p className="text-[#c2c7cc] text-sm">{llava.response ? llava.response.split(".").slice(0, 2).join(".") + "." : "—"}</p>
            }
          </div>

          {/* Qwen */}
          <div className="flex flex-col gap-1 pt-4">
            <p className="text-[#4A7FA7] text-xs uppercase font-semibold tracking-wider" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
              Qwen2-VL Analysis
            </p>
            {qwen.error
              ? <p className="text-red-400 text-sm italic">Failed — {qwen.error}</p>
              : <p className="text-[#c2c7cc] text-sm">{qwen.response ? qwen.response.split(".")[0] + "." : "—"}</p>
            }
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
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
  const [copiedId,      setCopiedId]      = useState(null);

  const fileInputRef = useRef(null);
  const chatEndRef   = useRef(null);
  const inputRef     = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, isTyping]);

  // ── File handling ──────────────────────────────────────────────────────────

  const handleFile = useCallback((f) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setFileError("Unsupported format — please upload a JPEG, PNG, or WebP image.");
      return;
    }
    if (f.size > MAX_FILE_SIZE) {
      setFileError(`File too large — ${(f.size / 1024 / 1024).toFixed(1)} MB exceeds the ${MAX_FILE_SIZE_MB} MB limit.`);
      return;
    }
    setFileError(null);
    setFile(f);
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

  // ── Reset ──────────────────────────────────────────────────────────────────

  const resetToUpload = () => {
    setPhase("upload");
    setFile(null);
    setPreviewUrl(null);
    setModelOutputs({});
    setModelStatus({});
    setDisasterCtx(null);
    setChatHistory([]);
    setInputValue("");
    setFileError(null);
    setAnalysisError(null);
    setTimeline([]);
  };

  // ── Analysis ───────────────────────────────────────────────────────────────

  const handleAnalyze = async () => {
    if (!file) {
      setFileError("Please select an image to analyze.");
      setTimeout(() => setFileError(null), 3000);
      return;
    }

    setPhase("analyzing");
    setModelOutputs({});
    setAnalysisError(null);
    setDisasterCtx(null);
    setChatHistory([]);
    setTimeline([{ id: 0, text: "Image loaded — preparing analysis" }]);
    setModelStatus(MODELS.reduce((a, m) => ({ ...a, [m.key]: "waiting" }), {}));

    const results = await Promise.all(
      MODELS.map(async (model, idx) => {
        await new Promise((r) => setTimeout(r, idx * 120));
        setModelStatus((prev) => ({ ...prev, [model.key]: "running" }));
        try {
          const data = await callModel(model.endpoint, file);
          setModelOutputs((prev) => ({ ...prev, [model.key]: data }));
          setModelStatus((prev) => ({ ...prev, [model.key]: "complete" }));
          setTimeline((prev) => [...prev, { id: Date.now() + Math.random(), text: MODEL_TIMELINE_LABEL[model.key] }]);
          return { key: model.key, data };
        } catch (err) {
          const errData = { error: err.message };
          setModelOutputs((prev) => ({ ...prev, [model.key]: errData }));
          setModelStatus((prev) => ({ ...prev, [model.key]: "failed" }));
          setTimeline((prev) => [...prev, { id: Date.now() + Math.random(), text: `${model.name} failed — continuing with remaining models` }]);
          return { key: model.key, data: errData };
        }
      })
    );

    const successCount = results.filter((r) => !r.data.error).length;
    if (successCount === 0) {
      setAnalysisError("All models failed to respond. Verify your network connection and that the backend is running.");
      setTimeline((prev) => [...prev, { id: Date.now(), text: "Analysis failed — check backend connection" }]);
      setTimeout(() => { setPhase("upload"); setAnalysisError(null); }, 5000);
      return;
    }

    setTimeline((prev) => [...prev, { id: Date.now(), text: "Synthesizing intelligence briefing" }]);

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

  // ── Shared top navigation ──────────────────────────────────────────────────

  const TopNav = (
    <header className="fixed top-0 left-0 right-0 z-50 bg-[#0A1931]/80 backdrop-blur-md border-b border-[#4A7FA7]/20">
      <div className="flex justify-between items-center w-full px-12 py-4 max-w-[1200px] mx-auto">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-[#B3CFE5] rounded flex items-center justify-center">
            <span
              className="material-symbols-outlined text-[#0A1931] text-[18px]"
              style={{ fontVariationSettings: "'FILL' 1" }}
            >
              tsunami
            </span>
          </div>
          <h1 className="text-[24px] font-semibold text-[#dfe3e6]" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
            Disaster Intelligence Assistant
          </h1>
        </div>
        <div className="flex items-center gap-4">
          {phase === "ready" && (
            <button
              onClick={resetToUpload}
              className="text-xs font-semibold text-[#c2c7cc] hover:text-[#B3CFE5] border border-[#4A7FA7]/40 hover:border-[#B3CFE5]/50 rounded-lg px-3 py-1.5 transition-colors"
            >
              ↩ New Analysis
            </button>
          )}
          <button className="text-[#c2c7cc] hover:text-[#B3CFE5] transition-colors p-2">
            <span className="material-symbols-outlined">help</span>
          </button>
          <button className="text-[#c2c7cc] hover:text-[#B3CFE5] transition-colors p-2">
            <span className="material-symbols-outlined">settings</span>
          </button>
        </div>
      </div>
    </header>
  );

  // ---------------------------------------------------------------------------
  // PHASE: upload
  // ---------------------------------------------------------------------------

  if (phase === "upload") {
    return (
      <div className="min-h-screen flex flex-col bg-[#0A1931]">
        {TopNav}

        <main className="flex-grow flex items-center justify-center px-4 md:px-12 pt-28 pb-12 relative overflow-hidden">
          {/* Atmospheric gradients */}
          <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10 opacity-20">
            <div className="absolute -top-[10%] -right-[5%] w-[50%] h-[50%] bg-[#B3CFE5]/20 blur-[140px] rounded-full" />
            <div className="absolute -bottom-[10%] -left-[5%] w-[40%] h-[40%] bg-[#4A7FA7]/15 blur-[120px] rounded-full" />
          </div>

          <div className="max-w-[800px] w-full flex flex-col items-center text-center z-10">

            {/* Hero */}
            <div className="mb-8">
              <h1
                className="text-[40px] leading-[48px] font-semibold tracking-tight text-[#dfe3e6] mb-3"
                style={{ fontFamily: "'Hanken Grotesk', sans-serif", letterSpacing: "-0.02em" }}
              >
                Disaster Intelligence Assistant
              </h1>
              <p className="text-[18px] leading-[28px] text-[#c2c7cc] max-w-[600px] mx-auto">
                Upload a disaster image and receive an AI-generated intelligence briefing. Research-grade analysis for environmental crises.
              </p>
            </div>

            {/* Drop zone */}
            <div className="w-full mb-6 group">
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
                className={`upload-dashed rounded-xl min-h-[300px] flex flex-col items-center justify-center p-6
                  cursor-pointer transition-all duration-300
                  ${isDragging ? "upload-dashed-active bg-[#234d7a]/60 scale-[1.01]" : "bg-[#1A3D63] hover:bg-[#234d7a]/80"}`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => handleFile(e.target.files[0])}
                />

                {previewUrl ? (
                  /* ── File selected ── */
                  <div className="space-y-3 w-full flex flex-col items-center" onClick={(e) => e.stopPropagation()}>
                    <div className="relative inline-block">
                      <img
                        src={previewUrl}
                        alt="preview"
                        className="max-h-52 max-w-full rounded-xl object-contain shadow-xl"
                        onClick={() => fileInputRef.current?.click()}
                        style={{ cursor: "pointer" }}
                      />
                      {/* Remove button */}
                      <button
                        onClick={clearImage}
                        className="absolute top-2 right-2 w-7 h-7 rounded-full bg-[#050d1a]/80 backdrop-blur-sm flex items-center justify-center hover:bg-[#234d7a] transition-colors border border-[#4A7FA7]/40"
                        title="Remove image"
                      >
                        <span className="material-symbols-outlined text-[#c2c7cc]" style={{ fontSize: "14px" }}>close</span>
                      </button>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-[#c2c7cc]">
                      <span className="material-symbols-outlined text-[#4A7FA7]" style={{ fontSize: "14px" }}>image</span>
                      <span className="max-w-[220px] truncate">{file?.name}</span>
                      <span className="text-[#4A7FA7]">·</span>
                      <span className="text-[#c2c7cc]/60 shrink-0">{file ? (file.size / 1024 / 1024).toFixed(1) + " MB" : ""}</span>
                      <span className="text-[#4A7FA7]">·</span>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="text-[#B3CFE5] hover:underline shrink-0"
                      >
                        replace
                      </button>
                    </div>
                  </div>
                ) : (
                  /* ── Empty state ── */
                  <div className="flex flex-col items-center space-y-4">
                    <div className="w-16 h-16 rounded-full bg-[#2c5685] flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                      <span
                        className="material-symbols-outlined text-[#B3CFE5] text-4xl"
                        style={{ fontVariationSettings: "'FILL' 0, 'wght' 200" }}
                      >
                        upload_file
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      <h3
                        className="text-[24px] font-medium text-[#dfe3e6]"
                        style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}
                      >
                        Drop image here
                      </h3>
                      <p className="text-[#c2c7cc] text-sm uppercase tracking-widest opacity-70">
                        or click to browse local files
                      </p>
                    </div>
                    <div className="flex gap-4 mt-4">
                      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#050d1a] border border-[#4A7FA7]/30">
                        <span className="material-symbols-outlined text-[#c2c7cc]" style={{ fontSize: "16px" }}>satellite_alt</span>
                        <span className="text-[12px] font-semibold text-[#c2c7cc]">Satellite Imagery</span>
                      </div>
                      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#050d1a] border border-[#4A7FA7]/30">
                        <span className="material-symbols-outlined text-[#c2c7cc]" style={{ fontSize: "16px" }}>camera_indoor</span>
                        <span className="text-[12px] font-semibold text-[#c2c7cc]">Field Capture</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* File error / validation message */}
            <div className="w-full mb-4 min-h-[24px] flex items-center justify-center">
              {fileError && (
                <p className="text-red-300 text-sm flex items-center gap-2">
                  <span className="material-symbols-outlined text-red-400" style={{ fontSize: "16px" }}>warning</span>
                  {fileError}
                </p>
              )}
            </div>

            {/* Analyze button */}
            <div className="w-full flex flex-col items-center gap-4">
              <button
                onClick={handleAnalyze}
                className={`px-10 py-4 font-semibold text-xl rounded-lg shadow-lg flex items-center gap-3 transition-all duration-200
                  ${fileError && !file
                    ? "bg-red-500/20 text-red-300 border border-red-500/40"
                    : "bg-[#B3CFE5] text-[#0A1931] hover:shadow-[#B3CFE5]/20 hover:scale-[1.02] active:scale-[0.98]"}`}
              >
                <span className="material-symbols-outlined">analytics</span>
                Analyze Image
              </button>
              <p className="text-[13px] text-[#c2c7cc]/60 flex items-center gap-2 font-medium">
                <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>verified_user</span>
                Multi-Model Vision Pipeline · CLIP · BLIP-2 · LLaVA · Qwen2-VL
              </p>
            </div>
          </div>

          {/* Decorative corner panels */}
          <div className="hidden lg:block absolute bottom-12 left-12">
            <div className="p-4 border-l-2 border-[#B3CFE5] bg-[#050d1a]/40 rounded-r-lg">
              <div className="text-[11px] font-bold text-[#B3CFE5] mb-1 tracking-wider uppercase" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                SYSTEM STATUS
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-[#B3CFE5] animate-pulse" />
                <span className="text-[13px] font-medium text-[#dfe3e6]">Intelligence Grid Online</span>
              </div>
            </div>
          </div>
          <div className="hidden lg:block absolute bottom-12 right-12 text-right">
            <div className="p-4 border-r-2 border-[#B3CFE5] bg-[#050d1a]/40 rounded-l-lg">
              <div className="text-[11px] font-bold text-[#B3CFE5] mb-1 tracking-wider uppercase" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                DATA PROCESSING
              </div>
              <div className="text-[13px] font-medium text-[#dfe3e6]">4 VLMs · Parallel Inference</div>
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
    const allDone = Object.values(modelStatus).length > 0 &&
      Object.values(modelStatus).every((s) => s === "complete" || s === "failed");

    return (
      <div className="min-h-screen flex flex-col bg-[#0A1931]">
        {TopNav}

        <main className="flex-grow flex items-center justify-center px-4 pt-28 pb-12">
          <div className="max-w-[600px] w-full space-y-5">

            {/* Image context strip */}
            {previewUrl && (
              <div className="flex items-center gap-4 bg-[#1A3D63] p-4 rounded-xl border border-[#4A7FA7]/30">
                <img src={previewUrl} alt="Analyzing" className="w-20 h-16 rounded-xl object-cover shrink-0" />
                <div>
                  <p className="text-[#dfe3e6] font-semibold" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                    {analysisError ? "Analysis failed" : "Analyzing your image…"}
                  </p>
                  <p className="text-[#c2c7cc] text-sm mt-0.5">
                    {analysisError ? "Returning to upload in 5 seconds" : `Running ${MODELS.length} vision models in parallel`}
                  </p>
                </div>
              </div>
            )}

            {/* All-models-failed error */}
            {analysisError && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-start gap-3">
                <span className="material-symbols-outlined text-red-400 shrink-0 mt-0.5">wifi_off</span>
                <p className="text-red-300 text-sm leading-relaxed">{analysisError}</p>
              </div>
            )}

            {/* Model progress cards */}
            <div className="grid grid-cols-2 gap-3">
              {MODELS.map((m) => {
                const status = modelStatus[m.key] ?? "waiting";
                return (
                  <div key={m.key} className="bg-[#1A3D63] p-4 rounded-xl border border-[#4A7FA7]/30 flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${m.gradient} shrink-0 transition-opacity duration-300
                      ${status === "waiting" ? "opacity-25" : status === "running" ? "opacity-100 animate-pulse" : "opacity-100"}`}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-[#dfe3e6] text-xs font-semibold">{m.name}</p>
                      {status === "waiting" && (
                        <p className="text-[#c2c7cc]/40 text-xs flex items-center gap-1">
                          <span className="material-symbols-outlined" style={{ fontSize: "11px" }}>schedule</span>
                          Waiting
                        </p>
                      )}
                      {status === "running" && (
                        <p className="text-[#B3CFE5] text-xs animate-pulse">Analyzing…</p>
                      )}
                      {status === "complete" && (
                        <p className="text-emerald-400 text-xs flex items-center gap-1">
                          <span className="material-symbols-outlined" style={{ fontSize: "11px", fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                          Complete
                        </p>
                      )}
                      {status === "failed" && (
                        <p className="text-red-400 text-xs flex items-center gap-1">
                          <span className="material-symbols-outlined" style={{ fontSize: "11px" }}>error</span>
                          Failed
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Analysis timeline */}
            {timeline.length > 0 && (
              <div className="bg-[#050d1a]/60 rounded-xl border border-[#4A7FA7]/20 p-4 space-y-2">
                {timeline.map((event) => (
                  <div key={event.id} className="flex items-center gap-2 text-xs text-[#c2c7cc]/70 timeline-enter">
                    <span
                      className="material-symbols-outlined text-emerald-400/80 shrink-0"
                      style={{ fontSize: "12px", fontVariationSettings: "'FILL' 1" }}
                    >
                      check_circle
                    </span>
                    <span>{event.text}</span>
                  </div>
                ))}
                {!analysisError && !allDone && (
                  <div className="flex items-center gap-2 text-xs text-[#B3CFE5]/60">
                    <div className="w-3 h-3 shrink-0 flex items-center justify-center">
                      <div className="w-2 h-2 rounded-full bg-[#B3CFE5]/60 animate-pulse" />
                    </div>
                    <span>Running vision model analysis…</span>
                  </div>
                )}
                {!analysisError && allDone && (
                  <div className="flex items-center gap-2 text-xs text-[#B3CFE5]/60">
                    <div className="w-3 h-3 shrink-0 flex items-center justify-center">
                      <div className="w-2 h-2 rounded-full bg-[#B3CFE5]/60 animate-pulse" />
                    </div>
                    <span>Building intelligence briefing…</span>
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
    <div className="h-screen overflow-hidden flex flex-col bg-[#0A1931]">
      {TopNav}

      <div className="flex-1 pt-20 flex overflow-hidden relative">

        {/* ── Left sidebar ─────────────────────────────────────────────────── */}
        <aside className="hidden md:flex fixed left-0 top-20 bottom-0 w-[320px] bg-[#1A3D63] border-r border-[#4A7FA7]/30 p-6 flex-col gap-6 z-40">

          <h3
            className="text-xs font-semibold text-[#dfe3e6] uppercase tracking-widest"
            style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}
          >
            Reference Context
          </h3>

          {/* Scene image */}
          <div className="rounded-xl overflow-hidden border border-[#4A7FA7]/40 relative group">
            {previewUrl ? (
              <>
                <img
                  src={previewUrl}
                  alt="Analysed scene"
                  className="w-full aspect-video object-cover grayscale brightness-75 group-hover:grayscale-0 group-hover:brightness-100 transition-all duration-500"
                />
                <div
                  className="absolute bottom-2 left-2 bg-[#050d1a]/80 backdrop-blur-sm px-2 py-1 rounded text-[10px] uppercase text-[#B3CFE5]"
                  style={{ fontFamily: "'JetBrains Mono', monospace" }}
                >
                  REF_ID: {disasterCtx?.eventType?.slice(0, 3).toUpperCase() ?? "EVT"}_{String(disasterCtx?.confidence ?? 0).replace(".", "")}
                </div>
              </>
            ) : (
              <div className="w-full aspect-video bg-[#050d1a] flex items-center justify-center">
                <span className="material-symbols-outlined text-[#4A7FA7] text-4xl">image</span>
              </div>
            )}
          </div>

          {/* Disaster metrics */}
          <div className="space-y-3">
            <div className="flex justify-between items-center border-b border-[#4A7FA7]/30 pb-2">
              <span className="text-[#c2c7cc] text-sm font-semibold">Disaster Type</span>
              <span className="text-[#dfe3e6] text-sm font-semibold" style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}>
                {disasterCtx?.eventType ?? "—"}
              </span>
            </div>
            <div className="flex justify-between items-center border-b border-[#4A7FA7]/30 pb-2">
              <span className="text-[#c2c7cc] text-sm font-semibold">Confidence</span>
              <span className="text-[#B3CFE5] text-sm font-semibold" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                {disasterCtx?.confidence ?? 0}%
              </span>
            </div>
            <div className="flex justify-between items-center border-b border-[#4A7FA7]/30 pb-2">
              <span className="text-[#c2c7cc] text-sm font-semibold">Severity</span>
              <span className={`px-2 py-0.5 rounded text-[11px] font-bold uppercase border ${severityChipClass(disasterCtx?.severity)}`}>
                {disasterCtx?.severity ?? "—"}
              </span>
            </div>
          </div>

          {/* System node status */}
          <div className="mt-auto">
            <div className="bg-[#050d1a] p-4 rounded-xl border border-[#4A7FA7]/30">
              <p className="text-[11px] text-[#c2c7cc] leading-relaxed" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                <span className="text-[#B3CFE5]">SYSTEM_NODE:</span>{" "}
                Ready for multi-modal reasoning. Evidence extraction complete. Contextual metadata latched.
              </p>
            </div>
          </div>
        </aside>

        {/* ── Main chat area ────────────────────────────────────────────────── */}
        <section className="flex-1 flex flex-col md:ml-[320px] chat-container overflow-y-auto pb-32">
          <div className="w-full max-w-[800px] mx-auto px-4 md:px-6 py-8 flex flex-col gap-8">

            {chatHistory.map((msg, msgIdx) => {

              /* ── Briefing message ── */
              if (msg.type === "briefing" && disasterCtx) {
                return (
                  <article key={msg.id} className="flex gap-4 items-start message-enter">
                    <div className="w-8 h-8 rounded-full bg-[#B3CFE5]/20 flex items-center justify-center shrink-0 mt-1">
                      <span
                        className="material-symbols-outlined text-[#B3CFE5]"
                        style={{ fontSize: "18px", fontVariationSettings: "'FILL' 1" }}
                      >
                        analytics
                      </span>
                    </div>

                    <div className="flex-1 space-y-5 pt-1">
                      <div className="space-y-2">
                        <h2
                          className="text-[#dfe3e6] text-2xl font-semibold"
                          style={{ fontFamily: "'Hanken Grotesk', sans-serif" }}
                        >
                          Analysis Complete
                        </h2>
                        <p className="text-[#c2c7cc] text-[18px] leading-[28px]">{msg.content}</p>
                      </div>

                      {/* Row 1: Risks | Actions */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-[#1A3D63] p-4 rounded-xl border border-[#4A7FA7]/30">
                          <h4 className="text-[#B3CFE5] text-xs font-semibold uppercase tracking-widest mb-3">
                            Potential Risks
                          </h4>
                          <ul className="text-[#c2c7cc] text-sm space-y-2">
                            {disasterCtx.impacts.map((item, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="w-1.5 h-1.5 bg-[#B3CFE5] rounded-full mt-[5px] shrink-0" />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div className="bg-[#1A3D63] p-4 rounded-xl border border-[#4A7FA7]/30">
                          <h4 className="text-[#4A7FA7] text-xs font-semibold uppercase tracking-widest mb-3">
                            Recommended Actions
                          </h4>
                          <ul className="text-[#c2c7cc] text-sm space-y-2">
                            {disasterCtx.actions.map((item, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="w-1.5 h-1.5 bg-[#4A7FA7] rounded-full mt-[5px] shrink-0" />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      {/* Row 2: Infrastructure | Human Impact */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-[#1A3D63] p-4 rounded-xl border border-[#4A7FA7]/30">
                          <h4 className="text-[#c2c7cc]/70 text-xs font-semibold uppercase tracking-widest mb-3">
                            Affected Infrastructure
                          </h4>
                          <ul className="text-[#c2c7cc] text-sm space-y-2">
                            {disasterCtx.infrastructure.map((item, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="w-1.5 h-1.5 bg-[#c2c7cc]/40 rounded-full mt-[5px] shrink-0" />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div className="bg-[#1A3D63] p-4 rounded-xl border border-[#4A7FA7]/30">
                          <h4 className="text-[#c2c7cc]/70 text-xs font-semibold uppercase tracking-widest mb-3">
                            Human Impact
                          </h4>
                          <ul className="text-[#c2c7cc] text-sm space-y-2">
                            {disasterCtx.humanImpact.map((item, i) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="w-1.5 h-1.5 bg-[#c2c7cc]/40 rounded-full mt-[5px] shrink-0" />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      {/* Row 3: Environmental Impact — full width */}
                      <div className="bg-[#1A3D63] p-4 rounded-xl border border-[#4A7FA7]/30">
                        <h4 className="text-[#c2c7cc]/70 text-xs font-semibold uppercase tracking-widest mb-3">
                          Environmental Impact
                        </h4>
                        <ul className="text-[#c2c7cc] text-sm grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2">
                          {disasterCtx.environmentalImpact.map((item, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="w-1.5 h-1.5 bg-[#c2c7cc]/40 rounded-full mt-[5px] shrink-0" />
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Dynamic suggested question chips */}
                      {chatHistory.length === 1 && (
                        <div className="flex flex-wrap gap-2 pt-2">
                          {getSuggestedQuestions(disasterCtx.eventType).map((q) => (
                            <button
                              key={q}
                              onClick={() => handleChat(q)}
                              className="bg-[#234d7a] hover:bg-[#4A7FA7]/30 transition-all text-[#dfe3e6] px-4 py-2 rounded-full text-xs border border-[#4A7FA7]/30 hover:border-[#4A7FA7]/60"
                            >
                              {q}
                            </button>
                          ))}
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
                      <div className="bg-[#234d7a] border border-[#4A7FA7]/30 px-4 py-3 rounded-xl text-[#dfe3e6] text-sm leading-relaxed">
                        {msg.content}
                      </div>
                      <p className="text-xs text-[#4A7FA7]/40 mt-1 text-right">{msg.time}</p>
                    </div>
                  </div>
                );
              }

              /* ── Assistant follow-up message ── */
              const isLastAsst = msgIdx === lastAsstNonBriefingIdx;
              return (
                <article key={msg.id} className="flex gap-4 items-start group message-enter">
                  <div className="w-8 h-8 rounded-full bg-[#B3CFE5]/20 flex items-center justify-center shrink-0 mt-1">
                    <span
                      className="material-symbols-outlined text-[#B3CFE5]"
                      style={{ fontSize: "18px", fontVariationSettings: "'FILL' 1" }}
                    >
                      analytics
                    </span>
                  </div>
                  <div className="flex-1 pt-1">
                    <p className="text-[#c2c7cc] text-[16px] leading-[24px] whitespace-pre-line">{msg.content}</p>
                    <div className="flex items-center gap-3 mt-2 flex-wrap">
                      <p className="text-xs text-[#4A7FA7]/50">{msg.time}</p>
                      {msg.isFallback && (
                        <p className="text-xs text-[#4A7FA7]/40 flex items-center gap-1">
                          <span className="material-symbols-outlined" style={{ fontSize: "11px" }}>wifi_off</span>
                          Local response — backend unavailable
                        </p>
                      )}
                      {/* Copy button */}
                      <button
                        onClick={() => handleCopy(msg.id, msg.content)}
                        className="opacity-40 group-hover:opacity-100 transition-opacity duration-150 text-[#4A7FA7] hover:text-[#B3CFE5] flex items-center gap-1"
                        title="Copy response"
                      >
                        <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>
                          {copiedId === msg.id ? "check" : "content_copy"}
                        </span>
                        {copiedId === msg.id && <span className="text-xs text-emerald-400">Copied</span>}
                      </button>
                      {/* Regenerate button — only on last assistant message */}
                      {isLastAsst && !isTyping && (
                        <button
                          onClick={handleRegenerate}
                          className="opacity-40 group-hover:opacity-100 transition-opacity duration-150 text-[#4A7FA7] hover:text-[#B3CFE5] flex items-center gap-1"
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
                <div className="w-8 h-8 rounded-full bg-[#B3CFE5]/20 flex items-center justify-center shrink-0 mt-1">
                  <span
                    className="material-symbols-outlined text-[#B3CFE5]"
                    style={{ fontSize: "18px", fontVariationSettings: "'FILL' 1" }}
                  >
                    analytics
                  </span>
                </div>
                <div className="pt-3 flex items-center gap-1.5">
                  {[0, 150, 300].map((d) => (
                    <span
                      key={d}
                      className="w-2 h-2 rounded-full bg-[#4A7FA7] animate-bounce"
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
        className="fixed bottom-0 left-0 right-0 md:left-[320px] flex justify-center px-4 md:px-6 pt-4 z-50"
        style={{ paddingBottom: "max(1.5rem, env(safe-area-inset-bottom))" }}
      >
        <div className="w-full max-w-[800px] bg-[#234d7a]/90 backdrop-blur-md rounded-full border border-[#4A7FA7]/50 shadow-2xl flex items-center px-4 md:px-6 py-2 focus-within:ring-1 focus-within:ring-[#B3CFE5]/50 transition-all">
          <button className="text-[#c2c7cc] hover:text-[#4A7FA7] p-2 transition-colors hidden sm:flex">
            <span className="material-symbols-outlined">attachment</span>
          </button>
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleChat()}
            placeholder="Ask about severity, risks, or deployment…"
            disabled={isTyping}
            className="flex-1 bg-transparent border-none focus:ring-0 text-[#dfe3e6] placeholder-[#c2c7cc]/50 px-3 md:px-4 py-2 text-sm outline-none disabled:cursor-not-allowed"
          />
          <div className="flex items-center gap-1 md:gap-2">
            <button className="text-[#c2c7cc] hover:text-[#4A7FA7] p-2 transition-colors hidden sm:flex">
              <span className="material-symbols-outlined">mic</span>
            </button>
            <button
              onClick={() => handleChat()}
              disabled={!inputValue.trim() || isTyping}
              className="w-9 h-9 md:w-10 md:h-10 rounded-full bg-[#B3CFE5] text-[#0A1931] flex items-center justify-center hover:brightness-110 transition-all shadow-lg disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>
                arrow_upward
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Atmospheric gradients */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10 opacity-20">
        <div className="absolute -top-[10%] -right-[5%] w-[50%] h-[50%] bg-[#B3CFE5]/20 blur-[140px] rounded-full" />
        <div className="absolute -bottom-[10%] -left-[5%] w-[40%] h-[40%] bg-[#4A7FA7]/15 blur-[120px] rounded-full" />
      </div>
    </div>
  );
}
