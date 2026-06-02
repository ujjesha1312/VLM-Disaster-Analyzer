import { useState, useRef, useCallback, useEffect } from "react";

// ---------------------------------------------------------------------------
// Backend base URL — update when the ngrok tunnel changes
// ---------------------------------------------------------------------------

const API_BASE_URL = "https://providing-earthy-phonebook.ngrok-free.dev";

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

const MODELS = [
  { key: "clip",  name: "CLIP",     subtitle: "Classification",  endpoint: "/predict/clip",  gradient: "from-blue-500 to-cyan-500"    },
  { key: "blip2", name: "BLIP-2",   subtitle: "Caption",         endpoint: "/predict/blip2", gradient: "from-violet-500 to-purple-500" },
  { key: "llava", name: "LLaVA",    subtitle: "Reasoning",       endpoint: "/predict/llava", gradient: "from-emerald-500 to-teal-500"  },
  { key: "qwen",  name: "Qwen2-VL", subtitle: "Scene Analysis",  endpoint: "/predict/qwen",  gradient: "from-orange-500 to-amber-500"  },
];

// ---------------------------------------------------------------------------
// Disaster knowledge base
// ---------------------------------------------------------------------------

const IMPACTS = {
  Flood:      ["Road and transportation infrastructure damage", "Residential and commercial property flooding", "Contamination of water supply systems", "Population displacement", "Agricultural and soil damage"],
  Fire:       ["Widespread vegetation and forest destruction", "Air quality degradation from smoke and particulates", "Wildlife habitat and biodiversity loss", "Structural damage to nearby buildings", "Waterway contamination from ash and debris"],
  Earthquake: ["Structural collapse of buildings and bridges", "Utility disruption — power, gas, water", "Secondary landslide and aftershock risk", "Population displacement and potential casualties", "Long-term geological instability"],
  Landslide:  ["Transportation corridor blockage", "Burial or structural damage to nearby structures", "Drainage and water system disruption", "Secondary flooding risk in downstream areas", "Terrain and slope destabilisation"],
  Cyclone:    ["Wind and storm surge damage to coastal infrastructure", "Widespread power and communications outages", "Low-lying area flooding", "Agricultural destruction across the affected region", "Debris hazards for emergency responders"],
};

const ACTIONS = {
  Flood:      ["Deploy water rescue teams and inflatable vessels", "Establish elevated evacuation routes and shelters", "Coordinate drainage authorities for water level management", "Pre-position pumping equipment in critical zones", "Activate emergency broadcast for affected communities"],
  Fire:       ["Mobilise aerial and ground fire suppression units", "Establish firebreaks to contain spread", "Order evacuation within the defined fire perimeter", "Issue public health advisories for air quality", "Deploy medical units for respiratory and burn treatment"],
  Earthquake: ["Activate urban search-and-rescue operations", "Deploy structural engineers for building safety assessment", "Establish emergency medical triage centres", "Shut down and inspect gas and utility lines", "Issue aftershock warnings to the public"],
  Landslide:  ["Close all roads within the slide corridor", "Conduct geotechnical assessment for further slide risk", "Evacuate settlements in the debris flow path", "Deploy heavy machinery for access route clearance", "Install slope monitoring sensors on unstable terrain"],
  Cyclone:    ["Activate coastal evacuation protocols for vulnerable zones", "Pre-position emergency shelters and backup power units", "Suspend maritime and aviation operations", "Issue emergency broadcasts via all public channels", "Conduct rapid damage assessment after the storm passes"],
};

// ---------------------------------------------------------------------------
// Disaster theme system — drives ambient tone, accents, and avatar identity
// ---------------------------------------------------------------------------

const DISASTER_THEMES = {
  Fire: {
    glow1:          "bg-orange-600/12",
    glow2:          "bg-red-800/10",
    dotColor:       "bg-orange-400",
    accentText:     "text-orange-400",
    avatarRing:     "ring-orange-500/30",
    avatarGradient: "from-orange-400 to-red-500",
    nodeColor:      "bg-orange-400/55",
    briefingBorder: "border-orange-500/25",
    sidebarRing:    "ring-orange-500/20",
  },
  Flood: {
    glow1:          "bg-blue-600/12",
    glow2:          "bg-cyan-800/10",
    dotColor:       "bg-cyan-400",
    accentText:     "text-cyan-400",
    avatarRing:     "ring-cyan-500/30",
    avatarGradient: "from-blue-400 to-cyan-500",
    nodeColor:      "bg-cyan-400/55",
    briefingBorder: "border-cyan-500/25",
    sidebarRing:    "ring-cyan-500/20",
  },
  Earthquake: {
    glow1:          "bg-amber-600/10",
    glow2:          "bg-stone-700/10",
    dotColor:       "bg-amber-400",
    accentText:     "text-amber-400",
    avatarRing:     "ring-amber-500/30",
    avatarGradient: "from-amber-400 to-orange-500",
    nodeColor:      "bg-amber-400/55",
    briefingBorder: "border-amber-500/25",
    sidebarRing:    "ring-amber-500/20",
  },
  Landslide: {
    glow1:          "bg-yellow-700/10",
    glow2:          "bg-amber-900/10",
    dotColor:       "bg-yellow-500",
    accentText:     "text-yellow-500",
    avatarRing:     "ring-yellow-600/30",
    avatarGradient: "from-yellow-500 to-amber-600",
    nodeColor:      "bg-yellow-500/55",
    briefingBorder: "border-yellow-600/25",
    sidebarRing:    "ring-yellow-600/20",
  },
  Cyclone: {
    glow1:          "bg-violet-600/10",
    glow2:          "bg-indigo-800/10",
    dotColor:       "bg-violet-400",
    accentText:     "text-violet-400",
    avatarRing:     "ring-violet-500/30",
    avatarGradient: "from-violet-400 to-indigo-500",
    nodeColor:      "bg-violet-400/55",
    briefingBorder: "border-violet-500/25",
    sidebarRing:    "ring-violet-500/20",
  },
  Drought: {
    glow1:          "bg-yellow-500/10",
    glow2:          "bg-orange-800/8",
    dotColor:       "bg-yellow-400",
    accentText:     "text-yellow-400",
    avatarRing:     "ring-yellow-500/30",
    avatarGradient: "from-yellow-400 to-orange-400",
    nodeColor:      "bg-yellow-400/55",
    briefingBorder: "border-yellow-500/25",
    sidebarRing:    "ring-yellow-500/20",
  },
};

const DEFAULT_THEME = {
  glow1:          "bg-blue-600/8",
  glow2:          "bg-violet-600/8",
  dotColor:       "bg-blue-400",
  accentText:     "text-blue-400",
  avatarRing:     "ring-blue-500/30",
  avatarGradient: "from-blue-400 to-cyan-500",
  nodeColor:      "bg-blue-400/55",
  briefingBorder: "border-blue-500/25",
  sidebarRing:    "ring-blue-500/15",
};

function getTheme(eventType) {
  return DISASTER_THEMES[eventType] ?? DEFAULT_THEME;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function callModel(endpoint, file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE_URL}${endpoint}`, {
    method:  "POST",
    headers: { "ngrok-skip-browser-warning": "true" },
    body:    fd,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function callChat(question, context, history) {
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
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.response;
}

// ---------------------------------------------------------------------------
// DisasterContext builder — synthesises all four model outputs
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

  const severityColor =
    severity === "Critical" ? { bg: "bg-red-500/20",    text: "text-red-300",    border: "border-red-500/30"    } :
    severity === "High"     ? { bg: "bg-orange-500/20", text: "text-orange-300", border: "border-orange-500/30" } :
    severity === "Moderate" ? { bg: "bg-yellow-500/20", text: "text-yellow-300", border: "border-yellow-500/30" } :
                              { bg: "bg-slate-500/20",  text: "text-slate-300",  border: "border-slate-500/30"  };

  const impacts = IMPACTS[eventType] ?? ["Environmental and infrastructure damage", "Potential civilian impact", "Service disruption"];
  const actions = ACTIONS[eventType] ?? ["Deploy emergency response teams", "Establish incident command", "Prioritise civilian evacuation"];

  return { eventType, confidence, caption, reasoning, sceneAnalysis, severity, severityColor, impacts, actions };
}

// ---------------------------------------------------------------------------
// Opening message generator — the AI's first response after analysis
// ---------------------------------------------------------------------------

function generateOpeningMessage(ctx) {
  const { eventType, confidence, severity, caption, reasoning, impacts, actions } = ctx;

  const sceneHint = caption
    ? caption.charAt(0).toLowerCase() + caption.slice(1).replace(/\.$/, "")
    : reasoning
      ? reasoning.split(".")[0].toLowerCase()
      : `${eventType.toLowerCase()} conditions`;

  const impactList = impacts.slice(0, 4).map((i) => `• ${i}`).join("\n");
  const actionList = actions.slice(0, 4).map((a) => `• ${a}`).join("\n");

  return (
    `**Analysis Complete.**\n\n` +
    `The uploaded image appears to depict a **${eventType}** — ${sceneHint}.\n\n` +
    `**Estimated Severity:** ${severity} *(${confidence}% CLIP classification confidence)*\n\n` +
    `**Potential Impacts:**\n${impactList}\n\n` +
    `**Recommended Immediate Actions:**\n${actionList}\n\n` +
    `---\n` +
    `*You may now ask questions about this disaster event.*`
  );
}

// ---------------------------------------------------------------------------
// Markdown renderer — bold, bullets, italic, HR
// ---------------------------------------------------------------------------

function renderMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/---/g, "<hr class=\"border-slate-600/50 my-2\"/>")
    .replace(/\n/g, "<br/>");
}

// ---------------------------------------------------------------------------
// Typing animation
// ---------------------------------------------------------------------------

function TypingDots() {
  return (
    <div className="flex items-center gap-1 px-1 py-0.5">
      {[0, 160, 320].map((d) => (
        <span
          key={d}
          className="w-2 h-2 rounded-full bg-slate-400 animate-bounce"
          style={{ animationDelay: `${d}ms` }}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Intelligence avatar — central core + four cardinal satellite nodes.
// Each satellite represents one VLM feeding the synthesis engine.
// Geometry is intentional, not decorative.
// ---------------------------------------------------------------------------

function IntelligenceAvatar({ theme, size = "md" }) {
  const isLg = size === "lg";

  // Cardinal positions — complete static strings required for Tailwind JIT scanning.
  // N=CLIP  E=BLIP-2  S=LLaVA  W=Qwen
  const nPos = isLg ? "absolute top-[6px] left-1/2 -translate-x-1/2"    : "absolute top-[5px] left-1/2 -translate-x-1/2";
  const ePos = isLg ? "absolute right-[6px] top-1/2 -translate-y-1/2"   : "absolute right-[5px] top-1/2 -translate-y-1/2";
  const sPos = isLg ? "absolute bottom-[6px] left-1/2 -translate-x-1/2" : "absolute bottom-[5px] left-1/2 -translate-x-1/2";
  const wPos = isLg ? "absolute left-[6px] top-1/2 -translate-y-1/2"    : "absolute left-[5px] top-1/2 -translate-y-1/2";

  return (
    <div className={`${isLg ? "w-9 h-9" : "w-8 h-8"} rounded-full bg-slate-900/80
      ring-1 ${theme.avatarRing} flex items-center justify-center relative shrink-0`}>

      {/* Four satellite nodes — one per VLM, cardinal symmetry */}
      <div className={`${nPos} w-1 h-1 rounded-full ${theme.nodeColor}`} />
      <div className={`${ePos} w-1 h-1 rounded-full ${theme.nodeColor}`} />
      <div className={`${sPos} w-1 h-1 rounded-full ${theme.nodeColor}`} />
      <div className={`${wPos} w-1 h-1 rounded-full ${theme.nodeColor}`} />

      {/* Central intelligence core */}
      <div className={`${isLg ? "w-[14px] h-[14px]" : "w-3 h-3"} rounded-full
        bg-gradient-to-br ${theme.avatarGradient} relative z-10`} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  // ── Phase state machine ──────────────────────────────────────────────────
  const [phase,         setPhase]         = useState("upload");

  // ── File / preview ───────────────────────────────────────────────────────
  const [file,          setFile]          = useState(null);
  const [previewUrl,    setPreviewUrl]    = useState(null);
  const [isDragging,    setIsDragging]    = useState(false);

  // ── Model outputs ────────────────────────────────────────────────────────
  const [modelOutputs,  setModelOutputs]  = useState({});
  const [loadingModels, setLoadingModels] = useState({});

  // ── Synthesised context (drives the entire chat session) ─────────────────
  const [disasterCtx,   setDisasterCtx]   = useState(null);

  // ── Chat ─────────────────────────────────────────────────────────────────
  const [chatHistory,   setChatHistory]   = useState([]);
  const [inputValue,    setInputValue]    = useState("");
  const [isTyping,      setIsTyping]      = useState(false);

  // ── Streaming — true while the opening briefing is being revealed ─────────
  const [isStreaming,   setIsStreaming]   = useState(false);

  const fileInputRef      = useRef(null);
  const chatEndRef        = useRef(null);
  const inputRef          = useRef(null);
  const streamIntervalRef = useRef(null);

  // Active theme — derived from context, no extra state needed
  const theme = disasterCtx ? getTheme(disasterCtx.eventType) : DEFAULT_THEME;

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, isTyping]);

  useEffect(() => {
    return () => {
      if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
    };
  }, []);

  // ---------------------------------------------------------------------------
  // File handling
  // ---------------------------------------------------------------------------

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    if (phase !== "upload") {
      setPhase("upload");
      setModelOutputs({});
      setDisasterCtx(null);
      setChatHistory([]);
    }
  }, [phase]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  // ---------------------------------------------------------------------------
  // Shared reset — also stops any in-progress stream
  // ---------------------------------------------------------------------------

  const resetToUpload = () => {
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }
    setIsStreaming(false);
    setPhase("upload");
    setFile(null);
    setPreviewUrl(null);
    setModelOutputs({});
    setDisasterCtx(null);
    setChatHistory([]);
  };

  // ---------------------------------------------------------------------------
  // Multi-model analysis → DisasterContext → Streaming opening briefing
  // ---------------------------------------------------------------------------

  const handleAnalyze = async () => {
    if (!file || phase === "analyzing") return;

    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }

    setPhase("analyzing");
    setModelOutputs({});
    setDisasterCtx(null);
    setChatHistory([]);
    setIsStreaming(false);
    setLoadingModels(MODELS.reduce((a, m) => ({ ...a, [m.key]: true }), {}));

    const settled = await Promise.allSettled(
      MODELS.map(async (model) => {
        try {
          const data = await callModel(model.endpoint, file);
          setModelOutputs((prev) => ({ ...prev, [model.key]: data }));
          return { key: model.key, data };
        } catch (err) {
          const errObj = { error: err.message };
          setModelOutputs((prev) => ({ ...prev, [model.key]: errObj }));
          return { key: model.key, data: errObj };
        } finally {
          setLoadingModels((prev) => ({ ...prev, [model.key]: false }));
        }
      })
    );

    const outputs = Object.fromEntries(
      settled
        .filter((r) => r.status === "fulfilled")
        .map((r) => [r.value.key, r.value.data])
    );

    const ctx = buildDisasterContext(outputs);
    setDisasterCtx(ctx);

    const opening = generateOpeningMessage(ctx);
    const msgId   = Date.now();
    const msgTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    setChatHistory([{ id: msgId, role: "assistant", content: "", time: msgTime }]);
    setPhase("ready");
    setIsStreaming(true);

    let charIndex = 0;
    const CHARS_PER_TICK = 4;
    const TICK_MS        = 16;

    streamIntervalRef.current = setInterval(() => {
      charIndex += CHARS_PER_TICK;
      const visible = opening.slice(0, charIndex);
      setChatHistory([{ id: msgId, role: "assistant", content: visible, time: msgTime }]);

      if (charIndex >= opening.length) {
        clearInterval(streamIntervalRef.current);
        streamIntervalRef.current = null;
        setChatHistory([{ id: msgId, role: "assistant", content: opening, time: msgTime }]);
        setIsStreaming(false);
        setTimeout(() => inputRef.current?.focus(), 50);
      }
    }, TICK_MS);
  };

  // ---------------------------------------------------------------------------
  // Chat — gated while isStreaming is true
  // ---------------------------------------------------------------------------

  const handleChat = async (question) => {
    const q = (question ?? inputValue).trim();
    if (!q || isTyping || isStreaming || phase !== "ready") return;

    if (!question) setInputValue("");
    const userMsg = {
      id:      Date.now(),
      role:    "user",
      content: q,
      time:    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
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
        id:      Date.now() + 1,
        role:    "assistant",
        content: fallback,
        time:    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      }]);
    } finally {
      setIsTyping(false);
      inputRef.current?.focus();
    }
  };

  // ---------------------------------------------------------------------------
  // Client-side fallback
  // ---------------------------------------------------------------------------

  function buildFallbackResponse(question, ctx) {
    const q         = question.toLowerCase();
    const event     = ctx?.eventType      ?? "disaster";
    const conf      = ctx?.confidence     ?? 0;
    const severity  = ctx?.severity       ?? "Unknown";
    const reasoning = ctx?.reasoning      ?? "";
    const scene     = ctx?.sceneAnalysis  ?? "";

    if (/sever|how bad|intensity|danger/i.test(q))
      return `This **${event}** event is assessed as **${severity}** severity (${conf}% CLIP confidence). ${reasoning.split(".")[0] + "." || ""}`;

    if (/emergency|response|protocol|action|help/i.test(q)) {
      const list = (ACTIONS[event] ?? []).slice(0, 4).map((a) => `• ${a}`).join("\n");
      return `**Emergency response for ${severity.toLowerCase()} ${event}:**\n\n${list}`;
    }

    if (/people|risk|casualt|human|injur/i.test(q))
      return `Civilian risk for this **${event}** event is assessed as **${conf > 80 ? "HIGH" : "MODERATE"}**. Immediate evacuation and medical staging are recommended.`;

    if (/impact|environment|damage|infrastructure/i.test(q)) {
      const list = (IMPACTS[event] ?? []).slice(0, 4).map((i) => `• ${i}`).join("\n");
      return `**Identified impacts of this ${event}:**\n\n${list}`;
    }

    return `Regarding the **${event}** event (${conf}% confidence, ${severity} severity): ${scene.split(".")[0] + "." || reasoning.split(".")[0] + "." || ""} How else can I assist?`;
  }

  // ---------------------------------------------------------------------------
  // Suggested questions
  // ---------------------------------------------------------------------------

  const SUGGESTED = [
    "How severe is this disaster?",
    "What emergency response is needed?",
    "What infrastructure is affected?",
    "Are people at risk?",
    "What environmental impacts may occur?",
  ];

  // ---------------------------------------------------------------------------
  // Upload / Analyzing phase view
  // ---------------------------------------------------------------------------

  const UploadView = (
    <div className="max-w-2xl mx-auto space-y-6">
      {phase === "analyzing" ? (

        // ── Dedicated analyzing state ─────────────────────────────────────────
        <div className="space-y-5">
          {previewUrl && (
            <div className="flex items-center gap-4 glass-card rounded-2xl p-4">
              <img
                src={previewUrl}
                alt="Analyzing"
                className="w-20 h-16 rounded-xl object-cover shrink-0"
              />
              <div>
                <p className="text-white font-semibold">Analyzing your image…</p>
                <p className="text-slate-400 text-sm mt-0.5">
                  Running {MODELS.length} vision models in parallel
                </p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            {MODELS.map((m) => (
              <div key={m.key} className="glass-card rounded-xl p-4 flex items-center gap-3">
                {/* Color swatch replaces emoji icon */}
                <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${m.gradient} shrink-0`} />
                <div className="flex-1 min-w-0">
                  <p className="text-white text-xs font-semibold">{m.name}</p>
                  {loadingModels[m.key]
                    ? <p className="text-blue-400 text-xs animate-pulse">Analyzing…</p>
                    : modelOutputs[m.key]?.error
                      ? <p className="text-red-400 text-xs">Error</p>
                      : <p className="text-emerald-400 text-xs">Complete</p>
                  }
                </div>
              </div>
            ))}
          </div>
        </div>

      ) : (

        // ── Upload state ──────────────────────────────────────────────────────
        <>
          <div
            role="button" tabIndex={0} aria-label="Upload disaster image"
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300
              ${isDragging
                ? "border-blue-400 bg-blue-500/10 scale-[1.01]"
                : "border-slate-700 hover:border-slate-500 hover:bg-slate-800/30"}`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />
            {previewUrl ? (
              <div className="space-y-3">
                <img
                  src={previewUrl}
                  alt="preview"
                  className="max-h-64 mx-auto rounded-xl object-contain shadow-xl"
                />
                <p className="text-slate-400 text-sm">
                  {file?.name}
                  <span className="text-slate-600 mx-1">·</span>
                  <span className="text-blue-400">click to change</span>
                </p>
              </div>
            ) : (
              <div className="space-y-4 py-6">
                {/* Minimal upload indicator — no emoji */}
                <div className="mx-auto w-12 h-12 rounded-xl border border-slate-700/80 flex items-center justify-center bg-slate-900/60">
                  <svg className="w-6 h-6 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                </div>
                <div className="space-y-1.5">
                  <p className="text-slate-200 font-semibold text-lg">
                    Upload a disaster image to begin analysis
                  </p>
                  <p className="text-slate-500 text-sm">Drag and drop or click to browse</p>
                </div>
                <p className="text-slate-600 text-xs">
                  The AI will generate a full disaster briefing and answer your questions
                </p>
              </div>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!file}
            className={`w-full py-4 rounded-xl font-bold text-base tracking-wide transition-all duration-300 flex items-center justify-center gap-3
              ${!file
                ? "bg-slate-800/60 text-slate-600 cursor-not-allowed border border-slate-700/40"
                : "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 active:scale-[0.99]"}`}
          >
            Analyze with AI →
          </button>
        </>
      )}
    </div>
  );

  // ---------------------------------------------------------------------------
  // Ready phase — conversational intelligence view
  // ---------------------------------------------------------------------------

  const IntelligenceView = disasterCtx && (
    <div className="flex flex-col lg:flex-row gap-5 h-[calc(100vh-140px)] min-h-[500px]">

      {/* ── Left sidebar ─────────────────────────────────────────────────── */}
      <aside className="order-2 lg:order-1 lg:w-72 shrink-0 flex flex-col gap-4">

        {/* Intelligence card — severity-first hierarchy */}
        <div className={`glass-card rounded-2xl overflow-hidden ring-1 ${theme.sidebarRing}`}>

          {/* Scene image — flush, full width */}
          {previewUrl && (
            <img
              src={previewUrl}
              alt="Analysed scene"
              className="w-full object-cover max-h-44"
            />
          )}

          <div className="p-4 space-y-4">

            {/* Severity — primary visual element */}
            <div className={`rounded-xl px-4 py-3 border ${disasterCtx.severityColor.bg} ${disasterCtx.severityColor.border}`}>
              <p className={`text-2xl font-black tracking-tight ${disasterCtx.severityColor.text}`}>
                {disasterCtx.severity}
              </p>
              <p className="text-xs text-slate-500 mt-0.5 uppercase tracking-wider">Severity Level</p>
            </div>

            {/* Disaster type */}
            <div>
              <p className="text-white font-black text-xl leading-tight">{disasterCtx.eventType}</p>
              <p className="text-slate-500 text-xs uppercase tracking-wider mt-0.5">Disaster Type</p>
            </div>

            {/* Confidence */}
            <div>
              <div className="flex justify-between items-baseline mb-1.5">
                <p className="text-xs text-slate-500 uppercase tracking-wider">Classification Confidence</p>
                <p className="text-white text-sm font-bold">{disasterCtx.confidence}%</p>
              </div>
              <div className="w-full bg-slate-700/50 rounded-full h-1.5">
                <div
                  className="h-1.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-1000"
                  style={{ width: `${disasterCtx.confidence}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Intelligence Status — visually secondary */}
        <div className="glass-card rounded-2xl p-4">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            Intelligence Status
          </p>
          <div className="space-y-1.5">
            {MODELS.map((m) => {
              const r = modelOutputs[m.key];
              const isError = r?.error;
              return (
                <div key={m.key} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {/* Color swatch replaces emoji icon */}
                    <div className={`w-4 h-4 rounded bg-gradient-to-br ${m.gradient} shrink-0`} />
                    <span className="text-xs text-slate-400">{m.name}</span>
                  </div>
                  <span className={`text-xs font-medium ${isError ? "text-red-400" : "text-emerald-400/80"}`}>
                    {isError ? "Error" : "✓"}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </aside>

      {/* ── Main chat area ────────────────────────────────────────────────── */}
      <main className="order-1 lg:order-2 flex-1 flex flex-col glass-card rounded-2xl overflow-hidden min-h-0">

        {/* Chat header */}
        <header className="px-6 py-4 border-b border-slate-700/50 flex items-center gap-3 shrink-0">
          <IntelligenceAvatar theme={theme} size="lg" />
          <div className="flex-1 min-w-0">
            <p className="font-bold text-white leading-tight">Disaster Intelligence Assistant</p>
            <p className="text-slate-400 text-xs truncate">
              Context: <span className={theme.accentText}>{disasterCtx.eventType}</span> · {disasterCtx.severity} severity
            </p>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
            <span className="text-emerald-400 text-xs">Active</span>
          </div>
        </header>

        {/* Message list */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4 chat-scroll min-h-0">
          {chatHistory.map((msg, index) => {
            const isOpeningBriefing = index === 0 && msg.role === "assistant";
            return (
              <div
                key={msg.id}
                className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
              >
                {/* Avatar */}
                {msg.role === "user" ? (
                  <div className="w-8 h-8 rounded-full bg-slate-800 ring-1 ring-slate-700/60
                    flex items-center justify-center text-xs font-bold text-slate-400 shrink-0 mt-0.5">
                    U
                  </div>
                ) : (
                  <div className="mt-0.5">
                    <IntelligenceAvatar theme={theme} />
                  </div>
                )}

                {/* Bubble */}
                <div className={`space-y-1 flex flex-col
                  ${isOpeningBriefing ? "max-w-[90%]" : "max-w-[75%]"}
                  ${msg.role === "user" ? "items-end" : "items-start"}`}>

                  {/* Themed label above the opening briefing only */}
                  {isOpeningBriefing && (
                    <p className={`text-xs font-semibold uppercase tracking-widest ml-1 ${theme.accentText}`}>
                      Intelligence Briefing
                    </p>
                  )}

                  <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed
                    ${isOpeningBriefing
                      ? `bg-slate-800/90 text-slate-100 border ${theme.briefingBorder}`
                      : msg.role === "user"
                        ? "bg-blue-600 text-white rounded-br-sm"
                        : "bg-slate-700/60 text-slate-100 rounded-bl-sm border border-slate-600/30"}`}>
                    <p dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />
                    {/* Blinking cursor while streaming */}
                    {isOpeningBriefing && isStreaming && (
                      <span className="inline-block w-0.5 h-[1em] bg-slate-300 animate-pulse ml-0.5 align-middle" />
                    )}
                  </div>

                  <p className="text-xs text-slate-600 px-1">{msg.time}</p>
                </div>
              </div>
            );
          })}

          {/* Suggested questions — inside scroll, hidden during streaming */}
          {chatHistory.length === 1 && !isTyping && !isStreaming && (
            <div className="flex flex-wrap gap-2 ml-11 pt-1">
              {SUGGESTED.map((q) => (
                <button
                  key={q}
                  onClick={() => handleChat(q)}
                  className="text-xs bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/60 hover:border-slate-600/60 rounded-full px-3 py-1.5 text-slate-400 hover:text-slate-200 transition-all duration-200"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {/* Typing indicator */}
          {isTyping && (
            <div className="flex gap-3">
              <div className="mt-0.5">
                <IntelligenceAvatar theme={theme} />
              </div>
              <div className="bg-slate-700/60 rounded-2xl rounded-bl-sm border border-slate-600/30 px-3 py-2">
                <TypingDots />
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input bar */}
        <div className="px-5 py-4 border-t border-slate-700/50 flex gap-3 shrink-0">
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && !isStreaming && handleChat()}
            placeholder={isStreaming ? "Generating briefing…" : "Ask about this disaster event…"}
            disabled={isStreaming}
            className={`flex-1 border rounded-xl px-4 py-3 text-white outline-none transition-all text-sm
              ${isStreaming
                ? "bg-slate-900/40 border-slate-700/40 placeholder-slate-600 cursor-not-allowed"
                : "bg-slate-800/40 border-slate-700/50 hover:border-slate-600 focus:border-slate-500 focus:ring-1 focus:ring-slate-500/20 placeholder-slate-500"}`}
          />
          <button
            onClick={handleChat}
            disabled={!inputValue.trim() || isTyping || isStreaming}
            className="px-5 py-3 rounded-xl font-semibold text-sm transition-all duration-200 flex items-center gap-2
              bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500
              disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-600 disabled:cursor-not-allowed
              shadow-md shadow-blue-500/15 active:scale-95"
          >
            Send
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </main>
    </div>
  );

  // ---------------------------------------------------------------------------
  // Root layout
  // ---------------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-slate-950 text-white overflow-x-hidden">

      {/* Ambient background — colors driven by active disaster theme */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none" aria-hidden>
        <div className={`absolute -top-40 -right-40 w-96 h-96 ${theme.glow1} rounded-full blur-3xl transition-all duration-1000`} />
        <div className={`absolute -bottom-40 -left-40 w-96 h-96 ${theme.glow2} rounded-full blur-3xl transition-all duration-1000`} />
      </div>

      <div className={`relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 transition-all duration-300
        ${phase === "ready" ? "py-4 space-y-3" : "py-10 space-y-8"}`}>

        {/* ── Header ───────────────────────────────────────────────────────── */}
        {phase === "ready" ? (
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 ${theme.dotColor} rounded-full animate-pulse`} />
              <span className="text-white font-semibold text-sm">Disaster Intelligence</span>
            </div>
            <button
              onClick={resetToUpload}
              className="text-xs font-semibold text-slate-500 hover:text-white border border-slate-700/50 hover:border-slate-500 rounded-lg px-3 py-1.5 transition-colors"
            >
              ↩ New Analysis
            </button>
          </header>
        ) : (
          <header className="text-center space-y-4 pb-4">
            <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/20 rounded-full px-4 py-1.5">
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
              <span className="text-blue-300 text-sm font-medium">Multi-Model AI Research System</span>
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight bg-gradient-to-br from-white via-slate-200 to-cyan-300 bg-clip-text text-transparent">
              VLM Disaster Intelligence Assistant
            </h1>
            <p className="text-slate-400 text-base sm:text-lg max-w-xl mx-auto">
              Upload a disaster image to generate an AI intelligence briefing
            </p>
            <div className="max-w-2xl mx-auto bg-slate-800/50 border border-slate-700/50 rounded-xl px-6 py-4 text-left">
              <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
                Four vision language models analyse the same disaster image simultaneously — each through a different lens. The results are synthesised into a real-time intelligence briefing you can interrogate.
              </p>
            </div>
          </header>
        )}

        {/* ── Phase content ─────────────────────────────────────────────────── */}
        {phase !== "ready" ? UploadView : IntelligenceView}

        {/* ── Footer ───────────────────────────────────────────────────────── */}
        <footer className="text-center text-slate-700 text-xs pb-2">
          VLM Disaster Intelligence Assistant · CLIP · BLIP-2 · LLaVA · Qwen2-VL
        </footer>

      </div>
    </div>
  );
}
