import { useState, useRef, useCallback, useEffect } from "react";

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

const MODELS = [
  {
    key:      "clip",
    name:     "CLIP",
    subtitle: "Zero-shot Classification",
    endpoint: "/predict/clip",
    gradient: "from-blue-500 to-cyan-500",
    glow:     "shadow-blue-500/20",
    border:   "border-blue-500/20",
    icon:     "🎯",
  },
  {
    key:      "blip2",
    name:     "BLIP-2",
    subtitle: "Caption Generation",
    endpoint: "/predict/blip2",
    gradient: "from-violet-500 to-purple-500",
    glow:     "shadow-violet-500/20",
    border:   "border-violet-500/20",
    icon:     "✍️",
  },
  {
    key:      "llava",
    name:     "LLaVA",
    subtitle: "Visual Reasoning",
    endpoint: "/predict/llava",
    gradient: "from-emerald-500 to-teal-500",
    glow:     "shadow-emerald-500/20",
    border:   "border-emerald-500/20",
    icon:     "🧠",
  },
  {
    key:      "qwen",
    name:     "Qwen2-VL",
    subtitle: "Scene Understanding",
    endpoint: "/predict/qwen",
    gradient: "from-orange-500 to-amber-500",
    glow:     "shadow-orange-500/20",
    border:   "border-orange-500/20",
    icon:     "🔍",
  },
];

// ---------------------------------------------------------------------------
// API call helper — each model gets its own FormData instance
// ---------------------------------------------------------------------------

async function callModel(endpoint, file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(endpoint, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`HTTP ${res.status} — ${res.statusText}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Chat intelligence — context-aware responses built from VLM outputs
// ---------------------------------------------------------------------------

function buildChatResponse(question, results) {
  const q        = question.toLowerCase();
  const disaster = results.clip?.prediction ?? "disaster";
  const conf     = results.clip?.confidence ?? 0;
  const caption  = results.blip2?.caption   ?? "";
  const llavaOut = results.llava?.response  ?? "";
  const qwenOut  = results.qwen?.response   ?? "";
  const severity = conf > 88 ? "catastrophic" : conf > 75 ? "severe" : conf > 60 ? "moderate" : "uncertain";

  const protocols = {
    Flood:      "Deploy water rescue teams and inflatable boats. Establish elevated evacuation points. Coordinate drainage authorities and set up emergency shelters on higher ground.",
    Fire:       "Activate aerial and ground fire suppression units. Create defensive firebreaks. Evacuate within a 2 km radius. Pre-position medical teams for smoke inhalation.",
    Earthquake: "Launch urban search-and-rescue operations immediately. Deploy structural engineers for building safety assessments. Establish medical triage centres.",
    Landslide:  "Close all affected road corridors. Deploy geotechnical teams for slope stability assessment. Evacuate settlements within the slide path.",
    Cyclone:    "Activate coastal evacuation protocols. Pre-position emergency shelters and backup power. Conduct rapid damage assessment after the storm passes.",
  };

  const resources = {
    Flood:      "Water pumps, inflatable rescue boats, life vests, emergency generators, water purification units, temporary shelter materials.",
    Fire:       "Aerial tankers, ground fire crews, thermal imaging cameras, protective breathing equipment, medical oxygen.",
    Earthquake: "Heavy rescue equipment, K9 search units, structural shoring materials, medical trauma kits, satellite communication relays.",
    Landslide:  "Excavators, slope stabilisation equipment, geotechnical sensors, drainage clearing machinery.",
    Cyclone:    "Reinforced storm shelters, backup generators, potable water reserves, medical supplies, satellite comms.",
  };

  if (/sever|how bad|intensity|scale|serious|extent|danger/i.test(q))
    return `Based on multi-model analysis, this **${disaster}** event is **${severity}** (CLIP confidence: **${conf}%**). ${llavaOut ? llavaOut.split(".")[0] + "." : ""} ${qwenOut ? "Scene assessment: " + qwenOut.split(".")[0] + "." : ""}`;

  if (/emergency|response|action|what.*(do|should|to do)|help|protocol/i.test(q))
    return `**Emergency protocol for ${severity} ${disaster}:** ${protocols[disaster] ?? "Deploy multi-agency response teams, establish incident command, and prioritise civilian evacuation."} ${qwenOut.split(".")[0] ? "\n\nCurrent scene context: " + qwenOut.split(".")[0] + "." : ""}`;

  if (/people|risk|casualt|injur|life|human|death|survivor|victim/i.test(q))
    return `${llavaOut ? llavaOut.split(".")[0] + ". " : ""}Given the **${severity} ${disaster}** scenario, ${conf > 80 ? "there is a HIGH probability of civilian impact — immediate search and rescue should be prioritised." : "civilian risk assessment is ongoing."} ${caption ? "Visual evidence: " + caption + "." : ""}`;

  if (/resource|deploy|equipment|need|send|material|supply|asset/i.test(q))
    return `**Recommended resources for ${severity} ${disaster}:**\n\n${resources[disaster] ?? "Multi-agency emergency equipment, medical supplies, communication systems, and evacuation transport."}\n\nScale deployment against **${conf}% confidence** assessment.`;

  if (/infrastructure|road|build|damage|structur|bridge|power|utility|network/i.test(q))
    return `${qwenOut ? qwenOut.split(".").slice(0, 2).join(". ") + ". " : ""}${caption ? "BLIP-2 visual report: " + caption + ". " : ""}Priority infrastructure items: transportation corridors, power grid, water/gas utilities, and emergency service access routes.`;

  if (/clip|blip|llava|qwen|model|confidence|caption|predict/i.test(q))
    return `**All model outputs:**\n\n🎯 **CLIP** → ${disaster} (${conf}%)\n✍️ **BLIP-2** → "${caption || "—"}"\n🧠 **LLaVA** → ${llavaOut ? llavaOut.split(".")[0] + "." : "—"}\n🔍 **Qwen2-VL** → ${qwenOut ? qwenOut.split(".")[0] + "." : "—"}`;

  if (/what|describe|tell|explain|show|see|image|picture|detect/i.test(q))
    return `This image shows a **${disaster}** event (${conf}% CLIP confidence). ${caption ? "**BLIP-2** describes: " + caption + ". " : ""}${llavaOut ? "**LLaVA** analysis: " + llavaOut.split(".")[0] + "." : ""}`;

  return `Regarding this **${disaster}** event (${conf}% confidence): ${llavaOut ? llavaOut.split(".")[0] + ". " : ""}${qwenOut ? qwenOut.split(".")[0] + ". " : ""}I can analyse severity, emergency protocols, infrastructure damage, resource deployment, or human risk. What would you like to explore?`;
}

// ---------------------------------------------------------------------------
// Combined disaster report from all VLM outputs
// ---------------------------------------------------------------------------

function buildReport(results) {
  const { clip, blip2, llava, qwen } = results;
  if (!clip || clip.error) return null;

  const disaster   = clip.prediction ?? "Unknown";
  const confidence = clip.confidence ?? 0;
  const severity   = confidence > 88 ? "Catastrophic" : confidence > 75 ? "Severe" : confidence > 60 ? "Moderate" : "Limited";

  const tagStyle =
    severity === "Catastrophic" ? "bg-red-500/20 text-red-300 border-red-500/30" :
    severity === "Severe"       ? "bg-orange-500/20 text-orange-300 border-orange-500/30" :
    severity === "Moderate"     ? "bg-yellow-500/20 text-yellow-300 border-yellow-500/30" :
                                  "bg-slate-500/20 text-slate-300 border-slate-500/30";

  const summary = [
    blip2?.caption  && `Visual scene: ${blip2.caption}.`,
    llava?.response && llava.response.split(".").slice(0, 2).join(". ") + ".",
    qwen?.response  && "Scene assessment: " + qwen.response.split(".")[0] + ".",
  ].filter(Boolean).join(" ");

  return { disaster, confidence, severity, tagStyle, summary };
}

// ---------------------------------------------------------------------------
// Typing dots component
// ---------------------------------------------------------------------------

function TypingDots() {
  return (
    <div className="flex gap-1 px-4 py-3">
      {[0, 200, 400].map((delay) => (
        <span
          key={delay}
          className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main application
// ---------------------------------------------------------------------------

export default function App() {
  const [file,             setFile]             = useState(null);
  const [preview,          setPreview]          = useState(null);
  const [isDragging,       setIsDragging]       = useState(false);
  const [isAnalyzing,      setIsAnalyzing]      = useState(false);
  const [loadingModels,    setLoadingModels]    = useState({});
  const [results,          setResults]          = useState({});
  const [report,           setReport]           = useState(null);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [messages,         setMessages]         = useState([]);
  const [chatInput,        setChatInput]        = useState("");
  const [isChatting,       setIsChatting]       = useState(false);

  const fileInputRef = useRef(null);
  const chatEndRef   = useRef(null);
  const inputRef     = useRef(null);

  // Auto-scroll chat to latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isChatting]);

  // ---------------------------------------------------------------------------
  // File handling
  // ---------------------------------------------------------------------------

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResults({});
    setReport(null);
    setMessages([]);
    setAnalysisComplete(false);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  // ---------------------------------------------------------------------------
  // Multi-model analysis
  // ---------------------------------------------------------------------------

  const handleAnalyze = async () => {
    if (!file || isAnalyzing) return;

    setIsAnalyzing(true);
    setResults({});
    setReport(null);
    setMessages([]);
    setAnalysisComplete(false);
    setLoadingModels(MODELS.reduce((a, m) => ({ ...a, [m.key]: true }), {}));

    // Fire all four models in parallel; each updates state as it resolves.
    const settled = await Promise.allSettled(
      MODELS.map(async (model) => {
        try {
          const data = await callModel(model.endpoint, file);
          setResults((prev) => ({ ...prev, [model.key]: data }));
          return { key: model.key, data };
        } catch (err) {
          const errObj = { error: err.message };
          setResults((prev) => ({ ...prev, [model.key]: errObj }));
          return { key: model.key, data: errObj };
        } finally {
          setLoadingModels((prev) => ({ ...prev, [model.key]: false }));
        }
      })
    );

    // Build final results map and report
    const finalResults = Object.fromEntries(
      settled
        .filter((r) => r.status === "fulfilled")
        .map((r) => [r.value.key, r.value.data])
    );

    const generatedReport = buildReport(finalResults);
    setReport(generatedReport);
    setAnalysisComplete(true);
    setIsAnalyzing(false);

    // Greet user in chat
    const disasterType = finalResults.clip?.prediction ?? "disaster";
    const conf         = finalResults.clip?.confidence ?? 0;
    setMessages([{
      role: "assistant",
      text: `Analysis complete! I've consulted **${MODELS.length} AI models** on your image. CLIP detected a **${disasterType}** event with **${conf}% confidence**. Ask me anything about this scenario — severity, emergency response, infrastructure impact, or resource requirements.`,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }]);
  };

  // ---------------------------------------------------------------------------
  // Chat
  // ---------------------------------------------------------------------------

  const handleChat = async () => {
    const q = chatInput.trim();
    if (!q || isChatting || !analysisComplete) return;

    setChatInput("");
    setMessages((prev) => [...prev, {
      role: "user",
      text: q,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }]);

    setIsChatting(true);
    // Simulate a brief reasoning delay for realism
    await new Promise((r) => setTimeout(r, 700 + Math.random() * 500));

    const response = buildChatResponse(q, results);
    setMessages((prev) => [...prev, {
      role: "assistant",
      text: response,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }]);
    setIsChatting(false);
    inputRef.current?.focus();
  };

  // ---------------------------------------------------------------------------
  // Render: model result content (varies by model type)
  // ---------------------------------------------------------------------------

  const renderResult = (model) => {
    const r = results[model.key];

    if (loadingModels[model.key]) {
      return (
        <div className="space-y-2 animate-pulse">
          <div className="h-3 bg-slate-700/60 rounded-full w-full" />
          <div className="h-3 bg-slate-700/60 rounded-full w-4/5" />
          <div className="h-3 bg-slate-700/60 rounded-full w-3/5" />
        </div>
      );
    }

    if (!r) {
      return <p className="text-slate-500 text-sm italic">Waiting for analysis…</p>;
    }

    if (r.error) {
      return (
        <p className="text-red-400 text-sm">
          <span className="font-semibold">Error:</span> {r.error}
        </p>
      );
    }

    if (model.key === "clip") {
      return (
        <div>
          <div className="flex items-baseline justify-between mb-3">
            <span className="text-2xl font-bold text-white">{r.prediction}</span>
            <span className="text-cyan-400 text-xl font-bold">{r.confidence}%</span>
          </div>
          <div className="w-full bg-slate-700/50 rounded-full h-2 overflow-hidden">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-1000 ease-out"
              style={{ width: `${r.confidence}%` }}
            />
          </div>
          <p className="text-slate-400 text-xs mt-1.5">Classification confidence</p>
        </div>
      );
    }

    if (model.key === "blip2") {
      return (
        <p className="text-slate-200 text-sm leading-relaxed italic">
          &ldquo;{r.caption}&rdquo;
        </p>
      );
    }

    return (
      <p className="text-slate-200 text-sm leading-relaxed line-clamp-5">
        {r.response}
      </p>
    );
  };

  // ---------------------------------------------------------------------------
  // Render: chat message with bold markdown support
  // ---------------------------------------------------------------------------

  const renderMessageText = (text) =>
    text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br/>");

  // ---------------------------------------------------------------------------
  // JSX
  // ---------------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-slate-950 text-white overflow-x-hidden">

      {/* ── Ambient background orbs ────────────────────────────────────────── */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none" aria-hidden>
        <div className="absolute -top-48 -right-48 w-96 h-96 bg-blue-600/8 rounded-full blur-3xl" />
        <div className="absolute -bottom-48 -left-48 w-96 h-96 bg-violet-600/8 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-500/4 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-8">

        {/* ── Hero ───────────────────────────────────────────────────────────── */}
        <header className="text-center space-y-5 pb-4">
          <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/20 rounded-full px-4 py-1.5">
            <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
            <span className="text-blue-300 text-sm font-medium tracking-wide">
              Multi-Model AI Research System
            </span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight leading-tight bg-gradient-to-br from-white via-slate-200 to-cyan-300 bg-clip-text text-transparent">
            VLM Disaster
            <br />
            Intelligence Assistant
          </h1>

          <p className="text-slate-400 text-base sm:text-lg max-w-xl mx-auto leading-relaxed">
            Multi-Model Disaster Understanding &amp; Decision Support System
          </p>

          <div className="flex flex-wrap justify-center gap-2 pt-1">
            {MODELS.map((m) => (
              <span
                key={m.key}
                className={`inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1 rounded-full
                  bg-gradient-to-r ${m.gradient} bg-opacity-10 border border-white/10 text-white/80`}
              >
                <span>{m.icon}</span> {m.name}
              </span>
            ))}
          </div>
        </header>

        {/* ── Upload card ────────────────────────────────────────────────────── */}
        <section className="glass-card rounded-2xl p-8 space-y-6">
          <h2 className="text-lg font-semibold flex items-center gap-2 text-slate-200">
            <span className="text-2xl">📤</span> Upload Disaster Image
          </h2>

          {/* Drag-and-drop zone */}
          <div
            role="button"
            tabIndex={0}
            aria-label="Upload image"
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300
              ${isDragging
                ? "border-blue-400 bg-blue-500/10 scale-[1.01]"
                : "border-slate-600 hover:border-blue-500/60 hover:bg-slate-700/20"
              }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />

            {preview ? (
              <div className="space-y-3">
                <img
                  src={preview}
                  alt="Uploaded preview"
                  className="max-h-72 mx-auto rounded-xl object-contain shadow-xl"
                />
                <p className="text-slate-400 text-sm">
                  {file?.name}
                  <span className="text-slate-600 mx-2">·</span>
                  <span className="text-blue-400 hover:underline">Click to change</span>
                </p>
              </div>
            ) : (
              <div className="space-y-3 py-4">
                <div className="text-6xl select-none">🌐</div>
                <p className="text-slate-200 font-semibold text-lg">
                  Drop your disaster image here
                </p>
                <p className="text-slate-500 text-sm">or click to browse</p>
                <p className="text-slate-600 text-xs">
                  JPG · PNG · WebP · BMP · TIFF · up to 10 MB
                </p>
              </div>
            )}
          </div>

          {/* Analyze button */}
          <button
            onClick={handleAnalyze}
            disabled={!file || isAnalyzing}
            className={`w-full py-4 rounded-xl font-bold text-base tracking-wide transition-all duration-300
              flex items-center justify-center gap-3
              ${!file || isAnalyzing
                ? "bg-slate-700/60 text-slate-500 cursor-not-allowed"
                : "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 active:scale-[0.99]"
              }`}
          >
            {isAnalyzing ? (
              <>
                <svg className="animate-spin w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Consulting {MODELS.length} AI Models…
              </>
            ) : (
              <>🔬 Run Multi-Model Analysis</>
            )}
          </button>
        </section>

        {/* ── Model result cards ─────────────────────────────────────────────── */}
        {(isAnalyzing || analysisComplete) && (
          <section className="grid sm:grid-cols-2 gap-5">
            {MODELS.map((model) => (
              <div
                key={model.key}
                className={`glass-card rounded-2xl p-6 shadow-lg ${model.glow} border ${model.border} transition-all duration-500`}
              >
                {/* Card header */}
                <div className="flex items-center gap-3 mb-5">
                  <div className={`w-10 h-10 shrink-0 rounded-xl bg-gradient-to-br ${model.gradient} flex items-center justify-center text-lg shadow-md`}>
                    {model.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-bold text-white leading-tight">{model.name}</p>
                    <p className="text-slate-400 text-xs">{model.subtitle}</p>
                  </div>
                  <div className="shrink-0 ml-auto">
                    {loadingModels[model.key] ? (
                      <span className="text-blue-400 text-xs animate-pulse">Processing…</span>
                    ) : results[model.key] && !results[model.key]?.error ? (
                      <span className="text-emerald-400 text-xs font-medium">✓ Done</span>
                    ) : results[model.key]?.error ? (
                      <span className="text-red-400 text-xs">✗ Error</span>
                    ) : null}
                  </div>
                </div>

                {/* Result content */}
                <div className="min-h-[64px]">
                  {renderResult(model)}
                </div>
              </div>
            ))}
          </section>
        )}

        {/* ── Combined disaster report ───────────────────────────────────────── */}
        {report && (
          <section className="glass-card rounded-2xl p-8 border border-red-500/20 shadow-lg shadow-red-500/5">
            {/* Report header */}
            <div className="flex flex-wrap items-start justify-between gap-4 mb-6">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-red-500/15 flex items-center justify-center text-3xl shrink-0">
                  🚨
                </div>
                <div>
                  <h2 className="text-2xl font-black text-white">
                    {report.disaster} Event Detected
                  </h2>
                  <p className="text-slate-400 text-sm mt-0.5">
                    Combined Multi-Model Analysis Report
                  </p>
                </div>
              </div>
              <span className={`px-4 py-2 rounded-xl text-sm font-bold border ${report.tagStyle}`}>
                {report.severity} Severity
              </span>
            </div>

            {/* KPI row */}
            <div className="grid grid-cols-3 gap-3 mb-6">
              {[
                { label: "Event Type",        value: results.clip?.prediction ?? "—",  color: "text-white" },
                { label: "CLIP Confidence",   value: `${results.clip?.confidence ?? 0}%`, color: "text-cyan-400" },
                { label: "Models Consulted",  value: `${MODELS.length} AI Models`,     color: "text-violet-400" },
              ].map(({ label, value, color }) => (
                <div key={label} className="bg-slate-800/50 rounded-xl p-4 text-center">
                  <p className="text-slate-500 text-xs mb-1">{label}</p>
                  <p className={`font-bold text-lg leading-tight ${color}`}>{value}</p>
                </div>
              ))}
            </div>

            {/* Summary paragraph */}
            {report.summary && (
              <div className="bg-slate-800/40 rounded-xl p-5 mb-5">
                <p className="text-slate-200 text-sm leading-relaxed">{report.summary}</p>
              </div>
            )}

            {/* Recommended action */}
            <div className="bg-amber-500/8 border border-amber-500/20 rounded-xl p-4 flex gap-3">
              <span className="text-amber-400 text-lg shrink-0">⚡</span>
              <div>
                <p className="text-amber-300 text-xs font-bold uppercase tracking-wider mb-1">
                  Recommended Action
                </p>
                <p className="text-slate-200 text-sm leading-relaxed">
                  Based on {report.severity.toLowerCase()} {report.disaster.toLowerCase()} conditions
                  detected with {report.confidence}% confidence, immediate emergency assessment and
                  multi-agency coordination is recommended.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* ── AI Chat interface ──────────────────────────────────────────────── */}
        {analysisComplete && (
          <section className="glass-card rounded-2xl p-6 space-y-5">
            {/* Chat header */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-xl shrink-0">
                🤖
              </div>
              <div className="flex-1">
                <h2 className="text-lg font-bold text-white">
                  Disaster Intelligence Assistant
                </h2>
                <p className="text-slate-400 text-xs">
                  Ask questions about the analysed disaster scene
                </p>
              </div>
              <div className="flex items-center gap-1.5 ml-auto">
                <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                <span className="text-emerald-400 text-xs font-medium">Online</span>
              </div>
            </div>

            {/* Message list */}
            <div className="h-80 overflow-y-auto space-y-3 pr-1 scroll-smooth chat-scroll">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {msg.role === "assistant" && (
                    <span className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-sm shrink-0 mt-1 mr-2">
                      🤖
                    </span>
                  )}
                  <div
                    className={`max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed
                      ${msg.role === "user"
                        ? "bg-blue-600 text-white rounded-br-sm"
                        : "bg-slate-700/60 text-slate-100 rounded-bl-sm border border-slate-600/30"
                      }`}
                  >
                    <p
                      dangerouslySetInnerHTML={{ __html: renderMessageText(msg.text) }}
                    />
                    <p className="text-xs opacity-40 mt-1.5 text-right">{msg.time}</p>
                  </div>
                </div>
              ))}

              {isChatting && (
                <div className="flex items-start gap-2">
                  <span className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-sm shrink-0 mt-1">
                    🤖
                  </span>
                  <div className="bg-slate-700/60 rounded-2xl rounded-bl-sm border border-slate-600/30">
                    <TypingDots />
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>

            {/* Suggested questions */}
            <div className="flex flex-wrap gap-2">
              {[
                "How severe is this disaster?",
                "What emergency response is needed?",
                "Are people at risk?",
                "What resources should be deployed?",
                "What infrastructure appears damaged?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => { setChatInput(q); inputRef.current?.focus(); }}
                  className="text-xs bg-slate-700/40 hover:bg-slate-600/50 border border-slate-600/40 hover:border-blue-500/40 rounded-full px-3 py-1.5 text-slate-300 transition-all duration-200"
                >
                  {q}
                </button>
              ))}
            </div>

            {/* Input row */}
            <div className="flex gap-3">
              <input
                ref={inputRef}
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleChat()}
                placeholder="Ask about the disaster scenario…"
                className="flex-1 bg-slate-700/40 border border-slate-600/50 hover:border-slate-500/60 focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20 rounded-xl px-4 py-3 text-white placeholder-slate-500 outline-none transition-all duration-200 text-sm"
              />
              <button
                onClick={handleChat}
                disabled={!chatInput.trim() || isChatting}
                className="px-5 py-3 rounded-xl font-semibold text-sm transition-all duration-200
                  bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500
                  disabled:from-slate-700 disabled:to-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed
                  flex items-center gap-2 shadow-md shadow-blue-500/20 active:scale-95"
              >
                Send
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
          </section>
        )}

        {/* ── Footer ─────────────────────────────────────────────────────────── */}
        <footer className="text-center text-slate-600 text-sm pb-4">
          VLM Disaster Intelligence Assistant · CLIP · BLIP-2 · LLaVA · Qwen2-VL
        </footer>

      </div>
    </div>
  );
}
