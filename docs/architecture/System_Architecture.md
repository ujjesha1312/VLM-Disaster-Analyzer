# System Architecture

**VLM Disaster Analyzer — Technical Architecture Reference**

> Source of truth: [TECHNICAL_REPORT.md](../reports/TECHNICAL_REPORT.md)  
> Related: [Deployment_Architecture.md](Deployment_Architecture.md) · [Historical_Retrieval.md](../methodology/Historical_Retrieval.md) · [Results_Summary.md](../reports/Results_Summary.md)

---

## Overview

The VLM Disaster Analyzer is structured as a three-tier backend fronted by an independently deployed React SPA.

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Vercel)                   │
│   Upload → Analyzing → Ready  |  Chat  |  Session History   │
└────────────────────┬────────────────────────────────────────┘
                     │  multipart/form-data  (HTTPS via ngrok)
┌────────────────────▼────────────────────────────────────────┐
│               Tier 1 — HTTP Routing Layer (FastAPI)         │
│  Request validation · File-size limits · Content-type check │
│  /predict/disaster  /predict/similar  /chat  /predict/*     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Tier 2 — Services Abstraction Layer            │
│  Deployment profile enforcement · GPU queue · Aggregation   │
│  disaster_service  clip_service  qwen_service  retrieval_   │
│  service  blip2_service  llava_service  video_service       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Tier 3 — Model Inference Layer                 │
│  src/models/  clip_model · qwen_model · blip2_model ·       │
│  llava_model · gpt4v_model · video_llava_model             │
│  src/retrieval/  search · build_index                       │
└─────────────────────────────────────────────────────────────┘
```

**Design principle:** Substituting a model or adding a new vision architecture requires no modification to Tier 1 or Tier 2. Each tier is unaware of the implementation details of tiers below it.

---

## Three-Stage Production Pipeline

The unified inference endpoint (`POST /predict/disaster`) runs three stages in series.

```
Image Upload
     │
     ▼
┌─────────────────────────────────────────┐
│  Stage 1 — CLIP Triage (~500 ms)        │
│  Zero-shot cosine similarity over 12    │
│  descriptive text prompts.              │
│  Output: disaster_type + confidence %  │
│  Embedding: 512-dim unit-norm vector   │
└────────────┬────────────────┬───────────┘
             │ category+conf  │ embedding
             ▼                ▼
┌────────────────────┐  ┌─────────────────────────────────────┐
│  Stage 2 — Qwen    │  │  Stage 3 — FAISS Retrieval (<100ms) │
│  2-VL (2–3 s GPU)  │  │  Top-5 similar historical events    │
│  Structured 7-field│  │  from 30-event India-focused DB.    │
│  disaster report   │  │  Best-effort: failure returns []    │
└────────────┬───────┘  └─────────────────┬───────────────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
                   Unified JSON Response
          { category, severity, visible_damage,
            affected_area, environmental_impact,
            recommendations, similar_events,
            processing_time_ms, confidence }
```

Stage 3 is gated on `ENABLE_RETRIEVAL=true` and wrapped in `try/except`. A missing FAISS index returns `similar_events: []` without breaking Stage 1–2 output.

---

## Model Selection

| Model | HuggingFace ID | Architecture | Params | VRAM | Role |
|---|---|---|---|---|---|
| CLIP | openai/clip-vit-base-patch32 | Zero-Shot Classification | ~151 M | ~0.3 GB | Stage 1 triage + FAISS embedding |
| Qwen2-VL | Qwen/Qwen2-VL-2B-Instruct | Instruction-Following VLM | ~2 B | ~2.0 GB (4-bit) | Stage 2 structured report |
| BLIP-2 | Salesforce/blip2-opt-2.7b | Bootstrapped Captioning | ~2.7 B | ~5.5 GB | Research: scene captioning |
| LLaVA | llava-hf/llava-1.5-7b-hf | Visual Instruction Tuning | ~7 B | ~14 GB | Research: structured QA |
| GPT-4V | gpt-4o (OpenAI API) | Cloud Multimodal LLM | ~1.8 T* | Cloud | Reference benchmark |

CLIP's 512-dimensional unit-normalized embeddings serve dual purpose: disaster classification (Stage 1) and FAISS cosine similarity search (Stage 3) — one forward pass, two uses.

---

## Lazy Loading and Singleton Pattern

No model weights are loaded at server startup. Each model uses double-checked locking:

```python
_model = None

def _load_model():
    global _model
    if _model is None:                    # First check (no lock)
        with _lock:
            if _model is None:            # Second check (under lock)
                _model = load_weights()
    return _model
```

First-request latency on T4 GPU: CLIP ~2 s, Qwen2-VL ~45–90 s (weight dequantization). Subsequent requests: inference latency only.

---

## Sequential GPU Execution

A single `asyncio.Lock` in `backend/services/gpu_queue.py` serializes all model forward passes:

```python
_GPU_LOCK = asyncio.Lock()

async def run_with_gpu_lock(coro, model_name):
    async with _GPU_LOCK:
        return await coro
```

**Why sequential, not parallel:** All four local models together require 12–16 GB VRAM. A T4 has 16 GB. Parallel execution would exceed headroom for activations and cause OOM exceptions — a hard failure. Sequential execution at 12–20 s total is preferable to a crashed process.

---

## Deployment Profile System

`backend/config.py` reads `ACTIVE_MODELS` once at import time and derives boolean flags:

```
ACTIVE_MODELS = "clip,qwen"          → ENABLE_CLIP=True, ENABLE_QWEN=True
                                        ENABLE_BLIP2=False, ENABLE_LLAVA=False
                                        DEPLOYMENT_PROFILE="production"

ACTIVE_MODELS = "clip,blip2,llava,qwen" → all True
                                          DEPLOYMENT_PROFILE="research"
```

Service files read `ENABLE_*` flags and return `DISABLED_RESPONSE` (HTTP 200, `status: "disabled"`) for inactive models — no error, no crash.

| Launcher | Profile | Active Models | VRAM |
|---|---|---|---|
| `start_backend.py` | Production | CLIP + Qwen2-VL | ~2.5 GB |
| `start_research.py` | Research | CLIP + BLIP-2 + LLaVA + Qwen2-VL | 12–16 GB |

See [Deployment_Architecture.md](Deployment_Architecture.md) for full setup instructions.

---

## Confidence Scoring

| Tier | CLIP Score | Interpretation | Action |
|---|---|---|---|
| High | > 88% | Strong visual cues, unambiguous | Immediate operational use |
| Strong | 75–88% | Clear disaster indicators | Recommended for deployment |
| Moderate | 60–75% | Partial occlusion or ambiguity | Secondary verification |
| Preliminary | < 60% | Low visual discriminability | Manual expert review |

Qwen2-VL confidence is approximated by averaging maximum token probability across generation steps (no self-reported logit is available). The same four-tier thresholds are applied for interface consistency.

---

## Prompt Engineering

**CLIP:** Twelve semantically rich scene-descriptive prompts instead of bare labels.  
- `"an image showing flood or water disaster with submerged areas"` outperforms `"Flood"` by an estimated 15–20 percentage points due to CLIP's contrastive pretraining on caption-style text.

**Qwen2-VL:** Labeled-field template with uppercase colon-delimited field names:
```
CLIP classified this image as "{category}" ({confidence:.1f}% confidence).
Analyze this disaster image.
Return ONLY in this exact format:
DISASTER TYPE: <type>
SEVERITY: <critical/high/moderate/low>
DESCRIPTION: <one sentence>
VISIBLE DAMAGE: <one sentence>
AFFECTED AREA: <one sentence>
ENVIRONMENTAL IMPACT: <one sentence>
RECOMMENDATIONS: <one sentence>
```

Deterministic line-by-line parsing extracts each field without regular expressions.

---

## API Endpoint Reference

| Endpoint | Method | Input | Function |
|---|---|---|---|
| `/predict/clip` | POST | Image | CLIP classification: type, confidence, top-3 |
| `/predict/qwen` | POST | Image | Qwen2-VL 7-field structured report |
| `/predict/blip2` | POST | Image | BLIP-2 dense captioning |
| `/predict/llava` | POST | Image | LLaVA instruction-following QA |
| `/predict/gpt4v` | POST | Image | GPT-4V cloud analysis (needs API key) |
| `/predict/disaster` | POST | Image | **Unified 3-stage pipeline (production)** |
| `/predict/similar` | POST | Image | FAISS top-k historical event retrieval |
| `/chat` | POST | JSON | GPT-4o chat + keyword fallback |
| `/predict/video/analyze` | POST | Video | Metadata extraction + thumbnail |
| `/retrieval/status` | GET | — | FAISS index health check |
| `/models` | GET | — | Active model inventory |
| `/` | GET | — | Health check |

Router registration order in `backend/main.py` is critical: `/predict/disaster` and `/predict/video/*` are registered before the wildcard `/predict/{model_name}` to prevent route capture.

---

## Error Handling and Fault Tolerance

| Layer | Failure Mode | Handling |
|---|---|---|
| GPU quantization | 4-bit unavailable | Degrades: 4-bit NF4 → fp16 → float32 |
| Stage 3 FAISS | Index not built | Returns `similar_events: []`, no error |
| Research pipeline | One model fails | Isolated; other models continue |
| Video metadata | ffprobe unavailable | Fallback: OpenCV → file-stat |
| Cloud API | No API key | HTTP 503 (not 500) |
| Frontend | Backend unreachable | Keyword-fallback chat, localStorage history |

---

*For deployment setup see [Deployment_Architecture.md](Deployment_Architecture.md)*  
*For historical retrieval deep-dive see [Historical_Retrieval.md](../methodology/Historical_Retrieval.md)*  
*For performance numbers see [Results_Summary.md](../reports/Results_Summary.md)*
