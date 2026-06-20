# VLM Disaster Analyzer

A multi-stage Vision Language Model pipeline for satellite and aerial disaster image analysis, severity assessment, historical precedent retrieval, and AI-powered follow-up Q&A.

Built as an ISRO internship project; deployed on Google Colab (T4 GPU) + Vercel.

**Live demo:** https://vlm-disaster-analyzer.vercel.app

---

## Overview

Upload a disaster image or video — receive in seconds:

- **Disaster classification** across 12 categories (flood, cyclone, earthquake, wildfire, landslide, drought, …)
- **Severity tier** — Critical / High / Moderate / Low with confidence score
- **Structured damage report** — visible damage, affected area, environmental impact, recommendations
- **Historical precedents** — top-5 visually similar events from a curated 30-event India-focused database
- **Follow-up chat** — GPT-4o powered contextual Q&A with template fallback

---

## Architecture

### Three-Stage Production Pipeline

```
Image Upload
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — CLIP ViT-B/32  (~150 ms)                    │
│  12 zero-shot prompts · 512-dim cosine similarity       │
│  Output: disaster_type, confidence, non-disaster gate   │
└───────────────────────────┬─────────────────────────────┘
                            │  non-disaster? → return immediately
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 — Qwen2-VL 2B  (2–5 s GPU / ~60 s CPU)       │
│  4-bit NF4 quantization · CLIP category injected        │
│  Output: 7-field structured report                      │
└───────────────────────────┬─────────────────────────────┘
                            │  (runs in parallel with Stage 3)
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 — FAISS IndexFlatIP  (~50 ms)                 │
│  CLIP embedding reused · 40% similarity threshold       │
│  Output: top-5 historical events                        │
└─────────────────────────────────────────────────────────┘
```

### Deployment

```
Browser (Vercel SPA)
        │  HTTPS
        ▼
ngrok tunnel
        │
        ▼
Google Colab T4 — FastAPI + uvicorn (port 8000)
        │
        ├── CLIP ViT-B/32  (lazy singleton, ~300 MB VRAM)
        ├── Qwen2-VL 2B    (4-bit NF4, ~2.0 GB VRAM)
        └── FAISS index    (30 events · 512-dim · in-memory)
```

---

## Repository Structure

```
vlm-disaster-analyzer/
├── backend/                    ← FastAPI application
│   ├── main.py                 ← App entry point, router registration
│   ├── config.py               ← ACTIVE_MODELS env var → deployment flags
│   ├── routes/
│   │   ├── predict_disaster.py ← POST /predict/disaster  (production)
│   │   ├── predict_video.py    ← POST /predict/video
│   │   ├── predict.py          ← POST /predict/{model}   (research mode)
│   │   ├── chat.py             ← POST /chat
│   │   └── retrieval.py        ← POST /retrieval/search
│   └── services/
│       ├── disaster_service.py ← CLIP → Qwen2-VL → FAISS pipeline
│       ├── video_service.py    ← Frame extraction + disaster_service
│       ├── chat_service.py     ← GPT-4o + template fallback
│       ├── gpu_queue.py        ← Single asyncio.Lock GPU serialiser
│       ├── clip_service.py     ← CLIP inference wrapper
│       ├── qwen_service.py     ← Qwen2-VL inference wrapper
│       ├── blip2_service.py    ← BLIP-2 (research mode)
│       ├── llava_service.py    ← LLaVA (research mode)
│       └── retrieval_service.py← FAISS retrieval wrapper
│
├── src/
│   ├── models/
│   │   ├── clip_model.py       ← CLIP loader + embed_image() + predict_disaster()
│   │   ├── qwen_model.py       ← Qwen2-VL loader + predict_response()
│   │   ├── blip2_model.py      ← BLIP-2 (research)
│   │   ├── llava_model.py      ← LLaVA (research)
│   │   ├── gpt4v_model.py      ← GPT-4o Vision (cloud)
│   │   ├── video_llava_model.py← Video-LLaVA (future)
│   │   └── model_registry.py   ← Dynamic dispatch for /predict/{model}
│   ├── retrieval/
│   │   ├── search.py           ← FAISS similarity search
│   │   └── build_index.py      ← Index builder (run once)
│   ├── pipeline.py             ← Batch evaluation pipeline
│   ├── visualize.py            ← Result plotting
│   └── utils/
│       └── metrics.py          ← Accuracy and evaluation utilities
│
├── frontend/                   ← React SPA (Vite + Tailwind)
│   └── src/
│       ├── App.jsx             ← Main UI (upload → analyze → results → chat)
│       ├── themeEngine.js      ← Disaster-type CSS variable theme system
│       └── components/
│           └── IntroAnimation.jsx
│
├── datasets/
│   ├── historical/             ← 30-event retrieval database (tracked in git)
│   │   ├── historical_events.json
│   │   ├── images/             ← Source images for index building
│   │   └── index/
│   │       ├── disaster.index  ← FAISS IndexFlatIP (512-dim, 30 events)
│   │       └── metadata.json
│   └── video_dataset/          ← VIDI benchmark (not tracked; run download scripts)
│       └── evaluation/
│
├── notebooks/
│   ├── VLM_Disaster_Analyzer_Colab.ipynb   ← Production Colab deployment
│   ├── VLM_Disaster_Evaluation_Colab.ipynb ← Model evaluation
│   ├── VIDI_75_Research_Pipeline.ipynb     ← 75-video benchmark pipeline
│   └── video_dataset_drive.ipynb           ← Dataset download helper
│
├── scripts/
│   ├── batch_evaluate.py        ← Run CLIP accuracy over image dataset
│   ├── generate_plots.py        ← Produce accuracy/confidence plots
│   ├── download_historical_images.py
│   ├── verify_retrieval_e2e.py  ← End-to-end retrieval integration test
│   ├── patch_index.py           ← FAISS index repair utility
│   └── video_pipeline/          ← VIDI dataset download and frame extraction
│
├── docs/
│   ├── reports/
│   │   ├── TECHNICAL_REPORT.md ← Full technical specification (18 sections)
│   │   ├── Results_Summary.md
│   │   └── ISRO_Project_Summary.md
│   ├── architecture/
│   │   ├── System_Architecture.md
│   │   └── Deployment_Architecture.md
│   └── methodology/
│       ├── Dataset_Preparation.md
│       ├── Evaluation_Methodology.md
│       └── Historical_Retrieval.md
│
├── tests/
│   └── fixtures/test.jpg
│
├── start_backend.py    ← Production launcher (CLIP + Qwen2-VL)
├── start_research.py   ← Research launcher (all 4 models)
└── requirements.txt
```

---

## Quick Start

### Backend — Google Colab (recommended)

Open `notebooks/VLM_Disaster_Analyzer_Colab.ipynb` and run all cells. It installs dependencies, starts the FastAPI server, and creates an ngrok tunnel automatically.

```python
# Manual launch (alternative)
!pip install -r requirements.txt
!python start_backend.py &
from pyngrok import ngrok
url = ngrok.connect(8000)
print(url)  # Paste this URL into frontend VITE_API_URL
```

### Backend — Local

```bash
pip install -r requirements.txt
python start_backend.py          # production (CLIP + Qwen2-VL)
# or
python start_research.py         # research (all 4 models)
```

API docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev      # → http://localhost:5173
```

Set `VITE_API_URL` in `frontend/.env.development` to point at your backend.

---

## Deployment Profiles

| Profile | Command | Active Models | VRAM |
|---------|---------|---------------|------|
| Production | `python start_backend.py` | CLIP + Qwen2-VL | ~2.5 GB |
| Research | `python start_research.py` | CLIP + BLIP-2 + LLaVA + Qwen2-VL | 12–16 GB |

Switch profiles by setting `ACTIVE_MODELS` environment variable:

```bash
export ACTIVE_MODELS=clip,qwen          # production
export ACTIVE_MODELS=clip,blip2,llava,qwen  # research
```

---

## API Reference

### `POST /predict/disaster`
Unified CLIP → Qwen2-VL → FAISS pipeline. Primary production endpoint.

**Request:** `multipart/form-data`, field `file` (JPEG/PNG/WebP/BMP/TIFF, max 20 MB)

**Response:**
```json
{
  "category": "Flood",
  "classification_confidence": 94.67,
  "severity": "High",
  "visible_damage": "...",
  "affected_area": "...",
  "environmental_impact": "...",
  "recommendations": "...",
  "similar_events": [...],
  "retrieval_status": "ok",
  "active_models": ["Disaster Intelligence Engine"],
  "processing_time_ms": 1842.3
}
```

Non-disaster images return:
```json
{
  "status": "non_disaster",
  "category": "Forest",
  "confidence": 72.1,
  "message": "The uploaded image does not appear to depict a disaster scene.",
  "processing_time_ms": 148.0
}
```

### `POST /predict/video`
Extracts 4 frames at 25/50/75/90% of video duration, runs CLIP majority vote, then disaster_service on the best frame.

**Request:** `multipart/form-data`, field `file` (MP4/MOV/MKV/WebM/AVI, max 200 MB)

### `POST /chat`
Context-aware follow-up Q&A. Requires a `DisasterContext` object from a prior analysis.

### `GET /`
Health check. Returns active deployment profile, model list, and retrieval status.

---

## Model Accuracy — VIDI 75-Video Benchmark

75 videos × 5 categories × 4 frames per video = 300 frames. Majority vote aggregation.

| Model | Overall | Wildfire | Flood | Cyclone | Earthquake | Landslide |
|-------|---------|----------|-------|---------|------------|-----------|
| CLIP (zero-shot) | 85.3% | 93.3% | 86.7% | 86.7% | 80.0% | 80.0% |
| BLIP-2 | 78.7% | 86.7% | 80.0% | 80.0% | 73.3% | 73.3% |
| LLaVA-1.5 | 82.7% | 86.7% | 86.7% | 86.7% | 80.0% | 73.3% |
| Qwen2-VL | 84.0% | 86.7% | 86.7% | 86.7% | 80.0% | 80.0% |
| **Ensemble** | **87.5%** | **93.3%** | **93.3%** | **93.3%** | **80.0%** | **80.0%** |

---

## Key Implementation Details

| Detail | Value |
|--------|-------|
| CLIP prompts | 12 scene-descriptive text prompts; descriptive text outperforms bare labels by ~15–20 pp |
| Non-disaster gate | Labels {Forest, Sea, Human, Buildings and Street} or confidence < 20% skip Qwen entirely |
| Qwen2-VL quantization | NF4 4-bit via `bitsandbytes`; degrades to fp16 → float32 on CPU |
| GPU serialisation | Single `asyncio.Lock` in `gpu_queue.py`; max queue depth 3, timeout 600 s |
| FAISS index | `IndexFlatIP`, 512-dim, 30 events, unit-normalised vectors (inner product = cosine) |
| Similarity threshold | 40% minimum — prevents misleading low-quality retrievals |
| Retrieval categories | Flood, Cyclone, Earthquake only; others return `retrieval_status: "unsupported_category"` |
| Image validation | Magic-byte check + PIL verify + min 32×32 px |
| Video validation | Magic-byte check for MP4/MOV/MKV/WebM/AVI |
| Chat input limits | 2000 chars per question, 500 chars per history message (last 8 exchanges) |

---

## Historical Retrieval Database

30 curated Indian disaster events across 3 categories:

| Category | Events | Date Range |
|----------|--------|------------|
| Cyclone | 10 | 2014–2023 (Hudhud, Fani, Amphan, Biparjoy, Mocha, …) |
| Flood | 10 | 2015–2023 (Kerala, Assam, Bihar, Chennai, …) |
| Earthquake | 10 | 2001–2023 (Bhuj, Sikkim, Manipur, …) |

Each event includes: year, location, description, casualties, affected population, damage (USD billion), source.

FAISS index: `datasets/historical/index/disaster.index` (614 KB)

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTIVE_MODELS` | `clip,qwen` | Comma-separated model keys to enable |
| `ENABLE_RETRIEVAL` | `true` | Enable/disable FAISS historical search |
| `QUANTIZE_QWEN` | `true` | 4-bit NF4 quantization for Qwen2-VL |
| `OPENAI_API_KEY` | — | Required for GPT-4o chat (template fallback if absent) |
| `LOG_LEVEL` | `INFO` | Python logging level |

Copy `.env.example` to `.env` and fill in values.

---

## Requirements

- Python 3.10+
- PyTorch 2.5.1 + CUDA 12.4 (optional, CPU fallback available)
- Node.js 18+ (frontend only)
- ~5 GB HuggingFace cache (production models: CLIP + Qwen2-VL)
- NVIDIA T4 or better recommended (16 GB VRAM for research mode)

```bash
pip install -r requirements.txt
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Technical Report](docs/reports/TECHNICAL_REPORT.md) | Full 18-section technical specification |
| [System Architecture](docs/architecture/System_Architecture.md) | Architecture overview and API reference |
| [Deployment Guide](docs/architecture/Deployment_Architecture.md) | Colab + Vercel setup and environment variables |
| [Dataset Preparation](docs/methodology/Dataset_Preparation.md) | VIDI benchmark, frame extraction, dataset structure |
| [Evaluation Methodology](docs/methodology/Evaluation_Methodology.md) | Benchmark protocol, majority vote, metrics |
| [Historical Retrieval](docs/methodology/Historical_Retrieval.md) | FAISS index, 30-event database, similarity search |
| [Results Summary](docs/reports/Results_Summary.md) | Accuracy tables, latency benchmarks |
| [ISRO Project Summary](docs/reports/ISRO_Project_Summary.md) | Executive summary for ISRO submission |

---

## License

MIT — see `LICENSE` for details.

---

*ISRO Space Applications Centre internship project · Deployed on Vercel + Google Colab T4*
