# VLM Disaster Analyzer

Multi-model Vision-Language system for disaster event classification, structured damage reporting, and historical retrieval — built for ISRO internship.

**Live demo:** https://vlm-disaster-analyzer.vercel.app  
**Backend:** Google Colab T4 + ngrok (launch `start_backend.py`)

---

## What It Does

Upload a disaster image → receive in seconds:

- **Disaster type** — classified across 12 categories (flood, wildfire, earthquake, landslide, cyclone, …)
- **Severity tier** — Critical / High / Moderate / Low with confidence score
- **Structured damage report** — 7 fields: visible damage, affected area, environmental impact, recommendations
- **Historical precedents** — top-5 visually similar events from a 30-event India-focused database with casualty and economic data
- **Chat interface** — GPT-4o powered contextual Q&A with keyword fallback

---

## Three-Stage Pipeline

```
Image Upload
    │
    ▼
Stage 1 — CLIP zero-shot triage (~500 ms)
    12 descriptive prompts · 512-dim cosine similarity
    Output: disaster_type + confidence %
    │
    ├──────────────────────────────────────────────────────┐
    ▼                                                      ▼
Stage 2 — Qwen2-VL 7-field report (2–3 s GPU)    Stage 3 — FAISS retrieval (<100 ms)
    Structured damage assessment                    Top-5 similar historical events
    Injected with CLIP category prefix              CLIP embedding reused, no 2nd pass
    └──────────────────────────────────────────────────────┘
                        │
                        ▼
              Unified JSON Response
```

---

## Model Accuracy (VIDI 75-Video Benchmark)

| Model | Overall | Wildfire | Flood | Cyclone | Earthquake | Landslide |
|---|---|---|---|---|---|---|
| CLIP (zero-shot) | 85.3% | 93.3% | 86.7% | 86.7% | 80.0% | 80.0% |
| BLIP-2 | 78.7% | 86.7% | 80.0% | 80.0% | 73.3% | 73.3% |
| LLaVA-1.5 | 82.7% | 86.7% | 86.7% | 86.7% | 80.0% | 73.3% |
| Qwen2-VL | 84.0% | 86.7% | 86.7% | 86.7% | 80.0% | 80.0% |
| **Ensemble** | **87.5%** | **93.3%** | **93.3%** | **93.3%** | **80.0%** | **80.0%** |

---

## Quick Start

### Backend (Google Colab)

```python
# 1. Install dependencies
!pip install -r requirements.txt

# 2. Set environment
import os
os.environ["ACTIVE_MODELS"] = "clip,qwen"
os.environ["ENABLE_RETRIEVAL"] = "true"

# 3. Start server + ngrok tunnel
from pyngrok import ngrok
import subprocess, threading

threading.Thread(target=lambda: subprocess.run(["python", "start_backend.py"]), daemon=True).start()
print(ngrok.connect(8000))  # Copy this URL to Vercel VITE_API_URL
```

### Frontend (local dev)

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### API (Swagger UI)

```
http://localhost:8000/docs
```

---

## Deployment Profiles

| Profile | Command | Models | VRAM |
|---|---|---|---|
| Production | `python start_backend.py` | CLIP + Qwen2-VL | ~2.5 GB |
| Research (all models) | `python start_research.py` | CLIP + BLIP-2 + LLaVA + Qwen2-VL | 12–16 GB |

---

## Documentation

| Document | Description |
|---|---|
| [TECHNICAL_REPORT.md](docs/reports/TECHNICAL_REPORT.md) | Complete technical specification (14 sections) |
| [System Architecture](docs/architecture/System_Architecture.md) | Three-tier architecture, pipeline, API reference |
| [Deployment Guide](docs/architecture/Deployment_Architecture.md) | Colab + Vercel setup, environment variables |
| [Dataset Preparation](docs/methodology/Dataset_Preparation.md) | VIDI 75-video dataset, frame extraction |
| [Evaluation Methodology](docs/methodology/Evaluation_Methodology.md) | Benchmark protocol, majority vote, metrics |
| [Historical Retrieval](docs/methodology/Historical_Retrieval.md) | FAISS index, 30-event database, search |
| [Results Summary](docs/reports/Results_Summary.md) | Accuracy tables, latency benchmarks, findings |
| [ISRO Project Summary](docs/reports/ISRO_Project_Summary.md) | Executive summary for ISRO submission |

---

## Repository Structure

```
vlm-disaster-analyzer/
├── backend/                    ← FastAPI application
│   ├── main.py                 ← Entry point, router registration
│   ├── config.py               ← ACTIVE_MODELS → boolean flags
│   ├── routes/                 ← HTTP route handlers
│   └── services/               ← Business logic, GPU queue
├── src/
│   ├── models/                 ← CLIP, Qwen2-VL, BLIP-2, LLaVA
│   ├── retrieval/              ← FAISS index builder and search
│   └── utils/                  ← Shared utilities
├── frontend/                   ← React SPA (Vite)
│   └── src/App.jsx             ← Main UI component
├── datasets/
│   ├── historical/             ← 30-event database + FAISS index
│   └── video_dataset/          ← VIDI evaluation frames
├── notebooks/                  ← Colab research notebooks
├── scripts/
│   └── video_pipeline/         ← Frame extraction, packaging
├── docs/                       ← Full project documentation
├── tests/                      ← Backend tests + fixtures
├── start_backend.py            ← Production launcher
└── start_research.py           ← Research launcher (all models)
```

---

## Key Technical Details

- **CLIP prompts:** 12 scene-descriptive prompts; descriptive text outperforms bare labels by ~15–20 pp
- **Qwen2-VL:** Loaded with `local_files_only=True` and `device_map=None` on CPU (prevents layer norm shape mismatch)
- **4-bit quantization:** NF4 via `bitsandbytes`; degrades automatically to fp16 → float32 if CUDA unavailable
- **GPU serialization:** Single `asyncio.Lock` in `gpu_queue.py` prevents OOM from concurrent model forward passes
- **FAISS:** `IndexFlatIP` with unit-normalized 512-dim vectors; inner product = cosine similarity
- **Stage 3 fault tolerance:** FAISS failure returns `similar_events: []` without affecting Stage 1–2 output

---

## Requirements

- Python 3.10+
- PyTorch 2.5.1
- CUDA 12.4 (GPU, optional)
- Node.js 18+ (frontend)
- ~5 GB HuggingFace cache (production models)
- NVIDIA T4 recommended (16 GB VRAM for research mode)

```bash
pip install -r requirements.txt
```

---

## License

MIT License — see `LICENSE` for details.

---

*Built during ISRO internship · Deployed on Vercel + Google Colab*
