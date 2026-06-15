# Deployment Architecture

**VLM Disaster Analyzer — Infrastructure and Setup Guide**

> Source of truth: [TECHNICAL_REPORT.md](../reports/TECHNICAL_REPORT.md) §13  
> Related: [System_Architecture.md](System_Architecture.md) · [Results_Summary.md](../reports/Results_Summary.md)

---

## Overview

The system uses a **decoupled deployment model**: the GPU backend runs on-demand in Google Colab while the frontend is hosted persistently on Vercel. This allows the frontend to remain online 24/7 at zero cost while GPU compute is provisioned only when active inference is needed.

```
┌──────────────────────────────────────────────────────────┐
│                  USER BROWSER                            │
└──────────────────────┬───────────────────────────────────┘
                       │  HTTPS
         ┌─────────────▼───────────────┐
         │   Vercel (Frontend)         │  → Always online
         │   vlm-disaster-analyzer     │  → Static SPA
         │   .vercel.app               │  → localStorage fallback
         └─────────────┬───────────────┘
                       │  HTTPS via ngrok tunnel
         ┌─────────────▼───────────────┐
         │   Google Colab              │  → On-demand GPU
         │   NVIDIA T4 (16 GB VRAM)    │  → uvicorn :8000
         │   FastAPI backend           │  → ngrok public URL
         └─────────────────────────────┘
```

**Why this architecture:**  
Persistent GPU hosting (e.g., a dedicated A10G instance) costs $200–800/month. A T4 Colab session costs nothing under free tier and handles research bursts adequately. The frontend degrades gracefully when the backend is offline — clients receive keyword-fallback chat responses and can browse their localStorage session history.

---

## Backend — Google Colab Setup

### Prerequisites
- Google account with Colab access
- ngrok account (free tier) with auth token
- Repository cloned or uploaded to Colab

### Step 1 — Open the notebook

Use `notebooks/VIDI_75_Research_Pipeline.ipynb` (research evaluation) or run `start_backend.py` directly in a Colab code cell.

### Step 2 — Install dependencies

```python
!pip install -r requirements.txt
```

Key packages: `fastapi`, `uvicorn`, `torch`, `transformers`, `faiss-cpu`, `Pillow`, `python-dotenv`, `openpyxl`, `pyngrok`, `accelerate`, `torchvision`, `bitsandbytes`

### Step 3 — Configure environment

Create `.env` in the Colab working directory:

```bash
OPENAI_API_KEY=sk-...          # Optional: enables GPT-4V and /chat
ACTIVE_MODELS=clip,qwen        # Production profile
ENABLE_RETRIEVAL=true
QUANTIZE_QWEN=true
LOG_LEVEL=INFO
```

Or set via code cell:
```python
import os
os.environ["ACTIVE_MODELS"]    = "clip,qwen"
os.environ["ENABLE_RETRIEVAL"] = "true"
os.environ["QUANTIZE_QWEN"]    = "true"
```

### Step 4 — Start backend with ngrok tunnel

```python
from pyngrok import ngrok
import subprocess, threading

# Set your ngrok auth token
ngrok.set_auth_token("YOUR_NGROK_TOKEN")

# Start uvicorn in background thread
def run_server():
    subprocess.run(["python", "start_backend.py"])

thread = threading.Thread(target=run_server, daemon=True)
thread.start()

# Open public tunnel
public_url = ngrok.connect(8000)
print(f"Backend URL: {public_url}")
# → Copy this URL into Vercel VITE_API_URL
```

### Step 5 — Update frontend API URL

In Vercel dashboard → Project Settings → Environment Variables:
```
VITE_API_URL = https://abc123.ngrok-free.app
```

Trigger a Vercel redeploy. The frontend will now route API calls to the Colab backend.

---

## Frontend — Vercel Deployment

### Repository connection
1. Push repository to `github.com/ujjesha1312/VLM-Disaster-Analyzer`
2. Import into Vercel → select `frontend/` as the root directory
3. Build command: `npm run build`
4. Output directory: `dist`

### Environment variables (Vercel dashboard)

| Variable | Value |
|---|---|
| `VITE_API_URL` | `https://<ngrok-subdomain>.ngrok-free.app` |

Update this value each Colab session (ngrok free tier rotates URLs on each tunnel open). Vercel redeploy is required after changing it, or use a fixed ngrok subdomain (paid plan).

### Production URL

```
https://vlm-disaster-analyzer.vercel.app
```

The frontend is pre-configured in `backend/main.py` CORS allowlist:
```python
allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://vlm-disaster-analyzer.vercel.app",
]
```

---

## Local Development (CPU)

For development without GPU. Qwen2-VL inference takes ~220 s on CPU.

### Backend

```bash
# Clone and enter repo
git clone https://github.com/ujjesha1312/VLM-Disaster-Analyzer
cd VLM-Disaster-Analyzer

# Create venv
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env: add OPENAI_API_KEY if needed

# Start production profile (CLIP + Qwen2-VL)
python start_backend.py
# → http://localhost:8000
# → http://localhost:8000/docs   (Swagger UI)
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Research mode (all 4 models — requires ~16 GB VRAM)

```bash
python start_research.py
```

---

## Deployment Profiles Reference

| Profile | Launcher | `ACTIVE_MODELS` | VRAM | Use Case |
|---|---|---|---|---|
| Production | `start_backend.py` | `clip,qwen` | ~2.5 GB | Live inference, demos, monitoring |
| Research | `start_research.py` | `clip,blip2,llava,qwen` | 12–16 GB | Benchmarking, VIDI evaluation |

To switch profile, simply stop the server and run the other launcher. No code changes required.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `ACTIVE_MODELS` | `clip,qwen` | Comma-separated list of active models |
| `ENABLE_RETRIEVAL` | `true` | Enable FAISS historical similarity search |
| `QUANTIZE_QWEN` | `true` | 4-bit NF4 quantization for Qwen2-VL on GPU |
| `OPENAI_API_KEY` | *(none)* | Enables GPT-4V and `/chat` endpoint |
| `LOG_LEVEL` | `INFO` | Python logging level |

`ACTIVE_MODELS` valid keys: `clip`, `qwen`, `blip2`, `llava`, `gpt4v`

---

## HuggingFace Model Cache

Models are cached at `~/.cache/huggingface/hub/` on first load. In Colab, this cache is in the session's ephemeral storage and must be re-downloaded each new runtime unless Google Drive is mounted.

**Cached models (production profile):**
- `openai/clip-vit-base-patch32` — ~0.6 GB
- `Qwen/Qwen2-VL-2B-Instruct` — ~4.1 GB

**Important:** Qwen2-VL is loaded with `local_files_only=True`. If the model is not in cache, it will raise `OSError: We couldn't connect to huggingface.co`. Ensure the model is downloaded before launching with this flag, or remove it for the initial download run.

---

## FAISS Index Setup

The retrieval module requires the index to be built before use:

```bash
# Step 1: Download reference images from Wikipedia
python scripts/download_historical_images.py

# Step 2: (Auto-triggered by step 1, or run manually)
python src/retrieval/build_index.py

# Verify:
curl http://localhost:8000/retrieval/status
# → { "index_built": true, "event_count": 28, ... }
```

If the index is not built, `/predict/disaster` still works — `similar_events` will be `[]`.

---

## System Requirements

| Component | Specification |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.5.1 |
| CUDA (optional) | 12.4 |
| Min VRAM (production) | 4 GB (CLIP + Qwen 4-bit) |
| Recommended VRAM (research) | 16 GB (NVIDIA T4) |
| RAM (CPU-only) | 8 GB minimum, 16 GB recommended |
| Image upload limit | 10 MB |
| Video upload limit | 500 MB |
| Supported image formats | JPEG, PNG, WebP, BMP, TIFF |
| Node.js (frontend) | 18+ |

---

*For API endpoint reference see [System_Architecture.md](System_Architecture.md)*  
*For performance benchmarks see [Results_Summary.md](../reports/Results_Summary.md)*
