"""
start_backend.py — Production launcher for the VLM Disaster Analyzer.

Active models : CLIP + Qwen2-VL
Retrieval     : Historical FAISS similarity search ENABLED
BLIP-2        : DISABLED (code intact — re-enable via start_research.py)
LLaVA         : DISABLED (code intact — re-enable via start_research.py)

To switch to research mode (all 4 models):
    python start_research.py

Memory footprint (production):
    CLIP alone          ~0.3 GB VRAM
    CLIP + Qwen 4-bit   ~2.5 GB VRAM
    CLIP + Qwen fp16    ~4.5 GB VRAM
"""

import os
import sys

# ── Deployment profile ────────────────────────────────────────────────────────
os.environ["ACTIVE_MODELS"]    = "clip,qwen"
os.environ["ENABLE_RETRIEVAL"] = "true"
os.environ["QUANTIZE_QWEN"]    = "true"   # 4-bit NF4 on CUDA, fp32 on CPU

# ── Startup ───────────────────────────────────────────────────────────────────
sys.path.insert(0, ".")

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("  VLM Disaster Analyzer — PRODUCTION MODE")
    print("  Models : CLIP + Qwen2-VL")
    print("  BLIP-2 : disabled  (start_research.py to enable)")
    print("  LLaVA  : disabled  (start_research.py to enable)")
    print("  URL    : http://localhost:8000")
    print("  Docs   : http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
