"""
start_research.py — Research mode launcher for the VLM Disaster Analyzer.

Active models : CLIP + Qwen2-VL + BLIP-2 + LLaVA
Retrieval     : Historical FAISS similarity search ENABLED

WARNING: Requires ~12-16 GB VRAM for all 4 models.
         On CPU this will be extremely slow (hours per image for LLaVA).
         Use a T4 GPU (Google Colab) for viable research-mode inference.

To return to production mode:
    python start_backend.py
"""

import os
import sys

# ── Research profile ──────────────────────────────────────────────────────────
os.environ["ACTIVE_MODELS"]    = "clip,blip2,llava,qwen"
os.environ["ENABLE_RETRIEVAL"] = "true"
os.environ["QUANTIZE_QWEN"]    = "true"

# ── Startup ───────────────────────────────────────────────────────────────────
sys.path.insert(0, ".")

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("  VLM Disaster Analyzer — RESEARCH MODE")
    print("  Models : CLIP + BLIP-2 + LLaVA + Qwen2-VL")
    print("  Note   : Requires ~12-16 GB VRAM for all models")
    print("  URL    : http://localhost:8000")
    print("  Docs   : http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
