"""
main.py — FastAPI application entry point.

Multi-VLM Image Understanding Platform for disaster scene analysis.
All five models (CLIP, BLIP, LLaVA, Qwen-VL, GPT-4V) share this common
FastAPI server, upload pipeline, and routing infrastructure.

Start the server from the backend/ directory:
    uvicorn main:app --reload --port 8000

Or with structured logging:
    uvicorn main:app --reload --port 8000 --log-level info
"""

import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from backend.routes.predict import router as predict_router
# Load .env at startup so OPENAI_API_KEY and LOG_LEVEL are available immediately.
load_dotenv()

# Configure Python logging level from environment (default: INFO).
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VLM Disaster Analyzer",
    description=(
        "**Multi-VLM Image Understanding Platform** for disaster scene analysis.\n\n"
        "Upload any disaster image to five different Vision Language Models and "
        "compare how each one interprets the same scene:\n\n"
        "| Model | Purpose | Key Output |\n"
        "|-------|---------|------------|\n"
        "| **CLIP** | Zero-shot classification | `prediction` + `confidence` |\n"
        "| **BLIP** | Image captioning | `caption` |\n"
        "| **LLaVA** | Visual scene reasoning | `response` |\n"
        "| **Qwen-VL** | Multimodal scene analysis | `response` |\n"
        "| **GPT-4V** | Cloud advanced reasoning | `response` (needs API key) |\n\n"
        "Use `GET /models` to explore all available backends and their output schemas."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# All VLM endpoints live under the predict router.
app.include_router(predict_router, tags=["VLM Inference"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"], summary="API health check")
def root() -> dict:
    """Confirm the server is running and return the available endpoints."""
    return {
        "status":    "running",
        "version":   "2.0.0",
        "endpoints": {
            "docs":         "/docs",
            "models":       "/models",
            "predict_clip": "/predict/clip",
            "predict_blip": "/predict/blip",
            "predict_llava":"/predict/llava",
            "predict_qwen": "/predict/qwen",
            "predict_gpt4v":"/predict/gpt4v",
        },
    }


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting VLM Disaster Analyzer API on http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
