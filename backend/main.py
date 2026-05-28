"""
main.py — FastAPI application entry point.

Multi-VLM Image Understanding Platform for disaster scene analysis.
Three Vision Language Models are active, each producing a different
style of output from the same uploaded image.

Start from the project root:
    uvicorn backend.main:app --reload --port 8000
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
        "## Multi-VLM Image Understanding Platform\n\n"
        "Upload the **same disaster image** to three Vision Language Models "
        "and compare how each interprets the scene differently:\n\n"
        "| Model | Endpoint | Output Style | Key Field |\n"
        "|-------|----------|-------------|----------|\n"
        "| **CLIP** | `POST /predict/clip` | Semantic classification | `prediction` + `confidence` |\n"
        "| **BLIP-2** | `POST /predict/blip2` | Multimodal caption generation | `caption` |\n"
        "| **LLaVA** | `POST /predict/llava` | Visual scene reasoning | `response` |\n\n"
        "### Same image — three perspectives\n"
        "```\n"
        "CLIP   → { \"prediction\": \"Flood\", \"confidence\": 87.3 }\n"
        "BLIP-2 → { \"caption\": \"a flooded road with submerged trees\" }\n"
        "LLaVA  → { \"response\": \"The image shows severe urban flooding...\" }\n"
        "```\n\n"
        "Use `GET /models` to explore all available backends and their output schemas."
    ),
    version="3.0.0",
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
    """Confirm the server is running and return the active endpoints."""
    return {
        "status":  "running",
        "version": "3.0.0",
        "active_models": ["clip", "blip2", "llava"],
        "endpoints": {
            "docs":          "/docs",
            "models":        "/models",
            "predict_clip":  "/predict/clip",
            "predict_blip2": "/predict/blip2",
            "predict_llava": "/predict/llava",
        },
    }


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting VLM Disaster Analyzer API on http://0.0.0.0:8000")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
