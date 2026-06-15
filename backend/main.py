"""
main.py — FastAPI application entry point.

Multi-VLM Image Understanding Platform for disaster scene analysis.
Five Vision Language Models are active, each producing a different
style of output from the same uploaded image.

Start from the project root:
    uvicorn backend.main:app --reload --port 8000
"""

import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.predict          import router as predict_router
from backend.routes.chat             import router as chat_router
from backend.routes.predict_video    import router as video_router
from backend.routes.predict_disaster import router as disaster_router
from backend.routes.retrieval        import router as retrieval_router

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
        "Upload the **same disaster image** to five Vision Language Models "
        "and compare how each interprets the scene differently:\n\n"
        "| Model | Endpoint | Backend | Output Style | Key Field |\n"
        "|-------|----------|---------|-------------|----------|\n"
        "| **CLIP** | `POST /predict/clip` | Local | Semantic classification | `prediction` + `confidence` |\n"
        "| **BLIP-2** | `POST /predict/blip2` | Local | Multimodal caption generation | `caption` |\n"
        "| **LLaVA** | `POST /predict/llava` | Local | Visual scene reasoning | `response` |\n"
        "| **Qwen2-VL** | `POST /predict/qwen` | Local | Structured scene understanding | `response` |\n"
        "| **GPT-4V** | `POST /predict/gpt4v` | Cloud ☁️ | Advanced multimodal reasoning | `response` |\n\n"
        "> **GPT-4V** requires `OPENAI_API_KEY` in `.env`. Returns HTTP 503 if not configured.\n\n"
        "### Same image — five perspectives\n"
        "```\n"
        "CLIP      → { \"prediction\": \"Flood\", \"confidence\": 87.3 }\n"
        "BLIP-2    → { \"caption\": \"a flooded road with submerged trees\" }\n"
        "LLaVA     → { \"response\": \"The image shows severe urban flooding...\" }\n"
        "Qwen2-VL  → { \"response\": \"Large-scale flooding with submerged roads...\" }\n"
        "GPT-4V    → { \"response\": \"The image depicts severe flooding affecting...\" }\n"
        "```\n\n"
        "Use `GET /models` to explore all available backends and their output schemas."
    ),
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the React dev server and the Vercel production frontend.
# allow_credentials=True enables cookies/auth headers from these origins.
# Specific origins are required when allow_credentials=True (wildcard not allowed).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://vlm-disaster-analyzer.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route registration order matters — FastAPI matches in registration order.
# Rules:
#   1. disaster_router  (/predict/disaster)     before predict_router (/predict/{model_name})
#   2. video_router     (/predict/video/*)      before predict_router (/predict/{model_name})
# Both ensure specific paths are not swallowed by the wildcard route.
app.include_router(disaster_router,   tags=["Unified Inference"])
app.include_router(retrieval_router,  tags=["Historical Retrieval"])
app.include_router(video_router,      tags=["Video VLM Inference"])
app.include_router(predict_router,    tags=["VLM Inference"])
app.include_router(chat_router,       tags=["Intelligence Chat"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"], summary="API health check")
def root() -> dict:
    """Confirm the server is running and return the active endpoints."""
    return {
        "status":  "running",
        "version": "5.0.0",
        "active_models": {
            "local": ["clip", "blip2", "llava", "qwen"],
            "cloud": ["gpt4v"],
        },
        "endpoints": {
            "docs":           "/docs",
            "models":         "/models",
            "predict_clip":   "/predict/clip",
            "predict_blip2":  "/predict/blip2",
            "predict_llava":  "/predict/llava",
            "predict_qwen":   "/predict/qwen",
            "predict_gpt4v":  "/predict/gpt4v",
        },
        "notes": {
            "gpt4v": "Requires OPENAI_API_KEY in .env — returns 503 if not configured",
        },
    }


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting VLM Disaster Analyzer API on http://0.0.0.0:8000")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
