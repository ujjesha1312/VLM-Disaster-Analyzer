"""
main.py — FastAPI application entry point.

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
        "## Disaster Intelligence Platform\n\n"
        "Upload a disaster image or video to receive a structured intelligence report "
        "with severity assessment, damage analysis, historical precedents, and follow-up "
        "chat capabilities.\n\n"
        "### Production pipeline\n"
        "```\n"
        "Image → CLIP triage (~150 ms) → Qwen2-VL report (2–5 s) → FAISS retrieval (~50 ms)\n"
        "```\n\n"
        "### Key endpoints\n"
        "| Endpoint | Purpose |\n"
        "|----------|---------|\n"
        "| `POST /predict/disaster` | Unified CLIP + Qwen2-VL report (production) |\n"
        "| `POST /predict/video` | Video frame extraction + disaster report |\n"
        "| `POST /chat` | Context-aware follow-up Q&A |\n"
        "| `POST /retrieval/search` | Standalone FAISS historical event search |\n"
        "| `GET /models` | List available models and their schemas |\n\n"
        "Research endpoints (`/predict/clip`, `/predict/blip2`, `/predict/llava`, "
        "`/predict/qwen`) are available when `ACTIVE_MODELS` includes the relevant key.\n\n"
        "> Set `ACTIVE_MODELS=clip,qwen` (production) or `clip,blip2,llava,qwen` (research)."
    ),
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins so the frontend works from Colab/ngrok tunnels (URL changes each session).
# allow_credentials must be False when allow_origins=["*"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    """Confirm the server is running and return the deployment profile."""
    from backend.config import ACTIVE_MODELS, DEPLOYMENT_PROFILE, ENABLE_RETRIEVAL
    return {
        "status":             "running",
        "version":            "5.0.0",
        "deployment_profile": DEPLOYMENT_PROFILE,
        "active_models":      sorted(ACTIVE_MODELS),
        "retrieval_enabled":  ENABLE_RETRIEVAL,
        "endpoints": {
            "docs":             "/docs",
            "unified_analysis": "/predict/disaster",
            "video_analysis":   "/predict/video",
            "chat":             "/chat",
            "retrieval":        "/retrieval/search",
            "models":           "/models",
        },
    }


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting VLM Disaster Analyzer API on http://0.0.0.0:8000")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
