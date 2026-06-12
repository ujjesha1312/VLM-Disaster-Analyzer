"""
predict_video.py — FastAPI router for video-based disaster analysis.

Mirrors the existing predict.py image router but accepts video uploads.
Each endpoint corresponds to one video understanding model.

Mount in backend/main.py:
    from backend.routes.predict_video import router as video_router
    app.include_router(video_router, tags=["Video VLM Inference"])

Endpoints:
    POST /predict/video/llava          — Video-LLaVA 7B
    POST /predict/video/internvideo    — InternVideo2 (stub)
    POST /predict/video/qwen           — Qwen2-VL video mode (stub)
    GET  /predict/video/models         — list available video models
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/predict/video")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_VIDEO_SIZE_MB = 500


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class VideoAnalysisResponse(BaseModel):
    model:       str
    response:    str
    confidence:  float
    frames_used: int
    duration_s:  float | None = None
    error:       str | None   = None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _validate_video(upload: UploadFile) -> None:
    ext = Path(upload.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported video format '{ext}'. "
                   f"Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )


async def _save_temp_video(upload: UploadFile) -> str:
    """Save uploaded video to a temp file and return its path."""
    ext  = Path(upload.filename or "video.mp4").suffix.lower()
    data = await upload.read()

    size_mb = len(data) / (1024 ** 2)
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video size {size_mb:.1f} MB exceeds limit of {MAX_VIDEO_SIZE_MB} MB.",
        )

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return tmp_path


def _get_video_duration(path: str) -> float | None:
    """Use ffprobe to get video duration in seconds."""
    import subprocess, json
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            info = json.loads(r.stdout)
            return float(info.get("format", {}).get("duration", 0)) or None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Unified placeholder analysis endpoint — works without any model installed
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    summary="Analyse a video file — metadata extraction and placeholder assessment",
)
async def analyze_video(
    video: Annotated[UploadFile, File(description="Video file (.mp4, .avi, .mov, .mkv, .webm)")],
):
    """
    Returns stream metadata, a thumbnail frame (base64 JPEG), and a structured
    placeholder disaster assessment — no video model required.

    Response schema::

        {
          "file_info":  {filename, format, duration_s, fps, resolution,
                         width, height, size_mb, total_frames, codec, source},
          "thumbnail_b64": "data:image/jpeg;base64,…" | null,
          "analysis":   {event_type, severity, confidence, summary,
                         frames_analyzed, assessment_note, pending_models},
          "processing_time_ms": float
        }

    Plug in a real model (Video-LLaVA / InternVideo2 / Qwen2-VL) to replace
    the mock ``analysis`` block without touching the response schema.
    """
    _validate_video(video)
    tmp_path = await _save_temp_video(video)
    try:
        from backend.services.video_service import process_video
        result = process_video(tmp_path)
        return JSONResponse(result)
    except Exception as exc:
        log.exception("Video analysis error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Video-LLaVA endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/llava",
    response_model=VideoAnalysisResponse,
    summary="Analyse a disaster video with Video-LLaVA 7B",
)
async def predict_video_llava(
    video: Annotated[UploadFile, File(description="Video clip (.mp4, .avi, .mov)")],
    prompt: Annotated[str | None, Form()] = None,
):
    """
    Accepts a short video clip (≤ 60 s recommended) and returns a structured
    disaster intelligence assessment from Video-LLaVA.

    The model samples **8 uniformly-spaced frames** from the clip and runs
    multimodal LLM inference with the optional *prompt* or a default
    disaster-analysis system prompt.
    """
    _validate_video(video)
    tmp_path = await _save_temp_video(video)

    try:
        duration = _get_video_duration(tmp_path)

        from src.models.video_llava_model import predict
        result = predict(tmp_path, prompt=prompt)

        return VideoAnalysisResponse(
            model       = result["model"],
            response    = result["response"],
            confidence  = result["confidence"],
            frames_used = result["frames_used"],
            duration_s  = duration,
            error       = result.get("error"),
        )
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Video-LLaVA not installed: {e}",
        )
    except Exception as e:
        log.exception("Video-LLaVA inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# InternVideo2 endpoint (stub — ready for implementation)
# ---------------------------------------------------------------------------

@router.post(
    "/internvideo",
    summary="[STUB] Analyse video with InternVideo2",
)
async def predict_internvideo(
    video: Annotated[UploadFile, File()],
    prompt: Annotated[str | None, Form()] = None,
):
    """
    InternVideo2 — state-of-the-art video understanding model from Shanghai AI Lab.
    Excellent for action/event recognition and temporal understanding.

    Install:  pip install huggingface_hub
    Model:    OpenGVLab/InternVideo2-Stage2_1B-224p-f4
    Paper:    https://arxiv.org/abs/2403.15377

    Implementation:
        1. Load InternVideo2 tokenizer + model from HuggingFace
        2. Sample 8 frames, resize to 224×224
        3. Run classify() or generate() depending on task
        4. Map class logits → disaster category probabilities
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "InternVideo2 endpoint not yet implemented. "
            "See src/models/intern_video_model.py for integration guide. "
            "Install: pip install timm einops"
        ),
    )


# ---------------------------------------------------------------------------
# Qwen2-VL video endpoint (extends existing qwen_model.py)
# ---------------------------------------------------------------------------

@router.post(
    "/qwen",
    summary="[STUB] Analyse video with Qwen2-VL",
)
async def predict_video_qwen(
    video: Annotated[UploadFile, File()],
    prompt: Annotated[str | None, Form()] = None,
):
    """
    Qwen2-VL natively supports both images and videos.
    The existing qwen_model.py uses the image interface;
    this endpoint extends it to accept video files.

    Implementation:
        1. Sample frames with decord (16 frames recommended for Qwen2-VL)
        2. Pass to the existing Qwen2-VL processor as a video input
        3. Run inference with the disaster analysis prompt

    Qwen2-VL video processing is documented at:
    https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Qwen2-VL video endpoint not yet implemented. "
            "The model supports video natively — extend src/models/qwen_model.py "
            "to accept a list of PIL frames instead of a single image."
        ),
    )


# ---------------------------------------------------------------------------
# Model inventory endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/models",
    summary="List available video analysis models",
)
def list_video_models():
    return JSONResponse({
        "available": [
            {
                "id":          "video-llava-7b",
                "endpoint":    "/predict/video/llava",
                "status":      "available",
                "description": "Video-LLaVA 7B — 8-frame multimodal analysis",
                "hf_repo":     "LanguageBind/Video-LLaVA-7B-hf",
                "install":     "pip install transformers accelerate decord",
                "vram_gb":     14,
            },
            {
                "id":          "internvideo2-1b",
                "endpoint":    "/predict/video/internvideo",
                "status":      "stub",
                "description": "InternVideo2 1B — temporal event classification",
                "hf_repo":     "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
                "install":     "pip install timm einops",
                "vram_gb":     4,
            },
            {
                "id":          "qwen2-vl-video",
                "endpoint":    "/predict/video/qwen",
                "status":      "stub",
                "description": "Qwen2-VL 7B — native video + text understanding",
                "hf_repo":     "Qwen/Qwen2-VL-7B-Instruct",
                "install":     "pip install qwen-vl-utils",
                "vram_gb":     16,
            },
        ],
        "recommended_for_disaster_video": [
            "Video-LLaVA-7B — best zero-shot scene description",
            "InternVideo2-1B — best action/event classification accuracy",
            "Qwen2-VL-7B — best overall multimodal reasoning",
        ],
    })
