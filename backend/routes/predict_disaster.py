"""
predict_disaster.py — Unified disaster analysis endpoint.

POST /predict/disaster
  Runs CLIP → Qwen2-VL in sequence and returns a single merged report.
  Both models must be active (ACTIVE_MODELS must include clip and qwen).

This is the primary endpoint for the production frontend.
Individual model endpoints (/predict/clip, /predict/qwen, etc.)
remain available for research and comparison workflows.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from backend.config import is_active, DISABLED_RESPONSE

log = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp", "image/tiff",
    "application/octet-stream",
}
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/predict/disaster",
    summary="Unified CLIP + Qwen2-VL disaster intelligence report",
    tags=["Unified Inference"],
)
async def predict_disaster(
    file: Annotated[UploadFile, File(description="Disaster scene image")],
):
    """
    Two-stage disaster analysis pipeline:

    1. **CLIP** classifies the disaster category and confidence score.
    2. **Qwen2-VL** performs detailed scene analysis, informed by CLIP's output.

    Both `clip` and `qwen` must be listed in `ACTIVE_MODELS`.

    Returns a unified report::

        {
          "category":                  "Flood",
          "classification_confidence": 94.67,
          "severity":                  "High",
          "visible_damage":            "...",
          "affected_area":             "...",
          "environmental_impact":      "...",
          "recommendations":           "...",
          "active_models":             ["CLIP", "Qwen2-VL"],
          "processing_time_ms":        1842.3
        }
    """
    # ── Deployment guard ─────────────────────────────────────────────────────
    if not is_active("clip") or not is_active("qwen"):
        return JSONResponse(
            status_code=200,
            content={
                **DISABLED_RESPONSE,
                "message": (
                    "Unified endpoint requires both CLIP and Qwen2-VL to be active. "
                    "Set ACTIVE_MODELS=clip,qwen (or include both in your list)."
                ),
            },
        )

    # ── Validate content type ────────────────────────────────────────────────
    if file.content_type is not None and file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image type.",
        )

    # ── Read and size-check ──────────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 10 MB limit.",
        )

    # ── Save temp file ───────────────────────────────────────────────────────
    suffix = {
        "image/png": ".png", "image/webp": ".webp",
        "image/bmp": ".bmp", "image/tiff": ".tiff",
    }.get(file.content_type or "", ".jpg")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        # ── Run unified pipeline ─────────────────────────────────────────────
        from backend.services.disaster_service import run
        result = run(str(tmp_path))
        return JSONResponse(result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    except Exception as exc:
        log.exception("Unified disaster analysis error")
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
