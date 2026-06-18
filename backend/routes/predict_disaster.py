"""
predict_disaster.py — Unified disaster analysis endpoint.

POST /predict/disaster
  Runs CLIP → Qwen2-VL in sequence and returns a single merged report.
  Both models must be active (ACTIVE_MODELS must include clip and qwen).

This is the primary endpoint for the production frontend.
Individual model endpoints (/predict/clip, /predict/qwen, etc.)
remain available for research and comparison workflows.

Hardening changes:
  Fix A — Image size limit raised to 20 MB.
  Fix B — Magic-byte validation: file content is verified against known
           image signatures so renamed non-image files are rejected even
           when content_type is application/octet-stream.
  Fix C — Temp file deletion is in a finally block (unchanged — already correct).
  Fix D — Original filename is never used in any filesystem path (unchanged).
  Fix I — PIL.Image.open() failure and images smaller than 32×32 pixels
           both return HTTP 422 with a descriptive message.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from backend.config import is_active, DISABLED_RESPONSE

log = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Allowed MIME types (Fix B — checked AND verified against magic bytes)
# ---------------------------------------------------------------------------

_ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp", "image/tiff",
    "application/octet-stream",  # Browser fallback — magic-byte check takes over
}

# Fix A — Raise image size limit to 20 MB
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB

# Fix B — Magic-byte signatures for each allowed image format.
# Checked against the first 12 bytes of the uploaded file.
_MAGIC_SIGS: list[bytes] = [
    b"\xff\xd8\xff",          # JPEG
    b"\x89PNG\r\n\x1a\n",    # PNG
    b"RIFF",                  # WebP (also need bytes[8:12] == b"WEBP")
    b"BM",                    # BMP
    b"II*\x00",               # TIFF little-endian
    b"MM\x00*",               # TIFF big-endian
]

# Minimum acceptable image dimension (pixels per side)
_MIN_DIM = 32


def _validate_magic_bytes(data: bytes) -> bool:
    """
    Return True if the first bytes of *data* match any allowed image format.
    Prevents accepting .py/.exe/.svg etc. with a renamed or missing extension.
    """
    header = data[:12]
    for sig in _MAGIC_SIGS:
        if header.startswith(sig):
            # WebP extra check: bytes 8–11 must be b"WEBP"
            if sig == b"RIFF" and header[8:12] != b"WEBP":
                continue
            return True
    return False


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

    # ── Validate content type (MIME header check) ─────────────────────────────
    if file.content_type is not None and file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Accepted formats: JPEG, PNG, WebP, BMP, TIFF."
            ),
        )

    # ── Read and size-check (Fix A) ───────────────────────────────────────────
    contents = await file.read()
    if len(contents) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File exceeds 20 MB limit "
                f"({len(contents) / 1_048_576:.1f} MB uploaded)."
            ),
        )

    # ── Magic-byte validation (Fix B) ────────────────────────────────────────
    if not _validate_magic_bytes(contents):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "File content does not match a supported image format. "
                "Only JPEG, PNG, WebP, BMP, and TIFF files are accepted."
            ),
        )

    # ── PIL open + minimum dimension check (Fix I) ────────────────────────────
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()                          # detects truncated/corrupted images
        img = Image.open(io.BytesIO(contents))  # re-open after verify() consumes stream
        w, h = img.size
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Cannot decode image: {exc}. "
                "The file may be corrupted or truncated."
            ),
        )

    if w < _MIN_DIM or h < _MIN_DIM:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Image is too small ({w}×{h} px). "
                f"Minimum accepted size is {_MIN_DIM}×{_MIN_DIM} pixels."
            ),
        )

    # ── Save temp file (Fix D — system-generated name only) ──────────────────
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
        result = await run(str(tmp_path))
        return JSONResponse(result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    except HTTPException:
        raise  # propagate 503 (queue full / timeout) without wrapping
    except Exception as exc:
        log.exception("Unified disaster analysis error")
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")
    finally:
        # Fix C — guaranteed cleanup even when exceptions occur mid-processing
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
