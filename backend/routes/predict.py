"""
predict.py — FastAPI router for VLM inference endpoints
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.config import is_active, DISABLED_RESPONSE
from backend.services.clip_service   import run as clip_run
from backend.services.qwen_service   import run as qwen_run
from backend.services.gpt4v_service  import run as gpt4v_run
from backend.services.gpu_queue      import run_with_gpu_lock

log = logging.getLogger(__name__)

# Research-only models — isolated imports so a load failure never crashes
# the production CLIP + Qwen pipeline.
try:
    from backend.services.blip2_service import run as blip2_run
except Exception as _blip2_err:
    log.warning("[predict] BLIP-2 service unavailable at startup: %s", _blip2_err)
    def blip2_run(_path: str) -> dict:  # type: ignore[misc]
        return {"model": "BLIP-2", "status": "unavailable", "error": str(_blip2_err)}

try:
    from backend.services.llava_service import run as llava_run
except Exception as _llava_err:
    log.warning("[predict] LLaVA service unavailable at startup: %s", _llava_err)
    def llava_run(_path: str) -> dict:  # type: ignore[misc]
        return {"model": "LLaVA", "status": "unavailable", "error": str(_llava_err)}

router = APIRouter()


# -------------------------------------------------------------------
# Active Models
# -------------------------------------------------------------------

_DISPATCH = {
    "clip":  clip_run,
    "blip2": blip2_run,
    "llava": llava_run,
    "qwen":  qwen_run,
    "gpt4v": gpt4v_run,
}


# -------------------------------------------------------------------
# File Validation
# -------------------------------------------------------------------

_ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",               # Some clients omit the 'e'
    "image/png",
    "image/bmp",
    "image/webp",
    "image/tiff",
    "application/octet-stream",  # Browser fallback; PIL validates actual content
}

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

_CONTENT_TYPE_TO_SUFFIX = {
    "image/jpeg":              ".jpg",
    "image/jpg":               ".jpg",
    "image/png":               ".png",
    "image/bmp":               ".bmp",
    "image/webp":              ".webp",
    "image/tiff":              ".tiff",
    "application/octet-stream": ".jpg",
}


# -------------------------------------------------------------------
# GET /models
# -------------------------------------------------------------------

@router.get("/models")
def list_models():

    return {
        "available_models": list(_DISPATCH.keys()),
        "model_details": {

            "clip": {
                "purpose": "Zero-shot disaster classification",
                "output_keys": [
                    "model",
                    "prediction",
                    "confidence"
                ]
            },

            "blip2": {
                "purpose": "Advanced multimodal image captioning (BLIP-2)",
                "output_keys": [
                    "model",
                    "caption",
                    "confidence"
                ]
            },

            "llava": {
                "purpose": "Visual scene reasoning and explanation",
                "output_keys": [
                    "model",
                    "response",
                    "confidence"
                ]
            },

            "qwen": {
                "purpose": "Structured multimodal scene understanding (Qwen2-VL-2B)",
                "output_keys": [
                    "model",
                    "response",
                    "confidence"
                ]
            },

            "gpt4v": {
                "purpose": "Advanced multimodal reasoning",
                "output_keys": [
                    "model",
                    "response"
                ]
            }
        }
    }


# -------------------------------------------------------------------
# POST /predict/sequential
# -------------------------------------------------------------------

@router.post("/predict/sequential")
async def predict_sequential(
    models: str = Form(default="clip,qwen"),
    file: UploadFile = File(...),
):
    """
    Run an ordered list of models one at a time, serialized through the GPU lock.
    Each model completes fully before the next starts.

    Send `models` as a JSON array string (["clip","qwen"]) or comma-separated
    ("clip,qwen") in a multipart/form-data request alongside the image file.
    """
    # ── Parse model list ─────────────────────────────────────────────────────
    try:
        model_list: List[str] = json.loads(models)
    except (json.JSONDecodeError, ValueError):
        model_list = [m.strip() for m in models.split(",") if m.strip()]

    unknown = [m for m in model_list if m not in _DISPATCH]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model(s): {unknown}. Valid: {list(_DISPATCH.keys())}",
        )

    # ── Validate file ─────────────────────────────────────────────────────────
    if file.content_type is not None and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    contents = await file.read()
    if len(contents) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    suffix = _CONTENT_TYPE_TO_SUFFIX.get(file.content_type or "", ".jpg")
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        results: dict = {}
        for model_name in model_list:
            if not is_active(model_name):
                results[model_name] = DISABLED_RESPONSE
                continue
            try:
                results[model_name] = await run_with_gpu_lock(
                    asyncio.to_thread(_DISPATCH[model_name], str(tmp_path)),
                    model_name,
                )
            except Exception as exc:
                results[model_name] = {"error": str(exc)}

        return JSONResponse({
            "results": results,
            "order":   model_list,
            "mode":    "sequential",
        })

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    except Exception as exc:
        log.error("Sequential inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


# -------------------------------------------------------------------
# POST /predict/disaster  — unified CLIP → Qwen → FAISS pipeline
# Must appear before /predict/{model_name} so the static path wins.
# -------------------------------------------------------------------

@router.post("/predict/disaster")
async def predict_disaster_unified(file: UploadFile = File(...)):
    """Full unified pipeline: CLIP classification → Qwen2-VL analysis → FAISS retrieval."""
    if file.content_type is not None and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    contents = await file.read()
    if len(contents) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    suffix = _CONTENT_TYPE_TO_SUFFIX.get(file.content_type or "", ".jpg")
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        from backend.services import disaster_service
        result = await disaster_service.run(str(tmp_path))
        return JSONResponse(result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    except Exception as exc:
        log.error("Unified disaster pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


# -------------------------------------------------------------------
# POST /predict/{model_name}
# -------------------------------------------------------------------

@router.post("/predict/{model_name}")
async def predict(
    model_name: str,
    file: UploadFile = File(...)
):

    # ---------------------------------------------------------------
    # Validate model
    # ---------------------------------------------------------------

    if model_name not in _DISPATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'"
        )

    # ---------------------------------------------------------------
    # Disabled model guard (deployment profile)
    # ---------------------------------------------------------------

    if not is_active(model_name):
        return JSONResponse(status_code=200, content=DISABLED_RESPONSE)

    # ---------------------------------------------------------------
    # Validate image type
    # ---------------------------------------------------------------

    # Allow None (FastAPI couldn't parse content-type) — PIL validates.
    if file.content_type is not None and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type"
        )

    # ---------------------------------------------------------------
    # Read file
    # ---------------------------------------------------------------

    contents = await file.read()

    if len(contents) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large"
        )

    # ---------------------------------------------------------------
    # Save temp image
    # ---------------------------------------------------------------

    suffix = _CONTENT_TYPE_TO_SUFFIX.get(
        file.content_type,
        ".jpg"
    )

    tmp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as tmp:

            tmp.write(contents)
            tmp_path = Path(tmp.name)

        # -----------------------------------------------------------
        # Run selected model (serialized through GPU lock)
        # -----------------------------------------------------------

        result = await run_with_gpu_lock(
            asyncio.to_thread(_DISPATCH[model_name], str(tmp_path)),
            model_name,
        )

    except EnvironmentError as exc:

        # Missing API key (e.g. OPENAI_API_KEY not set for GPT-4V).
        # 503 = Service Unavailable — configuration issue, not a code bug.
        raise HTTPException(
            status_code=503,
            detail=f"Service not configured: {exc}"
        )

    except ValueError as exc:

        # PIL rejected the file as a non-image or the file is corrupted.
        # 400 = Bad Request — the client sent unusable data.
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}"
        )

    except Exception as exc:
        log.error("Unhandled inference error for model '%s': %s", model_name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    finally:

        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return result
