"""
predict.py — FastAPI router for VLM inference endpoints
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.config import is_active, DISABLED_RESPONSE
from backend.services.clip_service   import run as clip_run
from backend.services.blip2_service  import run as blip2_run
from backend.services.llava_service  import run as llava_run
from backend.services.qwen_service   import run as qwen_run
from backend.services.gpt4v_service  import run as gpt4v_run

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
        # Run selected model
        # -----------------------------------------------------------

        result = _DISPATCH[model_name](str(tmp_path))

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
        import traceback
        traceback.print_exc()
        return {
            "error": str(exc),
            "type":  type(exc).__name__,
            "trace": traceback.format_exc(),
        }

    finally:

        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return result
