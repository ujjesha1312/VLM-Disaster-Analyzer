"""
predict.py — FastAPI router for all VLM inference endpoints.

Routing pattern:
  POST /predict/{model_name}   — run model-specific inference
  GET  /models                 — discover available models and their output schemas

Request handling for each POST:
  1. Validate model_name exists in dispatch table (400 if not)
  2. Validate MIME type (415 if unsupported)
  3. Read and size-check upload (413 if too large)
  4. Write to NamedTemporaryFile with delete=False (Windows-safe pattern)
  5. Dispatch to the service's run() function
  6. Clean up temp file in finally block (always, even on error)
  7. Return structured JSON response

Error mapping:
  FileNotFoundError → 404
  ValueError        → 422  (bad image data, encoding issues)
  EnvironmentError  → 503  (missing API key for GPT-4V)
  ImportError       → 501  (optional dependency missing, e.g. openai)
  Exception         → 500  (unexpected model/inference error)
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.clip_service   import run as clip_run
from services.blip_service   import run as blip_run
from services.llava_service  import run as llava_run
from services.qwen_service   import run as qwen_run
from services.gpt4v_service  import run as gpt4v_run

router = APIRouter()


# ---------------------------------------------------------------------------
# Dispatch table — model_name → service.run
# Extend here to add new VLM backends without touching any other file.
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, callable] = {
    "clip":  clip_run,
    "blip":  blip_run,
    "llava": llava_run,
    "qwen":  qwen_run,
    "gpt4v": gpt4v_run,
}


# ---------------------------------------------------------------------------
# File validation constants
# ---------------------------------------------------------------------------

_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
    "image/tiff",
})

_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB

_CONTENT_TYPE_TO_SUFFIX: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png":  ".png",
    "image/bmp":  ".bmp",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
}


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@router.get("/models", summary="List available VLM backends")
def list_models() -> dict:
    """Return metadata for every registered VLM backend.

    Use this endpoint to understand each model's purpose, output schema,
    and hardware/API requirements before calling /predict/{model_name}.
    """
    return {
        "available_models": list(_DISPATCH.keys()),
        "model_details": {
            "clip": {
                "purpose":         "Zero-shot disaster type classification",
                "approach":        "Cosine similarity between image and text prompt embeddings",
                "output_keys":     ["model", "prediction", "confidence"],
                "example_output":  {"model": "CLIP", "prediction": "Flood", "confidence": 82.4},
                "requires_gpu":    False,
                "requires_api_key": False,
            },
            "blip": {
                "purpose":         "Natural language image captioning",
                "approach":        "Unconditional autoregressive caption generation",
                "output_keys":     ["model", "caption"],
                "example_output":  {"model": "BLIP-2", "caption": "A flooded road surrounded by trees."},
                "requires_gpu":    False,
                "requires_api_key": False,
            },
            "llava": {
                "purpose":         "Visual instruction following and scene reasoning",
                "approach":        "Multimodal LLM with instruction prompt asking for scene analysis",
                "output_keys":     ["model", "response"],
                "example_output":  {"model": "LLaVA", "response": "The image shows severe flooding..."},
                "requires_gpu":    "Recommended (4-bit quant: 5 GB VRAM; fp16: 14 GB VRAM)",
                "requires_api_key": False,
                "quantization":    "4-bit NF4 via bitsandbytes (optional, auto-detected)",
            },
            "qwen": {
                "purpose":         "Multimodal scene understanding via chat-template interface",
                "approach":        "2B-parameter chat VLM with open-ended analysis question",
                "output_keys":     ["model", "response"],
                "example_output":  {"model": "Qwen-VL", "response": "Flood water is visible..."},
                "requires_gpu":    "Recommended (4-bit quant: 2 GB VRAM; fp32: 8 GB RAM)",
                "requires_api_key": False,
                "quantization":    "4-bit NF4 via bitsandbytes (optional, auto-detected)",
            },
            "gpt4v": {
                "purpose":         "Advanced cloud vision reasoning via provider abstraction",
                "approach":        "VisionProvider ABC → OpenAIVisionProvider → gpt-4o Vision API",
                "output_keys":     ["model", "provider", "response"],
                "example_output":  {"model": "GPT-4V", "provider": "OpenAI / gpt-4o", "response": "..."},
                "requires_gpu":    False,
                "requires_api_key": "OPENAI_API_KEY (set in .env)",
            },
        },
    }


# ---------------------------------------------------------------------------
# POST /predict/{model_name}
# ---------------------------------------------------------------------------

@router.post(
    "/predict/{model_name}",
    summary="Run VLM inference on a disaster image",
)
async def predict(
    model_name: str,
    file: UploadFile = File(
        ...,
        description="Disaster image — JPEG / PNG / BMP / WEBP / TIFF, max 10 MB",
    ),
) -> dict:
    """Upload a disaster image and run inference with the specified VLM.

    **Path parameter** `model_name`: `clip` | `blip` | `llava` | `qwen` | `gpt4v`

    Each model returns a different JSON schema reflecting its different purpose:
    - `clip`  → `{prediction, confidence}` — classification
    - `blip`  → `{caption}` — image captioning
    - `llava` → `{response}` — visual scene reasoning
    - `qwen`  → `{response}` — multimodal scene analysis
    - `gpt4v` → `{provider, response}` — cloud reasoning (needs OPENAI_API_KEY)
    """

    # ── 1. Validate model name ────────────────────────────────────────────────
    if model_name not in _DISPATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Available: {list(_DISPATCH.keys())}",
        )

    # ── 2. Validate MIME type ─────────────────────────────────────────────────
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Accepted: JPEG, PNG, BMP, WEBP, TIFF."
            ),
        )

    # ── 3. Read upload and check size ─────────────────────────────────────────
    contents = await file.read()
    if len(contents) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {_MAX_BYTES // (1024 * 1024)} MB.",
        )

    # ── 4. Write to temp file (Windows requires delete=False) ─────────────────
    suffix   = _CONTENT_TYPE_TO_SUFFIX.get(file.content_type, ".jpg")
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        # ── 5. Dispatch to service ────────────────────────────────────────────
        result = _DISPATCH[model_name](str(tmp_path))

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except ValueError as exc:
        # Bad image data, unexpected format, prompt encoding issues, etc.
        raise HTTPException(status_code=422, detail=str(exc))

    except EnvironmentError as exc:
        # Missing API key (GPT-4V) or unavailable external resource.
        raise HTTPException(status_code=503, detail=str(exc))

    except ImportError as exc:
        # Optional dependency missing (e.g. openai package).
        raise HTTPException(status_code=501, detail=str(exc))

    except Exception as exc:
        # OOM, model error, unexpected inference failure.
        raise HTTPException(
            status_code=500,
            detail=f"Inference error for model '{model_name}': {exc}",
        )

    finally:
        # ── 6. Always clean up the temp file ─────────────────────────────────
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return result
