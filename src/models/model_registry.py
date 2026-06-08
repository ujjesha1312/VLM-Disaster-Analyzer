"""
model_registry.py — Central registry for all VLM backends.

Maps model IDs to their module paths and metadata. Used by the legacy
inference_service layer; the new service-per-model architecture imports
model modules directly.

Registered models
-----------------
clip    openai/clip-vit-base-patch32   zero-shot classification
blip2   Salesforce/blip2-opt-2.7b      image captioning + confidence
llava   llava-hf/llava-1.5-7b-hf      visual scene reasoning
qwen    Qwen/Qwen2-VL-2B-Instruct      multimodal visual analysis
gpt4v   gpt-4o  (OpenAI API)           cloud advanced reasoning
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry — model_id → dotted module path inside src/
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, str] = {
    "clip":  "models.clip_model",
    "blip2": "models.blip2_model",
    "llava": "models.llava_model",
    "qwen":  "models.qwen_model",
    "gpt4v": "models.gpt4v_model",
}

AVAILABLE_MODELS: list[str] = list(_REGISTRY.keys())

# Metadata used by API discovery endpoints.
MODEL_INFO: dict[str, dict] = {
    "clip": {
        "name":             "CLIP",
        "source":           "openai/clip-vit-base-patch32  (local)",
        "purpose":          "Zero-shot disaster classification",
        "output":           "prediction + confidence",
        "requires_gpu":     False,
        "requires_api_key": False,
    },
    "blip2": {
        "name":             "BLIP-2",
        "source":           "Salesforce/blip2-opt-2.7b",
        "purpose":          "Advanced multimodal image captioning",
        "output":           "caption + confidence",
        "requires_gpu":     False,
        "requires_api_key": False,
    },
    "llava": {
        "name":             "LLaVA-1.5",
        "source":           "llava-hf/llava-1.5-7b-hf",
        "purpose":          "Visual scene reasoning and explanation",
        "output":           "response",
        "requires_gpu":     True,
        "requires_api_key": False,
    },
    "qwen": {
        "name":             "Qwen2-VL",
        "source":           "Qwen/Qwen2-VL-2B-Instruct",
        "purpose":          "Multimodal visual understanding and analysis",
        "output":           "response",
        "requires_gpu":     True,
        "requires_api_key": False,
    },
    "gpt4v": {
        "name":             "GPT-4o Vision",
        "source":           "gpt-4o  (OpenAI API)",
        "purpose":          "Advanced visual reasoning (cloud)",
        "output":           "response",
        "requires_gpu":     False,
        "requires_api_key": True,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(model_name: str, image_path: str) -> dict:
    """Dispatch inference to the named model module.

    Args:
        model_name:  One of the keys in AVAILABLE_MODELS.
        image_path:  Absolute path to the image file.

    Returns:
        Prediction dict from the model module.

    Raises:
        ValueError: If model_name is not registered.
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {AVAILABLE_MODELS}"
        )

    module = importlib.import_module(_REGISTRY[model_name])
    logger.info("Dispatching to model '%s'", model_name)
    return module.predict_disaster(image_path)


def list_models() -> list[dict]:
    """Return metadata for every registered model."""
    return [{"model_id": k, **v} for k, v in MODEL_INFO.items()]
