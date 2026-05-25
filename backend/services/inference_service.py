"""
inference_service.py — Bridge between FastAPI routes and the model registry.

Adds the project's src/ directory to sys.path so model modules resolve
correctly regardless of where uvicorn is started from.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any src/ import
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model_registry import AVAILABLE_MODELS, MODEL_INFO, list_models, predict  # noqa: E402

__all__ = ["run_inference", "get_available_models", "get_model_info"]


def run_inference(model_name: str, image_path: str) -> dict:
    """Validate the model name and delegate to the registry.

    Args:
        model_name:  Any key from AVAILABLE_MODELS.
        image_path:  Absolute path to the temporary image file.

    Returns:
        Prediction dict from the model module.

    Raises:
        ValueError:  If model_name is not registered.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not available. "
            f"Choose from: {AVAILABLE_MODELS}"
        )
    return predict(model_name, image_path)


def get_available_models() -> list[str]:
    """Return the list of registered model IDs."""
    return AVAILABLE_MODELS


def get_model_info() -> list[dict]:
    """Return metadata for all registered models."""
    return list_models()
