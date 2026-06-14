"""
config.py — Deployment configuration for the VLM Disaster Analyzer.

Controls which models are active via the ACTIVE_MODELS environment variable.

Deployment mode  (default):  ACTIVE_MODELS=clip,qwen
Research mode    (all):       ACTIVE_MODELS=clip,blip2,llava,qwen

The config is evaluated once at import time. Changing ACTIVE_MODELS
requires a server restart.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Active models
# ---------------------------------------------------------------------------

VALID_MODEL_KEYS = {"clip", "blip2", "llava", "qwen", "gpt4v"}


def _parse_active_models() -> frozenset[str]:
    raw = os.getenv("ACTIVE_MODELS", "clip,qwen")
    models = frozenset(m.strip().lower() for m in raw.split(",") if m.strip())
    unknown = models - VALID_MODEL_KEYS
    if unknown:
        log.warning(
            f"[Config] Unknown model keys in ACTIVE_MODELS: {unknown}. "
            f"Valid keys: {VALID_MODEL_KEYS}. Check for typos."
        )
    active = models & VALID_MODEL_KEYS
    log.info(f"[Config] Active models: {sorted(active)}")
    return active


ACTIVE_MODELS: frozenset[str] = _parse_active_models()


# ---------------------------------------------------------------------------
# Quantization flag
# ---------------------------------------------------------------------------

# When true, Qwen is loaded in 4-bit NF4 via bitsandbytes (~2 GB VRAM vs ~4 GB).
# Silently disabled if bitsandbytes is not installed or no CUDA is available.
QUANTIZE_QWEN: bool = os.getenv("QUANTIZE_QWEN", "true").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def is_active(model_key: str) -> bool:
    """Return True if *model_key* should be loaded and served."""
    return model_key.lower() in ACTIVE_MODELS


# Standard JSON body returned for every disabled-model request.
DISABLED_RESPONSE: dict = {
    "status":  "disabled",
    "message": "Model inactive in current deployment profile.",
    "hint":    "Set ACTIVE_MODELS=clip,blip2,llava,qwen to enable all models.",
}
