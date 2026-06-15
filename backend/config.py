"""
config.py — Deployment configuration for the VLM Disaster Analyzer.

Two deployment profiles — switch by changing ACTIVE_MODELS:

  Production (default)  ACTIVE_MODELS=clip,qwen
  Research  (all local) ACTIVE_MODELS=clip,blip2,llava,qwen

Named boolean flags are derived from ACTIVE_MODELS so calling code can
use them directly without string comparisons:

  ENABLE_CLIP      = True
  ENABLE_QWEN      = True
  ENABLE_RETRIEVAL = True   # controlled separately via ENABLE_RETRIEVAL env var
  ENABLE_BLIP2     = False  # set ACTIVE_MODELS to include 'blip2' to re-enable
  ENABLE_LLAVA     = False  # set ACTIVE_MODELS to include 'llava' to re-enable

All values are evaluated once at import time. A server restart is required
after changing any environment variable.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Active models — source of truth
# ---------------------------------------------------------------------------

VALID_MODEL_KEYS = {"clip", "blip2", "llava", "qwen", "gpt4v"}


def _parse_active_models() -> frozenset[str]:
    raw = os.getenv("ACTIVE_MODELS", "clip,qwen")
    models = frozenset(m.strip().lower() for m in raw.split(",") if m.strip())
    unknown = models - VALID_MODEL_KEYS
    if unknown:
        log.warning(
            "[Config] Unknown model keys in ACTIVE_MODELS: %s. "
            "Valid keys: %s. Check for typos.",
            unknown, VALID_MODEL_KEYS,
        )
    active = models & VALID_MODEL_KEYS
    log.info("[Config] Active models: %s", sorted(active))
    return active


ACTIVE_MODELS: frozenset[str] = _parse_active_models()


# ---------------------------------------------------------------------------
# Named boolean flags
# Derived from ACTIVE_MODELS — change ACTIVE_MODELS, these update automatically.
# ---------------------------------------------------------------------------

ENABLE_CLIP:  bool = "clip"  in ACTIVE_MODELS
ENABLE_QWEN:  bool = "qwen"  in ACTIVE_MODELS
ENABLE_BLIP2: bool = "blip2" in ACTIVE_MODELS
ENABLE_LLAVA: bool = "llava" in ACTIVE_MODELS

# Retrieval is controlled separately because it has no model weight cost —
# it reuses CLIP embeddings and a tiny FAISS index file.
# Set ENABLE_RETRIEVAL=false to skip historical similarity search entirely.
ENABLE_RETRIEVAL: bool = os.getenv("ENABLE_RETRIEVAL", "true").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Deployment profile label (informational)
# ---------------------------------------------------------------------------

def _profile_label() -> str:
    if ENABLE_BLIP2 or ENABLE_LLAVA:
        return "research"
    return "production"


DEPLOYMENT_PROFILE: str = _profile_label()

log.info(
    "[Config] Profile: %s | CLIP=%s | Qwen=%s | BLIP-2=%s | LLaVA=%s | Retrieval=%s",
    DEPLOYMENT_PROFILE.upper(),
    ENABLE_CLIP, ENABLE_QWEN, ENABLE_BLIP2, ENABLE_LLAVA, ENABLE_RETRIEVAL,
)


# ---------------------------------------------------------------------------
# Quantization flag
# ---------------------------------------------------------------------------

# When true, Qwen loads in 4-bit NF4 via bitsandbytes (~2 GB VRAM vs ~4 GB fp16).
# Silently disabled if bitsandbytes is not installed or CUDA is unavailable.
QUANTIZE_QWEN: bool = os.getenv("QUANTIZE_QWEN", "true").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Convenience helpers (backward-compatible)
# ---------------------------------------------------------------------------

def is_active(model_key: str) -> bool:
    """Return True if *model_key* is enabled in the current deployment profile."""
    return model_key.lower() in ACTIVE_MODELS


# Standard JSON body returned for every disabled-model request.
DISABLED_RESPONSE: dict = {
    "status":  "disabled",
    "message": "Model inactive in current deployment profile.",
    "hint":    "Set ACTIVE_MODELS=clip,blip2,llava,qwen to enable all models.",
}
