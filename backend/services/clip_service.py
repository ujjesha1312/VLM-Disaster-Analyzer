"""
clip_service.py — Service layer for CLIP zero-shot disaster classification.

Responsibility:
  - Bootstrap sys.path so src/models/ resolves regardless of launch directory
  - Import and delegate to clip_model.predict_disaster()
  - Return the model's response dict unchanged

This layer exists to separate HTTP concerns (routes) from model concerns (src/models/).
Routes never import model modules directly; they always go through a service.

Output schema:
    {"model": "CLIP", "prediction": "Flood", "confidence": 82.4}
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable from any working directory.
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.clip_model import predict_disaster  # noqa: E402


def run(image_path: str) -> dict:
    """Run CLIP zero-shot classification on a disaster image.

    Args:
        image_path: Absolute path to the temporary image file created by the route.

    Returns:
        {"model": "CLIP", "prediction": str, "confidence": float}

    Raises:
        FileNotFoundError: Propagated from model layer if path is missing.
        ValueError:        Propagated if file is not a valid image.
    """
    return predict_disaster(image_path)
