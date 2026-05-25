"""
blip_service.py — Service layer for BLIP image caption generation.

Responsibility:
  - Bootstrap sys.path so src/models/ resolves regardless of launch directory
  - Import and delegate to blip_model.predict_disaster()
  - Return the model's response dict unchanged

Output schema:
    {"model": "BLIP-2", "caption": "A flooded road surrounded by trees and water."}
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.blip_model import predict_disaster  # noqa: E402


def run(image_path: str) -> dict:
    """Generate a natural-language caption using BLIP captioning.

    Args:
        image_path: Absolute path to the temporary image file created by the route.

    Returns:
        {"model": "BLIP-2", "caption": str}

    Raises:
        FileNotFoundError: Propagated from model layer if path is missing.
        ValueError:        Propagated if file is not a valid image.
    """
    return predict_disaster(image_path)
