"""
llava_service.py — Service layer for LLaVA visual scene reasoning.

Responsibility:
  - Bootstrap sys.path so src/models/ resolves regardless of launch directory
  - Import and delegate to llava_model.predict_disaster()
  - Return the model's response dict unchanged

Output schema:
    {"model": "LLaVA", "response": "The image shows severe flooding affecting..."}
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.llava_model import predict_disaster  # noqa: E402


def run(image_path: str) -> dict:
    """Run LLaVA visual instruction following on a disaster image.

    Args:
        image_path: Absolute path to the temporary image file created by the route.

    Returns:
        {"model": "LLaVA", "response": str}

    Raises:
        FileNotFoundError: Propagated from model layer if path is missing.
        ValueError:        Propagated if file is not a valid image.
    """
    return predict_disaster(image_path)
