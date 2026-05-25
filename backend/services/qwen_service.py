"""
qwen_service.py — Service layer for Qwen2-VL multimodal visual analysis.

Responsibility:
  - Bootstrap sys.path so src/models/ resolves regardless of launch directory
  - Import and delegate to qwen_model.predict_disaster()
  - Return the model's response dict unchanged

Output schema:
    {"model": "Qwen-VL", "response": "Flood water is visible across the road..."}
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.qwen_model import predict_disaster  # noqa: E402


def run(image_path: str) -> dict:
    """Run Qwen2-VL multimodal analysis on a disaster image.

    Args:
        image_path: Absolute path to the temporary image file created by the route.

    Returns:
        {"model": "Qwen-VL", "response": str}

    Raises:
        FileNotFoundError: Propagated from model layer if path is missing.
        ValueError:        Propagated if file is not a valid image.
    """
    return predict_disaster(image_path)
