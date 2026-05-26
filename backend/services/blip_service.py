from __future__ import annotations

import sys
from pathlib import Path


# -------------------------------------------------------------------
# Path Bootstrap
# -------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent.parent / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# -------------------------------------------------------------------
# Model Import
# -------------------------------------------------------------------

from models.blip_model import predict_caption  # noqa: E402


# -------------------------------------------------------------------
# Service Function
# -------------------------------------------------------------------

def run(image_path: str) -> dict:
    """
    Generate caption using BLIP model.
    """

    return predict_caption(image_path)