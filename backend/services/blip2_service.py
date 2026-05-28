from __future__ import annotations

import sys
from pathlib import Path


# -------------------------------------------------------------------
# Path Bootstrap
# -------------------------------------------------------------------

# Add src/ to sys.path so 'models.blip2_model' resolves correctly
# regardless of which directory the server is launched from.

_SRC = Path(__file__).resolve().parent.parent.parent / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# -------------------------------------------------------------------
# Model Import
# -------------------------------------------------------------------

from models.blip2_model import predict_caption  # noqa: E402


# -------------------------------------------------------------------
# Service Function
# -------------------------------------------------------------------

def run(image_path: str) -> dict:
    """
    Generate caption using BLIP-2 model.
    """

    return predict_caption(image_path)
