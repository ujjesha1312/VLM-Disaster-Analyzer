from __future__ import annotations

import sys
from pathlib import Path


# -------------------------------------------------------------------
# Path Bootstrap
# -------------------------------------------------------------------

# Add src/ to sys.path so 'models.qwen_model' resolves correctly
# regardless of which directory the server is launched from.

_SRC = Path(__file__).resolve().parent.parent.parent / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# -------------------------------------------------------------------
# Model Import
# -------------------------------------------------------------------

from models.qwen_model import predict_response  # noqa: E402


# -------------------------------------------------------------------
# Service Function
# -------------------------------------------------------------------

def run(image_path: str) -> dict:
    """
    Generate structured scene understanding using Qwen2-VL.
    """

    return predict_response(image_path)
