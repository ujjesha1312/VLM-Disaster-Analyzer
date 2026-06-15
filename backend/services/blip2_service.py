from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Path Bootstrap
# -------------------------------------------------------------------

# Add src/ to sys.path so 'models.blip2_model' resolves correctly
# regardless of which directory the server is launched from.

_SRC = Path(__file__).resolve().parent.parent.parent / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# -------------------------------------------------------------------
# Service Function
# -------------------------------------------------------------------

def run(image_path: str) -> dict:
    """
    Generate caption using BLIP-2 model.

    The model import is intentionally deferred to this function body.
    This prevents BLIP-2's heavy transformer classes from being loaded
    at server startup when ACTIVE_MODELS excludes 'blip2'. A failed
    import returns a structured error instead of crashing the process.
    """
    try:
        from models.blip2_model import predict_caption  # noqa: E402
    except Exception as exc:
        logger.error("[BLIP-2] Model import failed: %s", exc)
        return {
            "model": "BLIP-2",
            "status": "unavailable",
            "error": f"Model unavailable: {exc}",
        }

    return predict_caption(image_path)
