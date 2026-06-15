from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Path Bootstrap
# -------------------------------------------------------------------

# Add src/ to sys.path so 'models.llava_model' resolves correctly
# regardless of which directory the server is launched from.

_SRC = Path(__file__).resolve().parent.parent.parent / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# -------------------------------------------------------------------
# Service Function
# -------------------------------------------------------------------

def run(image_path: str) -> dict:
    """
    Generate visual reasoning response using LLaVA model.

    The model import is intentionally deferred to this function body.
    This prevents LLaVA's heavy transformer classes from being loaded
    at server startup when ACTIVE_MODELS excludes 'llava'. A failed
    import returns a structured error instead of crashing the process.
    """
    try:
        from models.llava_model import predict_response  # noqa: E402
    except Exception as exc:
        logger.error("[LLaVA] Model import failed: %s", exc)
        return {
            "model": "LLaVA",
            "status": "unavailable",
            "error": f"Model unavailable: {exc}",
        }

    return predict_response(image_path)
