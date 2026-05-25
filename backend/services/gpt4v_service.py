"""
gpt4v_service.py — Service layer for GPT-4o Vision provider abstraction.

Responsibility:
  - Bootstrap sys.path so src/models/ resolves regardless of launch directory
  - Import and delegate to gpt4v_model.predict_disaster()
  - Return the model's response dict unchanged

The model layer (gpt4v_model.py) handles provider selection via get_provider().
When OPENAI_API_KEY is set, live GPT-4o Vision inference runs. When it is not
set, an EnvironmentError is raised — the route layer converts this to HTTP 503.

Output schema (live):
    {"model": "GPT-4V", "provider": "OpenAI / gpt-4o", "response": str}
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.gpt4v_model import predict_disaster  # noqa: E402


def run(image_path: str) -> dict:
    """Run GPT-4o Vision analysis on a disaster image.

    Args:
        image_path: Absolute path to the temporary image file created by the route.

    Returns:
        {"model": "GPT-4V", "provider": str, "response": str}

    Raises:
        FileNotFoundError: Propagated if path is missing.
        ValueError:        Propagated if file is not a valid image.
        EnvironmentError:  If OPENAI_API_KEY is not set (→ HTTP 503 at route layer).
        ImportError:       If openai package is not installed.
    """
    return predict_disaster(image_path)
