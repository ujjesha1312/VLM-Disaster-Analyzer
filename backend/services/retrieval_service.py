"""
retrieval_service.py — Backend wrapper for historical disaster similarity search.

Bridges the FastAPI route layer to the FAISS retrieval module in src/retrieval/.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Bootstrap src/ so retrieval.search resolves correctly.
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


async def run(image_path: str, top_k: int = 5, category_filter: str | None = None) -> dict:
    """
    Run FAISS similarity search for the given image.

    Args:
        image_path      : Absolute path to a temporary image file.
        top_k           : Number of similar events to return.
        category_filter : Optional category string to restrict results.

    Returns:
        {
          "similar_events": [...],
          "index_status":   {...},
          "top_k":          int,
        }
    """
    from retrieval.search import find_similar_events, index_status

    status  = index_status()
    similar = await asyncio.to_thread(
        find_similar_events,
        image_path,
        top_k,
        category_filter,
    )

    log.info("[Retrieval] Query complete — %d results returned", len(similar))

    return {
        "similar_events": similar,
        "index_status":   status,
        "top_k":          top_k,
    }
