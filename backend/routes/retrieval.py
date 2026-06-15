"""
retrieval.py — Historical disaster similarity search endpoints.

POST /predict/similar
    Upload an image → returns top-5 visually similar historical disaster events.

GET /retrieval/status
    Returns whether the FAISS index has been built and how many events are indexed.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp", "image/tiff",
    "application/octet-stream",
}
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/predict/similar",
    summary="Historical disaster similarity search",
    tags=["Historical Retrieval"],
    response_description="Top-k visually similar historical disaster events",
)
async def predict_similar(
    file: Annotated[UploadFile, File(description="Disaster scene image")],
    top_k: int = Query(default=5, ge=1, le=20, description="Number of similar events to return"),
    category: str | None = Query(default=None, description="Filter results to a specific disaster category"),
):
    """
    Find historical disaster events that are visually similar to the uploaded image.

    Uses CLIP embeddings and FAISS cosine similarity search against a curated
    database of 30 historical Indian and global disaster events.

    **Requires the FAISS index to be pre-built:**
        `python src/retrieval/build_index.py`

    Returns::

        {
          "similar_events": [
            {
              "event":               "Kerala Floods",
              "year":                2018,
              "location":            "Kerala, India",
              "category":            "flood",
              "description":         "...",
              "similarity":          87.3,
              "casualties":          483,
              "affected_population": "5.4 million"
            },
            ...
          ],
          "index_status": { "index_built": true, "event_count": 28 }
        }
    """
    if file.content_type is not None and file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image type.",
        )

    contents = await file.read()
    if len(contents) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 10 MB limit.",
        )

    suffix = {
        "image/png": ".png", "image/webp": ".webp",
        "image/bmp": ".bmp", "image/tiff": ".tiff",
    }.get(file.content_type or "", ".jpg")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        from backend.services.retrieval_service import run
        result = await run(str(tmp_path), top_k=top_k, category_filter=category)
        return JSONResponse(result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    except Exception as exc:
        log.exception("Retrieval error")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@router.get(
    "/retrieval/status",
    summary="FAISS index health check",
    tags=["Historical Retrieval"],
)
def retrieval_status():
    """
    Returns whether the FAISS similarity index has been built.

    If `index_built` is false, run:
        `python src/retrieval/build_index.py`
    """
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    from retrieval.search import index_status
    return JSONResponse(index_status())
