"""
build_index.py — Build the FAISS similarity index from historical disaster images.

Run this script once after placing reference images in:
    datasets/historical/images/{category}/{image_filename}

Example:
    datasets/historical/images/flood/kerala_2018.jpg
    datasets/historical/images/cyclone/amphan_2020.jpg

Outputs
-------
    datasets/historical/index/disaster.index   — FAISS IndexFlatIP (512-d)
    datasets/historical/index/metadata.json    — ordered list of indexed events

Usage
-----
    python src/retrieval/build_index.py
    python src/retrieval/build_index.py --dry-run   # check images without building
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np

# Bootstrap so src/models/ is importable from any working directory.
_SRC  = Path(__file__).resolve().parent.parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
logger = logging.getLogger(__name__)

EVENTS_JSON  = _ROOT / "datasets" / "historical" / "historical_events.json"
IMAGES_ROOT  = _ROOT / "datasets" / "historical" / "images"
INDEX_DIR    = _ROOT / "datasets" / "historical" / "index"
INDEX_FILE   = INDEX_DIR / "disaster.index"
METADATA_FILE = INDEX_DIR / "metadata.json"
EMBEDDING_DIM = 512   # CLIP ViT-B/32


def load_events() -> list[dict]:
    data = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    return data.get("events", [])


def resolve_image(event: dict) -> Path | None:
    """Return the local image path for an event, or None if not found."""
    filename = event.get("image_filename", "")
    if not filename:
        return None
    path = IMAGES_ROOT / filename
    return path if path.exists() else None


def build_index(events: list[dict], dry_run: bool = False) -> None:
    from models.clip_model import embed_image

    available   = [(e, resolve_image(e)) for e in events]
    with_images = [(e, p) for e, p in available if p is not None]
    missing     = [e for e, p in available if p is None]

    logger.info("Events in database : %d", len(events))
    logger.info("Images found       : %d", len(with_images))
    logger.info("Images missing     : %d", len(missing))

    if missing:
        logger.warning("Missing images (will be skipped):")
        for e in missing:
            logger.warning("  %-40s  → datasets/historical/images/%s",
                           e["name"], e.get("image_filename", "(no filename)"))
        logger.warning(
            "\nTo add images, place JPEG files at the paths above."
            "\nSee datasets/historical/historical_events.json for the 'wikipedia_search'"
            " field — use those search terms to find appropriate CC-licensed images."
        )

    if not with_images:
        logger.error("No images available. Cannot build index. Add images and re-run.")
        sys.exit(1)

    if dry_run:
        logger.info("\n[dry-run] Would index %d events. Exiting without writing.", len(with_images))
        return

    logger.info("\nBuilding FAISS index (IndexFlatIP, dim=%d)...", EMBEDDING_DIM)

    index     = faiss.IndexFlatIP(EMBEDDING_DIM)
    metadata: list[dict] = []

    for i, (event, img_path) in enumerate(with_images):
        logger.info("  [%3d/%3d] %s", i + 1, len(with_images), event["name"])
        try:
            vec = embed_image(str(img_path))           # (512,) float32, unit-norm
            vec = vec.reshape(1, -1)
            index.add(vec)
            metadata.append({
                "id":                  event["id"],
                "name":                event["name"],
                "year":                event["year"],
                "category":            event["category"],
                "location":            event["location"],
                "description":         event["description"],
                "casualties":          event.get("casualties"),
                "affected_population": event.get("affected_population", ""),
                "damage_usd_billion":  event.get("damage_usd_billion"),
                "source":              event.get("source", ""),
                "image_filename":      event.get("image_filename", ""),
            })
        except Exception as exc:
            logger.error("  FAILED %s: %s", event["name"], exc)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "\nIndex complete: %d events indexed → %s",
        index.ntotal,
        INDEX_FILE,
    )
    logger.info("Metadata       : %s", METADATA_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS historical disaster index")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check image availability without building the index")
    args = parser.parse_args()

    events = load_events()
    build_index(events, dry_run=args.dry_run)
