"""
patch_index.py — Add any events missing from the FAISS index (e.g. failed during build).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

import faiss
from models.clip_model import embed_image

IDX_PATH  = ROOT / "datasets" / "historical" / "index" / "disaster.index"
META_PATH = ROOT / "datasets" / "historical" / "index" / "metadata.json"
EVENTS_F  = ROOT / "datasets" / "historical" / "historical_events.json"
IMG_ROOT  = ROOT / "datasets" / "historical" / "images"

idx  = faiss.read_index(str(IDX_PATH))
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
all_events = json.loads(EVENTS_F.read_text(encoding="utf-8"))["events"]

indexed_ids = {m["id"] for m in meta}
missing = [e for e in all_events if e["id"] not in indexed_ids]

print(f"Already indexed : {idx.ntotal}")
print(f"Missing entries : {len(missing)}")

added = 0
for e in missing:
    img_path = IMG_ROOT / e["image_filename"]
    if not img_path.exists():
        print(f"  [SKIP] image not on disk: {img_path}")
        continue
    print(f"  [embed] {e['id']}  ->  {e['image_filename']}")
    vec = embed_image(str(img_path)).reshape(1, -1)
    idx.add(vec)
    meta.append({
        "id":                  e["id"],
        "name":                e["name"],
        "year":                e["year"],
        "category":            e["category"],
        "location":            e["location"],
        "description":         e["description"],
        "casualties":          e.get("casualties"),
        "affected_population": e.get("affected_population", ""),
        "damage_usd_billion":  e.get("damage_usd_billion"),
        "source":              e.get("source", ""),
        "image_filename":      e.get("image_filename", ""),
    })
    added += 1

faiss.write_index(idx, str(IDX_PATH))
META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nPatched {added} entries.")
print(f"Total vectors now: {idx.ntotal}")
