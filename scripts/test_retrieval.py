"""
test_retrieval.py — Verify FAISS index and run retrieval accuracy tests.

Usage:
    python scripts/test_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

import faiss
import numpy as np
from models.clip_model import embed_image


IDX_PATH  = ROOT / "datasets" / "historical" / "index" / "disaster.index"
META_PATH = ROOT / "datasets" / "historical" / "index" / "metadata.json"
IMG_ROOT  = ROOT / "datasets" / "historical" / "images"


def load_index():
    idx  = faiss.read_index(str(IDX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return idx, meta


def query(idx, meta, img_path: str, top_k: int = 5):
    vec = embed_image(img_path).reshape(1, -1)
    distances, indices = idx.search(vec, top_k)
    results = []
    for dist, i in zip(distances[0], indices[0]):
        if i < 0 or i >= len(meta):
            continue
        ev  = meta[i]
        sim = round(float(dist) * 100, 1)
        results.append({
            "event":    ev["name"],
            "year":     ev["year"],
            "category": ev["category"],
            "location": ev["location"],
            "sim":      sim,
        })
    return results


def print_section(title: str):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def main():
    # ── Index verification ──────────────────────────────────────────
    print_section("INDEX VERIFICATION")
    idx, meta = load_index()

    idx_size_kb = IDX_PATH.stat().st_size / 1024
    meta_size_kb = META_PATH.stat().st_size / 1024

    print(f"  disaster.index   {idx_size_kb:>8.1f} KB  ({IDX_PATH.stat().st_size:,} bytes)")
    print(f"  metadata.json    {meta_size_kb:>8.1f} KB  ({META_PATH.stat().st_size:,} bytes)")
    print()
    print(f"  Total vectors    {idx.ntotal}")
    print(f"  Embedding dim    {idx.d}")
    print(f"  Index type       {type(idx).__name__}")
    print()

    cats: dict[str, int] = {}
    for m in meta:
        c = m["category"]
        cats[c] = cats.get(c, 0) + 1
    print("  Category breakdown:")
    for k, v in sorted(cats.items()):
        bar = "#" * (v // 2)
        print(f"    {k:<14} {v:>4}  {bar}")

    # ── Retrieval tests ─────────────────────────────────────────────
    test_cases = [
        (
            "FLOOD",
            "flood/flood_kerala_2018_05.jpg",
            "Kerala Floods (2018) — monsoon flooding in South India",
        ),
        (
            "CYCLONE",
            "cyclone/cyclone_fani_2019_02.jpg",
            "Cyclone Fani (2019) — tropical cyclone, Bay of Bengal",
        ),
        (
            "EARTHQUAKE",
            "earthquake/earthquake_nepal_2015_01.jpg",
            "Nepal Earthquake (2015) — Gorkha earthquake, Kathmandu",
        ),
    ]

    for category, rel_path, description in test_cases:
        print_section(f"RETRIEVAL TEST — {category}")
        img_path = str(IMG_ROOT / rel_path)
        print(f"  Query image : {rel_path}")
        print(f"  Description : {description}")
        print()

        results = query(idx, meta, img_path, top_k=5)

        # count how many top-5 match the expected category
        same_cat = sum(1 for r in results if r["category"] == category.lower())
        print(f"  Top-5 results  (category match: {same_cat}/5):")
        print()
        for rank, r in enumerate(results, 1):
            match = "OK" if r["category"] == category.lower() else "--"
            print(
                f"    [{rank}] {match}  {r['event']:<30}  {r['year']}  "
                f"sim={r['sim']:>5.1f}%  [{r['category']}]"
            )
            print(f"         {r['location']}")
        print()

    # ── Cross-category retrieval accuracy ───────────────────────────
    print_section("CROSS-CATEGORY ACCURACY SUMMARY")
    print("  Querying 1 sample image per event — correct category in top-1?")
    print()

    events_tested = [
        # Flood
        ("flood", "flood/flood_assam_2022_01.jpg",        "Assam Floods"),
        ("flood", "flood/flood_chennai_2015_01.jpg",      "Chennai Floods"),
        ("flood", "flood/flood_germany_2021_01.jpg",      "Germany Floods"),
        # Cyclone
        ("cyclone", "cyclone/cyclone_amphan_2020_01.jpg", "Cyclone Amphan"),
        ("cyclone", "cyclone/cyclone_yaas_2021_01.png",   "Cyclone Yaas"),
        ("cyclone", "cyclone/cyclone_nargis_2008_01.jpg", "Cyclone Nargis"),
        # Earthquake
        ("earthquake", "earthquake/earthquake_gujarat_2001_01.png", "Bhuj Earthquake"),
        ("earthquake", "earthquake/earthquake_turkey_2023_01.jpg",  "Turkey Earthquake"),
        ("earthquake", "earthquake/earthquake_japan_2011_01.png",   "Tohoku Earthquake"),
    ]

    correct_top1 = 0
    correct_top3 = 0
    total = 0

    for expected_cat, rel_path, event_name in events_tested:
        img_path = str(IMG_ROOT / rel_path)
        if not Path(img_path).exists():
            print(f"  [SKIP] {event_name}  — image not found")
            continue

        results = query(idx, meta, img_path, top_k=3)
        top1_cat = results[0]["category"] if results else "—"
        top3_cats = [r["category"] for r in results]

        ok1 = top1_cat == expected_cat
        ok3 = expected_cat in top3_cats
        correct_top1 += ok1
        correct_top3 += ok3
        total += 1

        mark1 = "OK" if ok1 else "--"
        mark3 = "OK" if ok3 else "--"
        print(
            f"  {mark1} top-1  {mark3} top-3  |  {event_name:<28}  "
            f"top1={results[0]['event'][:22]:<22} ({results[0]['sim']}%)"
        )

    print()
    print(f"  Top-1 accuracy : {correct_top1}/{total} = {correct_top1/total*100:.0f}%")
    print(f"  Top-3 accuracy : {correct_top3}/{total} = {correct_top3/total*100:.0f}%")
    print()


if __name__ == "__main__":
    main()
