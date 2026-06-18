"""
verify_retrieval_e2e.py — End-to-end retrieval verification.

Sends real disaster images to the running backend (/predict/disaster)
and verifies that similar_events is populated in the response.

Usage:
    python scripts/verify_retrieval_e2e.py [--host http://localhost:8000]

Prerequisites:
    uvicorn backend.main:app --port 8000  (must be running)
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

ROOT    = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "datasets" / "historical" / "images"

# Images chosen for strong visual flood/cyclone/earthquake cues
TEST_CASES = [
    {
        "label":    "FLOOD",
        "images": [
            IMG_DIR / "flood" / "flood_assam_2022_01.jpg",
            IMG_DIR / "flood" / "flood_germany_2021_01.jpg",
            IMG_DIR / "flood" / "flood_kerala_2018_01.jpg",
        ],
        "expect": "flood",
    },
    {
        "label":    "CYCLONE",
        "images": [
            IMG_DIR / "cyclone" / "cyclone_amphan_2020_01.jpg",
            IMG_DIR / "cyclone" / "cyclone_fani_2019_01.jpg",
            IMG_DIR / "cyclone" / "cyclone_nargis_2008_01.jpg",
        ],
        "expect": "cyclone",
    },
    {
        "label":    "EARTHQUAKE",
        "images": [
            IMG_DIR / "earthquake" / "earthquake_gujarat_2001_01.png",
            IMG_DIR / "earthquake" / "earthquake_afghanistan_2022_02.png",
            IMG_DIR / "earthquake" / "earthquake_nepal_2015_03.jpg",
            IMG_DIR / "earthquake" / "earthquake_turkey_2023_01.jpg",
        ],
        "expect": "earthquake",
    },
]

SEP = "=" * 70

TIMEOUT = 600   # 10 min — Qwen on CPU takes ~7 min first run, ~2 min warm


def post_image(client: httpx.Client, path: Path, host: str) -> dict:
    suffix = path.suffix.lower()
    mime   = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        resp = client.post(
            f"{host}/predict/disaster",
            files={"file": (path.name, f, mime)},
            timeout=TIMEOUT,
        )
    resp.raise_for_status()
    return resp.json()


def render_similar_events(events: list[dict]) -> str:
    if not events:
        return "  (empty — no similar events returned)\n"
    lines = []
    for i, ev in enumerate(events, 1):
        sim  = ev.get("similarity", 0)
        bar  = "#" * int(sim / 5)
        mark = "HIGH" if sim >= 80 else ("MED" if sim >= 65 else "low")
        lines.append(
            f"  [{i}] {ev.get('event','?'):<34} {ev.get('year','?')}  "
            f"sim={sim:>5.1f}%  [{mark}]  cat={ev.get('category','?')}"
        )
        lines.append(f"       loc: {ev.get('location','?')}")
        lines.append(f"       {ev.get('description','')[:90]}")
        lines.append(f"       {'|' * int(sim / 3)}")
    return "\n".join(lines) + "\n"


def first_existing(images: list[Path]) -> Path | None:
    for p in images:
        if p.exists():
            return p
    return None


def main(host: str) -> None:
    print(SEP)
    print("  END-TO-END RETRIEVAL VERIFICATION  v2")
    print(f"  Backend: {host}  |  Timeout: {TIMEOUT}s per request")
    print(SEP)

    # Health check
    try:
        with httpx.Client() as c:
            r = c.get(f"{host}/", timeout=5)
        print(f"\n  Health check: {r.status_code} {r.json().get('status','?')}\n")
    except Exception as e:
        print(f"\n  ERROR: Backend not reachable at {host}: {e}")
        print("  Start it with: uvicorn backend.main:app --port 8000")
        sys.exit(1)

    results = []

    with httpx.Client() as client:
        for tc in TEST_CASES:
            img_path = first_existing(tc["images"])
            print(SEP)
            print(f"  TEST: {tc['label']}")
            if not img_path:
                print("  SKIP — no image found on disk\n")
                results.append({"label": tc["label"], "status": "SKIP"})
                continue
            print(f"  Image: {img_path.relative_to(ROOT)}")
            print(SEP)

            t0 = time.perf_counter()
            try:
                data = post_image(client, img_path, host)
            except httpx.TimeoutException:
                print(f"  ERROR: Request timed out after {TIMEOUT}s")
                results.append({"label": tc["label"], "status": "TIMEOUT"})
                continue
            except httpx.HTTPStatusError as e:
                print(f"  HTTP ERROR {e.response.status_code}: {e.response.text[:200]}")
                results.append({"label": tc["label"], "status": "HTTP_ERROR"})
                continue
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"label": tc["label"], "status": "ERROR"})
                continue
            elapsed = time.perf_counter() - t0

            cat    = data.get("category", "?")
            conf   = data.get("classification_confidence", 0)
            sev    = data.get("severity", "?")
            evts   = data.get("similar_events", [])
            ms     = data.get("processing_time_ms", elapsed * 1000)
            models = data.get("active_models", [])

            print(f"\n  CLIP category   : {cat} ({conf:.1f}% confidence)")
            print(f"  Qwen severity   : {sev}")
            print(f"  Active models   : {models}")
            print(f"  Processing time : {ms:.0f} ms  (wall: {elapsed:.1f}s)")
            print(f"\n  similar_events  : {len(evts)} events returned")

            print(f"\n  --- SimilarEventsCard render preview: ---")
            print(render_similar_events(evts))

            # Full API response snippet (skip raw model outputs)
            snippet = {}
            for k, v in data.items():
                if k in ("clip_raw", "qwen_raw"):
                    continue
                if isinstance(v, str) and len(v) > 120:
                    snippet[k] = v[:120] + "..."
                elif k == "similar_events":
                    snippet[k] = f"[{len(v)} events — see above]"
                else:
                    snippet[k] = v
            print("  --- Full API response (trimmed): ---")
            print("  " + json.dumps(snippet, indent=4).replace("\n", "\n  "))

            ok = len(evts) > 0
            results.append({
                "label":    tc["label"],
                "status":   "PASS" if ok else "FAIL",
                "category": cat,
                "severity": sev,
                "events":   len(evts),
                "sim_top1": evts[0]["similarity"] if evts else 0,
                "sim_top1_cat": evts[0]["category"] if evts else "—",
            })
            print()

    # Summary
    print(SEP)
    print("  SUMMARY")
    print(SEP)
    all_ok = True
    for r in results:
        st = r.get("status", "?")
        mark = {"PASS": "PASS", "FAIL": "FAIL", "TIMEOUT": "TIME", "SKIP": "SKIP"}.get(st, "???")
        if mark != "PASS":
            all_ok = False
        evts = r.get("events", "?")
        sim  = r.get("sim_top1", 0)
        scat = r.get("sim_top1_cat", "—")
        print(
            f"  [{mark}]  {r['label']:<12} "
            f"category={r.get('category','?'):<24} "
            f"events={evts}  top1_sim={sim:.1f}%  top1_cat={scat}"
        )

    print()
    if all_ok:
        print("  All tests PASSED — retrieval fully integrated and verified.")
    else:
        print("  Some tests need attention — review output above.")
    print(SEP)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:8000")
    args = p.parse_args()
    main(args.host)
