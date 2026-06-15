"""
download_historical_images.py
Fetches one representative image per historical event from Wikipedia's
pageimages API (CC-licensed thumbnails) and saves them under:
    datasets/historical/images/{category}/{image_filename}

Then runs the FAISS index builder automatically.

Usage
-----
    python scripts/download_historical_images.py
    python scripts/download_historical_images.py --skip-existing   # resume partial run
    python scripts/download_historical_images.py --no-build        # download only, don't build index
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

ROOT        = Path(__file__).resolve().parent.parent
EVENTS_JSON = ROOT / "datasets" / "historical" / "historical_events.json"
IMAGES_ROOT = ROOT / "datasets" / "historical" / "images"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")
logger = logging.getLogger(__name__)

# Wikipedia article title for each event (used by pageimages API)
WIKI_ARTICLES: dict[str, str] = {
    "india_flood_kerala_2018":          "2018 Kerala floods",
    "india_flood_chennai_2015":         "2015 South India floods",
    "india_flood_assam_2022":           "2022 Assam floods",
    "india_flood_bihar_2019":           "2019 Bihar floods",
    "pak_flood_super_2010":             "2010 Pakistan floods",
    "india_flood_uttarakhand_2013":     "2013 North India floods",
    "india_flood_odisha_2020":          "2020 Odisha cyclone",
    "india_wildfire_uttarakhand_2021":  "2021 Uttarakhand forest fires",
    "india_wildfire_himachal_2023":     "2023 Himachal Pradesh monsoon",
    "australia_wildfire_blacksummer_2019": "2019–20 Australian bushfire season",
    "usa_wildfire_camp_2018":           "Camp Fire (2018)",
    "canada_wildfire_2023":             "2023 Canadian wildfires",
    "greece_wildfire_2021":             "2021 Greece wildfires",
    "nepal_earthquake_2015":            "2015 Nepal earthquake",
    "india_earthquake_gujarat_2001":    "2001 Gujarat earthquake",
    "turkey_earthquake_2023":           "2023 Turkey–Syria earthquake",
    "japan_earthquake_tohoku_2011":     "2011 Tōhoku earthquake and tsunami",
    "haiti_earthquake_2010":            "2010 Haiti earthquake",
    "india_landslide_kedarnath_2013":   "2013 Kedarnath floods",
    "india_landslide_wayanad_2024":     "2024 Wayanad landslides",
    "india_landslide_himachal_2023":    "2023 Himachal Pradesh monsoon",
    "india_landslide_maharashtra_2021": "2021 Maharashtra floods",
    "srilanka_landslide_aranayake_2016": "2016 Sri Lanka floods",
    "india_cyclone_amphan_2020":        "Cyclone Amphan",
    "india_cyclone_fani_2019":          "Cyclone Fani",
    "india_cyclone_yaas_2021":          "Cyclone Yaas",
    "india_cyclone_tauktae_2021":       "Cyclone Tauktae",
    "india_cyclone_biparjoy_2023":      "Cyclone Biparjoy",
    "india_cyclone_phailin_2013":       "Cyclone Phailin",
    "mozambique_cyclone_idai_2019":     "Cyclone Idai",
}

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS  = {"User-Agent": "VLM-DisasterAnalyzer/1.0 (ISRO Research Project)"}
THUMB_SIZE = 800   # px — large enough for CLIP's 224×224 crop


def fetch_wiki_thumbnail(article_title: str) -> str | None:
    """Return the thumbnail URL for a Wikipedia article, or None if unavailable."""
    try:
        resp = requests.get(
            WIKI_API,
            params={
                "action":      "query",
                "titles":      article_title,
                "prop":        "pageimages",
                "format":      "json",
                "pithumbsize": THUMB_SIZE,
                "pilicense":   "any",
            },
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {})
            if thumb.get("source"):
                return thumb["source"]
    except Exception as exc:
        logger.warning("Wikipedia API error for '%s': %s", article_title, exc)
    return None


def download_image(url: str, dest: Path) -> bool:
    """Download image from url to dest. Returns True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as exc:
        logger.error("Download failed from %s: %s", url, exc)
        return False


def main(skip_existing: bool = False, no_build: bool = False) -> None:
    data   = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    events = data.get("events", [])

    succeeded, failed, skipped = [], [], []

    for event in events:
        eid      = event["id"]
        dest     = IMAGES_ROOT / event["image_filename"]
        article  = WIKI_ARTICLES.get(eid)

        if skip_existing and dest.exists():
            logger.info("SKIP   %s (already exists)", event["name"])
            skipped.append(eid)
            continue

        if not article:
            logger.warning("NO ARTICLE  %s — add to WIKI_ARTICLES dict", event["name"])
            failed.append(eid)
            continue

        logger.info("Fetching  %s  → '%s'", event["name"], article)
        thumb_url = fetch_wiki_thumbnail(article)

        if not thumb_url:
            logger.warning("No thumbnail found for '%s'", article)
            failed.append(eid)
            time.sleep(0.5)
            continue

        ok = download_image(thumb_url, dest)
        if ok:
            size_kb = round(dest.stat().st_size / 1024, 1)
            logger.info("  ✓  %s  (%s KB)", dest.relative_to(ROOT), size_kb)
            succeeded.append(eid)
        else:
            failed.append(eid)

        time.sleep(0.3)   # polite delay between API calls

    logger.info("\n── Download summary ──────────────────────────────")
    logger.info("  Downloaded : %d", len(succeeded))
    logger.info("  Skipped    : %d", len(skipped))
    logger.info("  Failed     : %d", len(failed))

    if failed:
        logger.warning("  Failed events: %s", failed)
        logger.warning("  You can manually place images at the paths shown in --dry-run output.")

    if not no_build and (succeeded or skipped):
        logger.info("\nBuilding FAISS index...")
        sys.path.insert(0, str(ROOT / "src"))
        from retrieval.build_index import load_events, build_index
        build_index(load_events(), dry_run=False)
    elif not no_build:
        logger.warning("No images available — skipping index build.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical disaster images and build FAISS index")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip events whose images already exist on disk")
    parser.add_argument("--no-build", action="store_true",
                        help="Download images but do not rebuild the FAISS index")
    args = parser.parse_args()
    main(skip_existing=args.skip_existing, no_build=args.no_build)
