"""
config.py — Central configuration for the VIDI video dataset pipeline.

All paths, constants, category mappings, and hyper-parameters live here.
Every other script imports from this module — change values once, affects all.

Environment override:
    Set VLM_PROJECT_ROOT=/your/path to relocate the project root
    (useful in Google Colab: os.environ["VLM_PROJECT_ROOT"] = "/content/vlm-disaster-analyzer")
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
# Default: two levels up from this file (scripts/video_dataset/ → scripts/ → root)
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT",
                                   Path(__file__).resolve().parents[2]))

# ── Dataset roots ─────────────────────────────────────────────────────────────
DATASETS_ROOT    = PROJECT_ROOT / "datasets"
IMAGE_DATASETS   = DATASETS_ROOT / "image_datasets"
VIDEO_DATASETS   = DATASETS_ROOT / "video_datasets"
VIDI_ROOT        = VIDEO_DATASETS / "vidi_dataset"

# ── VIDI internal directories ─────────────────────────────────────────────────
RAW_VIDEOS       = VIDI_ROOT / "raw_videos"       # full YouTube downloads (untrimmed)
PROCESSED_BASE   = VIDI_ROOT / "processed_videos"  # trimmed clips  /{split}/{label}/
THUMBNAILS_BASE  = VIDI_ROOT / "thumbnails"        # keyframes       /{split}/{label}/
METADATA_DIR     = VIDI_ROOT / "metadata"          # CSV manifests
ANNOTATIONS_DIR  = VIDI_ROOT / "annotations"       # raw VIDI CSVs (one per category)
REPORTS_DIR      = VIDI_ROOT / "reports"           # statistics + quality reports
LOGS_DIR         = VIDI_ROOT / "logs"              # per-run download logs (.jsonl)

SPLITS = ("train", "val", "test")

# ── VIDI GitHub source ────────────────────────────────────────────────────────
VIDI_REPO_RAW = "https://raw.githubusercontent.com/vididataset/VIDI/main/data"

# All 43 VIDI category slugs (match the CSV filenames exactly, including "oil_spil" typo)
VIDI_CATEGORIES = [
    # Transportation (9)
    "airplane_accident", "bicycle_accident", "bus_accident", "car_accident",
    "motorcycle_accident", "ship_accident", "train_accident", "truck_accident",
    "van_accident",
    # Weather (11)
    "derecho", "drought", "dust_devil", "dust_sand_storm", "fog", "hailstorm",
    "heavy_rainfall", "ice_storm", "thunderstorm", "tornado", "tropical_cyclone",
    # Natural Disasters (8)
    "earthquake", "landslide", "mudslide_mudflow", "rockslide_rockfall",
    "snowslide_avalanche", "sinkhole", "volcanic_eruption", "wildfire",
    # Environmental & Infrastructure (15)
    "blocked", "burned", "collapsed", "damaged", "dirty_contamined",
    "fire_whirl", "flooded", "nuclear_explosion", "oil_spil", "on_fire",
    "storm_surge", "traffic_jam", "under_construction", "with_smoke", "snow_covered",
]

# ── Fine-grained → broad disaster category mapping ───────────────────────────
# Aligns VIDI's 43 incident types with the 8 broader classes used in the
# existing VLM Disaster Analyzer (CLIP image pipeline).
CATEGORY_MAP: dict[str, str] = {
    # Flood / Water
    "flooded":              "Flood",
    "heavy_rainfall":       "Flood",
    "storm_surge":          "Flood",
    # Wildfire
    "wildfire":             "Wildfire",
    "on_fire":              "Wildfire",
    "burned":               "Wildfire",
    "fire_whirl":           "Wildfire",
    # Earthquake
    "earthquake":           "Earthquake",
    # Landslide / Geological movement
    "landslide":            "Landslide",
    "mudslide_mudflow":     "Landslide",
    "rockslide_rockfall":   "Landslide",
    "sinkhole":             "Landslide",
    "snowslide_avalanche":  "Landslide",
    # Cyclone / Severe storm
    "tropical_cyclone":     "Cyclone",
    "tornado":              "Cyclone",
    "derecho":              "Cyclone",
    "thunderstorm":         "Storm",
    "hailstorm":            "Storm",
    "ice_storm":            "Storm",
    # Volcanic
    "volcanic_eruption":    "Volcanic Eruption",
    # Infrastructure damage
    "collapsed":            "Infrastructure Damage",
    "damaged":              "Infrastructure Damage",
    "blocked":              "Infrastructure Damage",
    "traffic_jam":          "Infrastructure Damage",
    "under_construction":   "Infrastructure Damage",
    # Transportation accidents
    "airplane_accident":    "Transportation Accident",
    "bicycle_accident":     "Transportation Accident",
    "bus_accident":         "Transportation Accident",
    "car_accident":         "Transportation Accident",
    "motorcycle_accident":  "Transportation Accident",
    "ship_accident":        "Transportation Accident",
    "train_accident":       "Transportation Accident",
    "truck_accident":       "Transportation Accident",
    "van_accident":         "Transportation Accident",
    # Environmental / Hazmat
    "dust_devil":           "Dust/Sand Storm",
    "dust_sand_storm":      "Dust/Sand Storm",
    "drought":              "Drought",
    "fog":                  "Weather Hazard",
    "nuclear_explosion":    "Explosion/CBRN",
    "oil_spil":             "Environmental Contamination",  # typo in VIDI source
    "oil_spill":            "Environmental Contamination",  # normalised spelling
    "dirty_contamined":     "Environmental Contamination",
    "with_smoke":           "Smoke/Air Quality",
    "snow_covered":         "Winter Event",
}

# ── Reverse lookup: broad → list of fine-grained labels ──────────────────────
BROAD_TO_FINE: dict[str, list[str]] = {}
for _fine, _broad in CATEGORY_MAP.items():
    BROAD_TO_FINE.setdefault(_broad, []).append(_fine)

# ── Download settings ─────────────────────────────────────────────────────────
MAX_RESOLUTION   = 720    # max video height to request from yt-dlp
DOWNLOAD_WORKERS = 8      # parallel yt-dlp + ffmpeg workers
RETRY_ATTEMPTS   = 3      # download retry count per video
RETRY_DELAY_BASE = 5      # seconds; multiplied exponentially on each retry
MIN_FILE_SIZE_KB = 50     # minimum healthy file size (flags corrupted/empty files)

# ── Video preprocessing targets ───────────────────────────────────────────────
TARGET_FPS    = 16        # standard frame-rate for model input
TARGET_HEIGHT = 256       # output height in pixels
TARGET_WIDTH  = 256       # output width  in pixels (square crop + pad)
VIDEO_CODEC   = "libx264"
AUDIO_CODEC   = "aac"
CRF           = 22        # H.264 constant rate factor (lower = higher quality)

# ── Dataset split ratios ──────────────────────────────────────────────────────
# Split is performed at the YouTube-video level (not clip level) to prevent
# temporal data leakage between splits.
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9
RANDOM_SEED = 42

# ── Language filter (optional) ────────────────────────────────────────────────
# Set to None to include all languages; or e.g. ["EN"] for English-only subset.
LANGUAGE_FILTER: list[str] | None = None
