"""
config.py — 75-video VIDI research subset pipeline configuration.

Target: 5 disaster categories × 15 videos = 75 videos total.

VIDI label → research category mapping:
  flood      ← flooded (primary), heavy_rainfall, storm_surge
  wildfire   ← wildfire (primary), on_fire, fire_whirl
  earthquake ← earthquake (primary), collapsed
  landslide  ← landslide (primary), mudslide_mudflow, rockslide_rockfall
  cyclone    ← tropical_cyclone (primary), tornado, derecho

Environment variable:
  Set VLM_PROJECT_ROOT before importing to override the project root.
  In Google Colab:
      import os
      os.environ["VLM_PROJECT_ROOT"] = "/content/drive/MyDrive/ISRO_Project"
"""

import os
from pathlib import Path

# ── Project root (auto-detected or overridden via env var) ────────────────────
# Default: two levels up from scripts/video_pipeline/ → project repo root.
# In Google Colab, VLM_PROJECT_ROOT should point to your Google Drive folder
# so all data is persisted between Colab sessions.
PROJECT_ROOT = Path(os.environ.get(
    "VLM_PROJECT_ROOT",
    Path(__file__).resolve().parents[2],
))

# ── Dataset layout ─────────────────────────────────────────────────────────────
DATASET_ROOT    = PROJECT_ROOT / "datasets" / "video_dataset"
RAW_VIDEOS_ROOT = DATASET_ROOT / "raw_videos"
FRAMES_ROOT     = DATASET_ROOT / "extracted_frames"
METADATA_DIR    = DATASET_ROOT / "metadata"
EVALUATION_DIR  = DATASET_ROOT / "evaluation"

# Temporary cache for full YouTube videos before trimming.
# Cleared automatically after each successful trim to save space.
YT_CACHE_DIR    = METADATA_DIR / "_yt_cache"

# ── Research targets ───────────────────────────────────────────────────────────
TARGET_CATEGORIES   = ["flood", "wildfire", "earthquake", "landslide", "cyclone"]
VIDEOS_PER_CATEGORY = 15
TOTAL_VIDEOS        = len(TARGET_CATEGORIES) * VIDEOS_PER_CATEGORY  # 75

# ── VIDI fine-grained label → research category ────────────────────────────────
# Keys are VIDI CSV filename slugs (must match exactly, including the
# "oil_spil" typo present in the upstream repository).
# Values are the 5 research category names used in this pipeline.
VIDI_TO_RESEARCH: dict[str, str] = {
    # Flood
    "flooded":              "flood",
    "heavy_rainfall":       "flood",
    "storm_surge":          "flood",
    # Wildfire
    "wildfire":             "wildfire",
    "on_fire":              "wildfire",
    "fire_whirl":           "wildfire",
    # Earthquake
    "earthquake":           "earthquake",
    "collapsed":            "earthquake",   # structural collapse (earthquake aftermath)
    # Landslide
    "landslide":            "landslide",
    "mudslide_mudflow":     "landslide",
    "rockslide_rockfall":   "landslide",
    # Cyclone
    "tropical_cyclone":     "cyclone",
    "tornado":              "cyclone",
    "derecho":              "cyclone",
}

# Priority waterfall within each category.
# Primary label is tried first; secondary/tertiary labels fill the gap
# if the primary label does not have enough quality clips.
LABEL_PRIORITY: dict[str, list[str]] = {
    "flood":      ["flooded", "heavy_rainfall", "storm_surge"],
    "wildfire":   ["wildfire", "on_fire", "fire_whirl"],
    "earthquake": ["earthquake", "collapsed"],
    "landslide":  ["landslide", "mudslide_mudflow", "rockslide_rockfall"],
    "cyclone":    ["tropical_cyclone", "tornado", "derecho"],
}

# All unique VIDI labels needed (derived — do not edit manually)
ALL_REQUIRED_LABELS: list[str] = [
    label for labels in LABEL_PRIORITY.values() for label in labels
]

# ── VIDI GitHub raw source ─────────────────────────────────────────────────────
VIDI_REPO_RAW = "https://raw.githubusercontent.com/vididataset/VIDI/main/data"

# ── Clip selection quality filters ────────────────────────────────────────────
MIN_DURATION_S      = 2     # discard clips shorter than this (seconds)
MAX_DURATION_S      = 300   # discard clips longer than 5 minutes
PREFER_ENGLISH      = True  # sort English-language clips first; fall back to all
MAX_CLIPS_PER_YT_ID = 2     # max clips from the same YouTube video (diversity guard)

# ── yt-dlp / ffmpeg download settings ─────────────────────────────────────────
MAX_RESOLUTION      = 720   # maximum video height to request (720p)
DOWNLOAD_WORKERS    = 4     # parallel download threads (conservative for Colab)
RETRY_ATTEMPTS      = 3
RETRY_DELAY_S       = 5     # base delay between retries (doubles each attempt)
MIN_FILE_SIZE_KB    = 50    # files smaller than this are treated as corrupted
YT_TIMEOUT_S        = 600   # yt-dlp per-video timeout in seconds

# ── Frame extraction ───────────────────────────────────────────────────────────
FRAMES_PER_VIDEO    = 4     # frames sampled uniformly per video
FRAME_FORMAT        = "jpg"
FRAME_QUALITY       = 90    # JPEG quality (1–100)

# ── VLM evaluation ─────────────────────────────────────────────────────────────
# Override in the notebook before importing: os.environ["VLM_BACKEND"] = "https://..."
VLM_BACKEND_URL     = os.environ.get("VLM_BACKEND", "http://localhost:8000")
VLM_TIMEOUT_S       = 60

# ── Excel styling (matches VLM Disaster Analyzer warm ivory theme) ─────────────
HEADER_BG   = "543A14"
HEADER_FG   = "FFF0DC"
ALT_ROW_BG  = "F7F4EE"
CORRECT_BG  = "C6EFCE"
WRONG_BG    = "FFC7CE"
