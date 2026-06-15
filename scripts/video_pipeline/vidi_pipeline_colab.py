# %% [markdown]
# # VIDI Video Dataset Pipeline — Google Colab
#
# **Paper:** Sesver et al., *VIDI: A Video Dataset of Incidents*, IEEE IVMSP 2022
# **Dataset:** https://github.com/vididataset/VIDI
#
# ### What this notebook does
# 1. Installs dependencies and mounts Google Drive
# 2. Clones the VLM Disaster Analyzer repo (or uses an existing clone)
# 3. Fetches all 43 VIDI annotation CSVs directly from GitHub
# 4. Builds stratified train / val / test splits
# 5. Downloads English-language video clips via yt-dlp + ffmpeg
# 6. Verifies integrity with ffprobe
# 7. Extracts thumbnails
# 8. Generates dataset statistics and a Markdown report
#
# ### Storage estimates
# | Subset | Clips | Approx. Size |
# |--------|-------|-------------|
# | English only (default) | ~2,100 | ~8–12 GB |
# | All languages | ~4,500 | ~18–25 GB |
# | Raw downloads only | varies | ~80–120 GB |
#
# **Tip:** Point `VIDI_ROOT` to your Google Drive to persist across sessions.

# %% [markdown]
# ## Cell 1 — Install system dependencies

# %%
# Run this cell first — installs ffmpeg and yt-dlp
import subprocess, sys

subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "yt-dlp", "pandas", "scikit-learn", "tqdm", "requests", "matplotlib"],
               check=True)

print("✓ System dependencies installed")
# Confirm ffmpeg is working
result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
print(result.stdout.split("\n")[0])

# %% [markdown]
# ## Cell 2 — Mount Google Drive (optional but recommended)

# %%
# Skip this cell if you don't want to use Drive.
# Mounting Drive lets downloads survive Colab session resets.

USE_DRIVE = True   # ← set False to use ephemeral /content storage

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    # Choose your preferred location inside Drive:
    PROJECT_DIR = "/content/drive/MyDrive/vlm-disaster-analyzer"
else:
    PROJECT_DIR = "/content/vlm-disaster-analyzer"

print(f"Project directory: {PROJECT_DIR}")

# %% [markdown]
# ## Cell 3 — Clone or update the repository

# %%
import os
from pathlib import Path

REPO_URL = "https://github.com/YOUR_USERNAME/vlm-disaster-analyzer.git"
# ↑ Replace with your actual GitHub repo URL, or comment out the clone
#   if you're uploading the scripts manually.

if not Path(PROJECT_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, PROJECT_DIR], check=True)
    print(f"✓ Cloned to {PROJECT_DIR}")
else:
    print(f"✓ Directory already exists: {PROJECT_DIR}")

# Tell config.py where the project root is
os.environ["VLM_PROJECT_ROOT"] = PROJECT_DIR

# Add scripts to Python path so imports work
SCRIPTS_DIR = f"{PROJECT_DIR}/scripts/video_dataset"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

print(f"VLM_PROJECT_ROOT = {os.environ['VLM_PROJECT_ROOT']}")

# %% [markdown]
# ## Cell 4 — Verify folder structure

# %%
import config as cfg

# Print resolved paths
for name, path in [
    ("project root",    cfg.PROJECT_ROOT),
    ("raw_videos",      cfg.RAW_VIDEOS),
    ("processed_videos",cfg.PROCESSED_BASE),
    ("metadata",        cfg.METADATA_DIR),
    ("annotations",     cfg.ANNOTATIONS_DIR),
    ("reports",         cfg.REPORTS_DIR),
    ("logs",            cfg.LOGS_DIR),
]:
    cfg.Path(path).mkdir(parents=True, exist_ok=True)
    print(f"  {name:<20}: {path}")

print("\n✓ All directories ready")

# %% [markdown]
# ## Cell 5 — Fetch annotation CSVs and build master manifest

# %%
from generate_metadata import build_master

# Fetch all 43 category CSVs from the VIDI GitHub repo and merge them.
# This is fast (~2 min) and does not require any video downloads.
master = build_master()

print(f"\n{'='*50}")
print(f"Total clips:      {len(master):,}")
print(f"Categories:       {master['label'].nunique()}")
print(f"YouTube IDs:      {master['youtube_id'].nunique():,}")
print(f"Total duration:   {master['duration_sec'].sum()/3600:.1f} h")
print(f"Languages:        {sorted(master['lang'].unique())}")
master.head()

# %% [markdown]
# ## Cell 6 — Inspect category distribution

# %%
import pandas as pd

print("Broad category distribution:")
print(master.groupby("broad_category")["clip_id"].count()
      .sort_values(ascending=False)
      .to_string())

print("\nLanguage distribution:")
print(master["lang"].value_counts().to_string())

# %% [markdown]
# ## Cell 7 — Create train / val / test splits

# %%
from create_splits import create_splits

splits = create_splits(seed=42)

for name, df in splits.items():
    print(f"{name:<6}: {len(df):>5} clips  "
          f"from {df['youtube_id'].nunique():>4} videos  "
          f"across {df['label'].nunique()} categories")

# %% [markdown]
# ## Cell 8 — Download video clips
#
# **Options:**
# - `english_only=True`  → ~2,100 clips, ~8–12 GB, faster
# - `english_only=False` → ~4,500 clips, ~18–25 GB, all languages
#
# yt-dlp is rate-limited on Colab. If you hit `HTTP Error 429 (Too Many Requests)`:
# - Reduce `workers` to 2–3
# - Add `--sleep-interval 3 --max-sleep-interval 10` to the yt-dlp command in config
# - Use a VPN or different Colab runtime

# %%
from download_videos import download_all

# Recommended for first run: English-only, 4 workers
result = download_all(
    lang_filter=["EN"],   # Remove this line to download all languages
    workers=4,            # Reduce to 2 if you get 429 errors
    dry_run=False,        # Set True to simulate without downloading
)

print(f"\nDownload complete: {result['total_ok']:,} ok, {result['total_fail']} failed")

# %% [markdown]
# ## Cell 9 — Verify downloaded clips

# %%
from verify_videos import run_verification

report = run_verification()

# Show status breakdown
print(report["status"].value_counts().to_string())
print(f"\nOverall success rate: {(report['status'] == 'valid').mean()*100:.1f}%")

# %% [markdown]
# ## Cell 10 — Extract thumbnails

# %%
from preprocess_videos import run_preprocessing

proc = run_preprocessing(normalize=False, workers=4)

print(f"Thumbnails extracted: {(proc['thumb_status'] == 'ok').sum():,}")

# %% [markdown]
# ## Cell 11 — Generate statistics and reports

# %%
from dataset_statistics import generate_reports

generate_reports(plots=True)

# Display the summary markdown
summary_path = cfg.REPORTS_DIR / "dataset_summary.md"
print(summary_path.read_text(encoding="utf-8"))

# %% [markdown]
# ## Cell 12 — Display class distribution chart

# %%
from IPython.display import Image as IPImage, display

chart = cfg.REPORTS_DIR / "class_distribution.png"
if chart.exists():
    display(IPImage(filename=str(chart)))
else:
    print("Chart not found — run Cell 11 first.")

# %% [markdown]
# ## Cell 13 — Dataset summary statistics

# %%
master   = pd.read_csv(cfg.METADATA_DIR / "master_annotations.csv")
fine_csv = pd.read_csv(cfg.REPORTS_DIR  / "class_distribution.csv")
stor_csv = pd.read_csv(cfg.REPORTS_DIR  / "storage_report.csv")

print("=" * 55)
print(f"  VIDI DATASET — FINAL STATISTICS")
print("=" * 55)
print(f"  Total clips           : {len(master):,}")
print(f"  Unique YouTube videos : {master['youtube_id'].nunique():,}")
print(f"  Fine categories       : {master['label'].nunique()}")
print(f"  Broad categories      : {master['broad_category'].nunique()}")
print(f"  Total duration        : {master['duration_sec'].sum()/3600:.1f} h")
print(f"  Avg clip duration     : {master['duration_sec'].mean():.1f} s")
if not stor_csv.empty:
    total_gb = stor_csv["size_bytes"].sum() / (1024**3)
    print(f"  Processed storage     : {total_gb:.2f} GB")
print("=" * 55)

print("\nFine-grained category clip counts (top 20):")
display(fine_csv[["label", "broad_category", "clip_count", "avg_duration",
                   "unique_videos"]].head(20))

# %% [markdown]
# ## Cell 14 — Reusable loader for model training
#
# This cell shows how to load the processed dataset as a PyTorch DataLoader
# for training a custom video classification model.

# %%
# Example: load the train split metadata
train_df = pd.read_csv(cfg.METADATA_DIR / "train_splits.csv")

print(f"Train split: {len(train_df):,} clips")
print(f"Labels: {train_df['label'].nunique()} fine-grained, "
      f"{train_df['broad_category'].nunique()} broad")

# Build label-to-index mappings
fine_labels   = sorted(train_df["label"].unique())
broad_labels  = sorted(train_df["broad_category"].unique())
fine_to_idx   = {l: i for i, l in enumerate(fine_labels)}
broad_to_idx  = {l: i for i, l in enumerate(broad_labels)}

print(f"\nFine label map (first 5):  {list(fine_to_idx.items())[:5]}")
print(f"Broad label map (first 5): {list(broad_to_idx.items())[:5]}")

# Save label maps for reproducibility
import json
maps_path = cfg.METADATA_DIR / "label_maps.json"
json.dump({"fine_to_idx": fine_to_idx, "broad_to_idx": broad_to_idx},
          open(maps_path, "w"), indent=2)
print(f"\nLabel maps saved: {maps_path}")

# %% [markdown]
# ## Cell 15 — Quick sanity check: play a downloaded clip

# %%
from IPython.display import Video as IPVideo, display

# Pick a random valid clip and display it
valid_clips = report[report["status"] == "valid"]["expected_path"].tolist()

if valid_clips:
    import random
    sample = random.choice(valid_clips)
    print(f"Playing: {sample}")
    display(IPVideo(sample, embed=True, width=480))
else:
    print("No valid clips found yet — run the download cell first.")

# %% [markdown]
# ---
# ## Notes on integrating VIDI into VLM Disaster Analyzer
#
# ### Recommended next steps
#
# 1. **Fine-tune CLIP on VIDI frames**
#    - Extract all thumbnails (Cell 10)
#    - Use `datasets/video_datasets/vidi_dataset/thumbnails/` as an image dataset
#    - Follow the existing CLIP fine-tuning workflow from `src/models/clip_model.py`
#
# 2. **Add Video-LLaVA to the backend**
#    ```python
#    # In backend/main.py:
#    from backend.routes.predict_video import router as video_router
#    app.include_router(video_router, tags=["Video VLM Inference"])
#    ```
#
# 3. **Add video upload to the frontend**
#    - Extend the file-input in `frontend/src/App.jsx` to accept `video/*` MIME types
#    - Add a frame preview strip using the canvas API
#    - Route video files to `/predict/video/llava` instead of `/predict/clip`
#
# 4. **Recommended models by use-case**
#
#    | Task | Model | VRAM |
#    |------|-------|------|
#    | Zero-shot scene description | Video-LLaVA 7B | 14 GB |
#    | Incident classification | InternVideo2 1B | 4 GB |
#    | Chat-style Q&A on video | Qwen2-VL 7B | 16 GB |
#    | Frame-level CLIP similarity | X-CLIP | 2 GB |
#
# 5. **Colab GPU availability**
#    - T4 (free): Video-LLaVA with `load_in_4bit=True` (quantised)
#    - A100 (Pro+): Full float16 precision for all models
