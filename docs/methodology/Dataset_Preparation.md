# Dataset Preparation

**VLM Disaster Analyzer — Data Pipelines and Dataset Documentation**

> Source of truth: [TECHNICAL_REPORT.md](../reports/TECHNICAL_REPORT.md) §5, Appendix A  
> Related: [Evaluation_Methodology.md](Evaluation_Methodology.md) · [Historical_Retrieval.md](Historical_Retrieval.md)

---

## Datasets Overview

The system uses two distinct datasets for different purposes:

| Dataset | Purpose | Location | Size |
|---|---|---|---|
| VIDI 75-video dataset | Multi-model evaluation benchmark | `datasets/video_dataset/` | ~800 MB |
| Historical events database | FAISS retrieval reference | `datasets/historical/` | ~50 MB |
| Comprehensive disaster images | CLIP training/evaluation | `datasets/comprehensive_disaster_dataset/` | ~800 MB |

---

## Dataset 1 — VIDI 75-Video Evaluation Dataset

### Overview

The Video Intelligence for Disaster Identification (VIDI) dataset comprises 75 short disaster videos sourced from public repositories. It is used exclusively for multi-model comparative evaluation — not for training.

**Coverage:** 5 disaster categories × 15 videos per category  
**Categories:** Flood, Wildfire, Earthquake, Landslide, Cyclone  
**Frame extraction:** 4 frames per video at evenly spaced temporal positions  
**Total frames:** 300 (packaged as `colab_frames.zip`, 31.74 MB)

### Directory Structure

```
datasets/video_dataset/
├── extracted_frames/           ← 300 PNG frames (organized by video)
│   ├── <video_id>_frame_01.png
│   ├── <video_id>_frame_02.png
│   ├── <video_id>_frame_03.png
│   └── <video_id>_frame_04.png
├── frame_manifest.csv          ← Frame index with video_id, category, frame_filename
└── colab_frames.zip            ← 31.74 MB archive for Colab upload
```

### Frame Extraction

Run `scripts/video_pipeline/extract_frames.py`:

```bash
python scripts/video_pipeline/extract_frames.py \
    --input datasets/video_dataset/raw_videos/ \
    --output datasets/video_dataset/extracted_frames/ \
    --frames-per-video 4
```

The script extracts frames at positions 20%, 40%, 60%, 80% of each video's duration, avoiding scene-change artifacts at the very start and end.

`frame_manifest.csv` columns:

| Column | Description |
|---|---|
| `video_id` | Unique video identifier |
| `category` | Ground-truth disaster category |
| `frame_filename` | PNG filename in `extracted_frames/` |
| `frame_index` | Frame number (1–4) within the video |
| `timestamp_s` | Extraction timestamp in seconds |

### Packaging for Colab

```bash
python scripts/video_pipeline/package_for_colab.py
# Output: datasets/video_dataset/colab_frames.zip (31.74 MB, 301 entries)
```

The archive contains: 300 PNG frames + `frame_manifest.csv`.

Upload this file to Google Drive and mount in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
!unzip /content/drive/MyDrive/colab_frames.zip -d /content/frames/
```

---

## Dataset 2 — Historical Events Database

**30 curated events** used for FAISS retrieval. See [Historical_Retrieval.md](Historical_Retrieval.md) for complete documentation including the full event list, database schema, and index construction steps.

**Quick reference:**

| Category | Count | Example Events |
|---|---|---|
| Flood | 7 | Kerala 2018, Assam 2020/2022, Bihar 2017 |
| Cyclone | 7 | Fani 2019, Amphan 2020, Haiyan 2013 |
| Wildfire | 6 | Uttarakhand 2021/2024, Camp Fire 2018 |
| Earthquake | 5 | Nepal 2015, Gujarat 2001, Turkey 2023 |
| Landslide | 5 | Wayanad 2024, Kedarnath 2013 |

Reference images are fetched from Wikipedia's `pageimages` API and stored at `datasets/historical/images/<category>/`.

---

## Dataset 3 — Comprehensive Disaster Image Dataset

Static image dataset used for CLIP classification evaluation and pipeline testing.

```
datasets/comprehensive_disaster_dataset/
└── Damaged_Infrastructure/
    ├── Earthquake/       ← Satellite/aerial imagery (small, low-res frames)
    └── Infrastructure/   ← Larger resolution infrastructure damage images
```

**Source:** Extracted frames from publicly available disaster imagery.

---

## Disaster Categories Supported

The system classifies 12 categories via CLIP zero-shot prompting:

| Category | CLIP Prompt (descriptive) | Knowledge Base | FAISS Events |
|---|---|---|---|
| Flood | "an image showing flood or water disaster with submerged areas" | ✓ | 7 |
| Water Disaster | "an image showing water disaster" | ✓ (uses Flood KB) | — |
| Wildfire | "an image showing wildfire with flames and smoke" | ✓ | 6 |
| Urban Fire | "an image showing urban fire with burning buildings" | ✓ (uses Fire KB) | — |
| Earthquake | "an image showing earthquake damage with collapsed buildings" | ✓ | 5 |
| Infrastructure Damage | "an image showing severe infrastructure damage" | — | — |
| Human Damage | "an image showing human casualties and rescue operations" | — | — |
| Landslide | "an image showing landslide on roads or mountains" | ✓ | 5 |
| Cyclone | "an image showing cyclone destruction with damaged structures" | ✓ | 7 |
| Drought | "an image showing drought with dry cracked land" | — | — |
| Forest Scene | "a forest or jungle scene without disaster" | — (baseline) | — |
| Urban/Buildings | "urban or city buildings scene without disaster" | — (baseline) | — |

The two baseline categories (Forest, Urban) serve as negative controls in the zero-shot classification prompt set, reducing false positives on non-disaster imagery.

---

## Video Download Pipeline

To download new VIDI videos:

```bash
python scripts/video_pipeline/download_videos.py \
    --output datasets/video_dataset/raw_videos/ \
    --category flood cyclone earthquake \
    --count 15
```

Replacement downloads for unavailable videos:
```bash
python scripts/video_pipeline/download_replacements.py
```

Dataset statistics:
```bash
python scripts/video_pipeline/generate_manifest.py
# Generates/updates frame_manifest.csv
```

---

## Data Directory Reference

```
datasets/
├── comprehensive_disaster_dataset/     ← Static image eval dataset
│   └── Damaged_Infrastructure/
│       ├── Earthquake/                 ← 36 PNG frames (low-res)
│       └── Infrastructure/             ← 231 PNG frames (high-res)
├── historical/                         ← FAISS retrieval module data
│   ├── historical_events.json          ← 30-event curated database
│   ├── images/                         ← Wikipedia reference thumbnails
│   │   ├── flood/
│   │   ├── cyclone/
│   │   ├── wildfire/
│   │   ├── earthquake/
│   │   └── landslide/
│   └── index/                          ← Built FAISS index
│       ├── disaster.index
│       └── metadata.json
└── video_dataset/                      ← VIDI evaluation dataset
    ├── extracted_frames/               ← 300 PNG frames
    ├── frame_manifest.csv              ← Frame index
    └── colab_frames.zip                ← 31.74 MB Colab archive
```

---

*For evaluation methodology see [Evaluation_Methodology.md](Evaluation_Methodology.md)*  
*For historical retrieval technical details see [Historical_Retrieval.md](Historical_Retrieval.md)*
