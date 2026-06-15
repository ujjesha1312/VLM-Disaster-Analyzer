# Evaluation Methodology

**VLM Disaster Analyzer — VIDI 75-Video Benchmark and Multi-Model Evaluation**

> Source of truth: [TECHNICAL_REPORT.md](../reports/TECHNICAL_REPORT.md) §6, §7  
> Related: [Dataset_Preparation.md](Dataset_Preparation.md) · [Results_Summary.md](../reports/Results_Summary.md)

---

## Evaluation Goals

The evaluation answers three questions:

1. **How accurate is each model** at disaster type classification across 5 categories?
2. **How do models differ** in speed, verbosity, and structural consistency?
3. **Does multi-model fusion** (majority vote) outperform any individual model?

Evaluation is entirely held-out: models see only the extracted frames — no ground-truth labels are used at inference time.

---

## Dataset

**75 videos**, 5 disaster categories × 15 videos:

| Category | Ground Truth Label | Videos |
|---|---|---|
| Flood | `flood` | 15 |
| Wildfire | `wildfire` | 15 |
| Earthquake | `earthquake` | 15 |
| Landslide | `landslide` | 15 |
| Cyclone | `cyclone` | 15 |

Frame extraction: 4 frames per video at positions 20%, 40%, 60%, 80% of duration.  
Total frames evaluated: **300 PNG images**.

See [Dataset_Preparation.md](Dataset_Preparation.md) for packaging and upload details.

---

## Evaluation Pipeline

### Colab Notebook

The evaluation runs in `notebooks/VIDI_75_Research_Pipeline.ipynb`.

**Workflow:**

```
colab_frames.zip (31.74 MB)
        │
        ▼
!unzip → /content/frames/
        │
        ▼
Load frame_manifest.csv
(video_id, category, frame_filename)
        │
        ├── For each model in [CLIP, BLIP-2, LLaVA, Qwen2-VL]:
        │       For each video_id:
        │           For each of 4 frames:
        │               inference → predicted_category
        │           majority_vote(4 predictions) → video_prediction
        │       accuracy = correct / 75
        │       Write per-model CSV
        │
        ▼
Consolidate all CSVs → results.xlsx (5 sheets)
```

### Per-Frame Inference

Each frame is passed independently to the model API endpoint:

```python
response = requests.post(
    f"http://localhost:8000/predict/{model_name}",
    files={"file": open(frame_path, "rb")}
)
pred_category = response.json()["disaster_type"].lower()
```

Models available: `clip`, `blip2`, `llava`, `qwen`

### Majority Vote Aggregation

Four frame-level predictions are aggregated per video using plurality vote:

```python
from collections import Counter

def aggregate_predictions(frame_preds: list[str]) -> str:
    return Counter(frame_preds).most_common(1)[0][0]
```

**Tie-breaking:** `Counter.most_common()` preserves insertion order on ties — the first predicted category wins. In practice, ties are rare with 4 frames over 5 categories.

**Fault tolerance:** If a frame request fails (HTTP error, timeout), that frame's prediction is omitted. A video with only 2 valid frames still aggregates via majority vote. A video with 0 valid frames is logged and excluded from accuracy calculation.

---

## Per-Model Output CSVs

Each model writes `results_{model}.csv`:

| Column | Type | Description |
|---|---|---|
| `video_id` | str | Unique video identifier |
| `true_category` | str | Ground-truth disaster category |
| `predicted_category` | str | Majority-vote prediction |
| `correct` | bool | `predicted == true` |
| `frame_1_pred` | str | Frame-level prediction |
| `frame_2_pred` | str | Frame-level prediction |
| `frame_3_pred` | str | Frame-level prediction |
| `frame_4_pred` | str | Frame-level prediction |
| `inference_time_s` | float | Mean per-frame inference time (s) |

---

## Excel Consolidation

`results.xlsx` is generated with one sheet per model plus a summary sheet:

| Sheet | Content |
|---|---|
| `CLIP` | 75 rows, per-video predictions |
| `BLIP2` | 75 rows |
| `LLaVA` | 75 rows |
| `Qwen2VL` | 75 rows |
| `Summary` | Accuracy table + confusion matrices |

Summary sheet columns:
- Model, Accuracy %, Flood Acc, Wildfire Acc, Earthquake Acc, Landslide Acc, Cyclone Acc
- Mean inference time (s)
- Majority vote row (across all 4 models)

---

## Majority Vote Ensemble

The ensemble prediction for each video is the plurality vote across all four individual-model video-level predictions:

```python
ensemble_pred = Counter([
    clip_pred, blip2_pred, llava_pred, qwen_pred
]).most_common(1)[0][0]
```

The ensemble is computed post-hoc — no additional inference is needed. It is reported as a fifth row in the summary sheet.

---

## Metrics

| Metric | Definition |
|---|---|
| Overall Accuracy | Correct video predictions / 75 |
| Per-category Accuracy | Correct in category / 15 |
| Confusion Matrix | 5 × 5 predicted × true |
| Mean Inference Time | Mean per-frame API call time (s) |

Precision, recall, and F1 per category are computed from the confusion matrix and included in `Results_Summary.md`.

---

## Research Mode Requirement

The VIDI evaluation requires **Research mode** (`start_research.py`) since it loads all four local models simultaneously. Running in Production mode (`start_backend.py`) would leave BLIP-2 and LLaVA disabled, returning HTTP 200 `status: "disabled"` — these would be counted as incorrect predictions if not filtered.

The evaluation notebook checks model status before running:

```python
status = requests.get("http://localhost:8000/models").json()
active = status["active_models"]
if "blip2" not in active or "llava" not in active:
    raise RuntimeError("Start server in Research mode: python start_research.py")
```

---

## Running the Evaluation

### Step 1 — Start Research backend

```python
# In Colab cell:
import subprocess, threading
from pyngrok import ngrok

def run():
    subprocess.run(["python", "start_research.py"])

threading.Thread(target=run, daemon=True).start()
public_url = ngrok.connect(8000)
```

### Step 2 — Upload frames

```python
from google.colab import files
files.upload()  # Upload colab_frames.zip
!unzip colab_frames.zip -d /content/frames/
```

### Step 3 — Run evaluation cells

Execute all cells in `notebooks/VIDI_75_Research_Pipeline.ipynb`.  
Expected runtime: 45–90 min on T4 (Qwen2-VL is the bottleneck at ~3 s/frame × 300 frames).

### Step 4 — Download results

```python
files.download("results.xlsx")
```

---

## Reproducibility

All randomness in the pipeline is avoided:
- Frame extraction positions are deterministic (percentile-based)
- CLIP inference is deterministic (no sampling)
- Qwen2-VL uses `temperature=0` equivalent via greedy decoding (`do_sample=False`)
- BLIP-2 and LLaVA use greedy decoding

Re-running the evaluation notebook on the same `colab_frames.zip` will produce identical results given identical model weights.

---

*For benchmark results see [Results_Summary.md](../reports/Results_Summary.md)*  
*For dataset preparation see [Dataset_Preparation.md](Dataset_Preparation.md)*
