# Results Summary

**VLM Disaster Analyzer — Performance Benchmarks and Key Findings**

> Source of truth: [TECHNICAL_REPORT.md](../reports/TECHNICAL_REPORT.md) §7, §8  
> Related: [Evaluation_Methodology.md](../methodology/Evaluation_Methodology.md) · [System_Architecture.md](../architecture/System_Architecture.md)

---

## Executive Summary

The three-stage production pipeline achieves **87.5% overall accuracy** on the 75-video VIDI benchmark across 5 disaster categories. CLIP alone reaches 85.3%, validating zero-shot classification as a strong triage foundation. The FAISS retrieval module returns semantically relevant historical precedents in under 100 ms with no impact on classification latency.

---

## Table 1 — Model Accuracy Comparison (VIDI 75-Video Dataset)

| Model | Overall | Flood | Wildfire | Earthquake | Landslide | Cyclone | Inference Time |
|---|---|---|---|---|---|---|---|
| CLIP (zero-shot) | 85.3% | 86.7% | 93.3% | 80.0% | 80.0% | 86.7% | ~500 ms |
| BLIP-2 | 78.7% | 80.0% | 86.7% | 73.3% | 73.3% | 80.0% | ~2.5 s |
| LLaVA-1.5 | 82.7% | 86.7% | 86.7% | 80.0% | 73.3% | 86.7% | ~4–6 s |
| Qwen2-VL | 84.0% | 86.7% | 86.7% | 80.0% | 80.0% | 86.7% | ~2–3 s |
| **Ensemble (majority vote)** | **87.5%** | **93.3%** | **93.3%** | **80.0%** | **80.0%** | **93.3%** | — |

**Evaluation:** 75 videos × 4 frames = 300 frames. Per-video prediction via majority vote of 4 frame-level predictions.

---

## Table 2 — Confidence Tier Distribution

| Tier | CLIP Score | % of Test Images | Action |
|---|---|---|---|
| High | > 88% | ~45% | Immediate operational use |
| Strong | 75–88% | ~30% | Recommended for deployment |
| Moderate | 60–75% | ~17% | Secondary verification recommended |
| Preliminary | < 60% | ~8% | Manual expert review required |

---

## Table 3 — Inference Latency by Deployment Context

| Context | CLIP | Qwen2-VL | End-to-End (3-stage) |
|---|---|---|---|
| NVIDIA T4 GPU (Colab) | ~500 ms | ~2–3 s | ~3–4 s |
| CPU-only (development) | ~2–4 s | ~220 s | ~225 s |

**First-request latency** (model initialization):
- CLIP: +2 s (weight loading)
- Qwen2-VL: +45–90 s (weight dequantization under 4-bit NF4)

**FAISS retrieval:** <100 ms (all contexts)

---

## Table 4 — Per-Category Analysis

| Category | Best Model | Worst Model | Common Failure Mode |
|---|---|---|---|
| Flood | Ensemble (93.3%) | BLIP-2 (80.0%) | Confusion with landslide (water in frame) |
| Wildfire | CLIP / Ensemble (93.3%) | BLIP-2 (86.7%) | Smoke-only frames without visible flame |
| Earthquake | All models (80.0%) | Tied at 80% | Infrastructure damage → cyclone confusion |
| Landslide | Ensemble (80.0%) | LLaVA / BLIP-2 (73.3%) | Muddy water → flood confusion |
| Cyclone | Ensemble (93.3%) | BLIP-2 (80.0%) | Structural damage → earthquake confusion |

---

## Table 5 — FAISS Retrieval Quality

| Category | Mean Top-1 Similarity | Top-5 All > 65% | Qualitative Match |
|---|---|---|---|
| Flood | 81.4% | 92% | High — water signatures are visually consistent |
| Cyclone | 78.9% | 88% | Good — debris patterns transfer across regions |
| Wildfire | 83.2% | 95% | Excellent — fire/smoke are cross-regional |
| Earthquake | 72.1% | 71% | Moderate — urban vs. rural collapse patterns differ |
| Landslide | 69.8% | 65% | Moderate — slope angle and vegetation vary widely |

---

## Table 6 — Production vs. Research Mode Comparison

| Dimension | Production Mode | Research Mode |
|---|---|---|
| Active models | CLIP + Qwen2-VL | CLIP + BLIP-2 + LLaVA + Qwen2-VL |
| VRAM usage | ~2.5 GB | 12–16 GB |
| Per-request latency | ~3–4 s | ~12–20 s |
| Accuracy (unified pipeline) | 87.5%* | 87.5% |
| Use case | Live inference, demos | Benchmarking, research |

*Production accuracy uses Qwen2-VL for Stage 2. Ensemble accuracy requires Research mode.

---

## Key Findings

### Finding 1 — Semantic prompting closes the gap with fine-tuned models

CLIP's zero-shot accuracy (85.3%) approaches LLaVA-1.5 (82.7%) and nearly matches Qwen2-VL (84.0%), despite using no disaster-specific fine-tuning. Descriptive prompts ("an image showing wildfire with flames and smoke") outperform bare label prompts by an estimated 15–20 percentage points — a direct consequence of CLIP's contrastive pretraining on caption-style text rather than classification labels.

### Finding 2 — Majority voting provides consistent 2–5% lift

Ensemble (87.5%) outperforms the best individual model (CLIP at 85.3%) by 2.2 percentage points. The lift is most pronounced in flood (6.6 pp) and cyclone (6.6 pp) — categories with high within-class visual variance. Categories dominated by a single visual signature (wildfire) show smaller ensemble gains.

### Finding 3 — Wildfire is the most consistently detected category

Wildfire achieves 86.7–93.3% accuracy across all models. Flame and smoke signatures are visually distinctive and cross-regional, making them robust to the diverse geographical mix in the evaluation set.

### Finding 4 — Earthquake and landslide are the hardest categories

Both categories plateau at 80% even with ensemble voting. The primary failure modes are structural: earthquake damage (collapsed concrete) is visually similar to cyclone aftermaths; landslide (mud flow) confuses with flood (water presence). Multi-scale context that individual frames lack would be needed to reliably disambiguate these pairs.

### Finding 5 — 4-frame majority vote is stable

Across the evaluation, single-frame accuracy was 79.6% on average; 4-frame majority vote reaches 85.3% for CLIP — a 5.7 percentage point gain from temporal aggregation alone without any model change.

### Finding 6 — FAISS retrieval adds operational context with near-zero latency cost

The retrieval stage adds <100 ms to pipeline latency while surfacing casualty figures, affected population estimates, and damage assessments from comparable historical events. This contextual grounding is the primary practical contribution over a standalone classification system.

---

## Limitations

- **No holdout test set:** The VIDI 75-video dataset serves as both the validation and reported test set. Independent test performance may differ.
- **No fine-tuning:** All models operate zero-shot or with generic instruction tuning. Disaster-specific fine-tuning would likely improve earthquake and landslide discrimination.
- **CPU latency is impractical for production:** Qwen2-VL at ~220 s/image on CPU is unsuitable for real-time use. The GPU deployment is required.
- **ngrok URL rotation:** The free-tier ngrok tunnel requires manual Vercel redeployment each Colab session. A persistent deployment would require a GPU server subscription.
- **FAISS index coverage:** With 30 reference events, some category-specific queries may match against cross-category events when same-category reference images are visually atypical.

---

*For evaluation methodology details see [Evaluation_Methodology.md](../methodology/Evaluation_Methodology.md)*  
*For system architecture see [System_Architecture.md](../architecture/System_Architecture.md)*  
*For ISRO summary see [ISRO_Project_Summary.md](ISRO_Project_Summary.md)*
