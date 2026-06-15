# ISRO Project Summary

**VLM Disaster Analyzer — Internship Submission Summary**

> Prepared by: Ujjesha Nityarouthu  
> Internship Organization: Indian Space Research Organisation (ISRO)  
> Full technical documentation: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

---

## Project Title

**Multi-Model Vision-Language Analysis System for Disaster Event Classification and Historical Retrieval**

---

## Motivation

Rapid satellite and aerial image analysis is critical to effective disaster response. Existing classification pipelines require either expensive GPU infrastructure for large vision-language models or rely on task-specific fine-tuning that generalizes poorly across disaster types and geographies. ISRO's disaster monitoring mission requires a system that is:

- **Fast:** Triage results within seconds of image acquisition
- **Informative:** Beyond a label — structured damage assessments with severity and affected-area estimates
- **Contextual:** Linking observed imagery to documented historical precedents with outcome data
- **Deployable:** Operable from Colab with a persistent public frontend, requiring no dedicated server

This project addresses all four constraints in a single integrated system.

---

## Objectives

1. Build a production-ready three-stage pipeline combining zero-shot CLIP classification, Qwen2-VL structured reporting, and FAISS historical retrieval
2. Evaluate four vision-language models (CLIP, BLIP-2, LLaVA-1.5, Qwen2-VL) on a 75-video disaster benchmark
3. Construct a curated 30-event India-focused historical disaster database with FAISS index for visual similarity retrieval
4. Deploy as a live web application accessible globally without local installation

---

## Technical Contributions

### 1. Three-Stage Unified Pipeline

A novel three-stage inference pipeline that chains a zero-shot classifier (CLIP), an instruction-following VLM (Qwen2-VL), and a FAISS similarity search (historical retrieval) in a single API call:

```
Stage 1: CLIP zero-shot triage (~500 ms, 12 descriptive prompts)
Stage 2: Qwen2-VL 7-field structured report (2–3 s on GPU)
Stage 3: FAISS top-5 historical event retrieval (<100 ms)
```

CLIP's embedding is reused for FAISS retrieval — one forward pass serving two purposes.

### 2. Zero-Shot Disaster Classification

CLIP (`openai/clip-vit-base-patch32`) achieves **85.3% accuracy** across 5 disaster categories with no fine-tuning. Semantic descriptive prompts replace bare class labels — "an image showing wildfire with flames and smoke" versus "Wildfire" — yielding an estimated 15–20 pp improvement through better alignment with CLIP's caption-style pretraining.

### 3. Structured Damage Reporting

Qwen2-VL (2B parameters, 4-bit NF4 quantized) generates seven structured fields per image: disaster type, severity tier, description, visible damage, affected area, environmental impact, and recommendations. A labeled-field prompt template with deterministic line-by-line parsing eliminates hallucination noise in downstream field consumption.

### 4. India-Focused Historical Retrieval

A curated 30-event database covering India's major disasters (Kerala 2018, Amphan 2020, Wayanad 2024, Nepal 2015, etc.) with casualty figures, affected population estimates, and economic damage data. FAISS `IndexFlatIP` (512-dim unit-normalized vectors) enables cosine similarity search in under 100 ms. The top-5 most visually similar historical events are returned with each analysis, surfacing outcome data relevant to the identified disaster type and location.

### 5. Multi-Model Evaluation Framework

A systematic comparative evaluation across CLIP, BLIP-2, LLaVA-1.5, and Qwen2-VL on the 75-video VIDI dataset (300 extracted frames, majority vote per video). The ensemble achieves **87.5% overall accuracy**, a 2.2 pp improvement over the best single model. Per-category confusion analysis identifies earthquake and landslide as the hardest categories — structural damage patterns that require multi-scale context beyond single frames.

### 6. Production Deployment Profile System

A runtime deployment configuration system driven by the `ACTIVE_MODELS` environment variable — no code changes required to switch between production (CLIP + Qwen, 2.5 GB VRAM) and research (all 4 models, 12–16 GB VRAM) modes. Disabled models return HTTP 200 with a `status: "disabled"` field rather than 503 errors, enabling graceful degradation.

---

## System Capabilities

| Capability | Details |
|---|---|
| Disaster type classification | 12 categories, zero-shot |
| Structured damage report | 7 fields: type, severity, description, damage, area, environment, recommendations |
| Confidence scoring | 4-tier system with operational guidance |
| Historical context | Top-5 similar events with casualty and economic data |
| Chat interface | GPT-4o powered with keyword fallback |
| Video analysis | Frame extraction + multi-frame majority vote |
| API | 12 REST endpoints, Swagger UI documentation |

---

## Performance Summary

| Metric | Value |
|---|---|
| VIDI benchmark accuracy (ensemble) | **87.5%** |
| VIDI benchmark accuracy (CLIP alone) | 85.3% |
| CLIP inference time (GPU) | ~500 ms |
| Qwen2-VL inference time (GPU) | ~2–3 s |
| FAISS retrieval time | <100 ms |
| Historical events database | 30 events, 5 categories |
| VIDI evaluation dataset | 75 videos, 300 frames |

---

## Deployment Status

**Live and publicly accessible:**

| Component | URL |
|---|---|
| Frontend | https://vlm-disaster-analyzer.vercel.app |
| Backend (on-demand) | Google Colab T4 + ngrok tunnel |
| Repository | github.com/ujjesha1312/VLM-Disaster-Analyzer |

The frontend is deployed persistently on Vercel (zero-cost). The GPU backend runs on Colab T4 (free tier) and is launched on-demand, requiring no dedicated server subscription. The system is designed to be fully operational within 5 minutes of session startup.

---

## Future Roadmap

1. **Geospatial integration:** Overlay ISRO satellite imagery (Cartosat-3, RESOURCESAT-2) with real-time disaster analysis
2. **Fine-tuning:** Domain-adapted CLIP on India-specific disaster imagery to address earthquake/landslide disambiguation
3. **Temporal analysis:** Multi-frame video inference with change detection across acquisition timestamps
4. **Historical database expansion:** Expand from 30 to 500+ events with structured NDMA, CEEW, and ReliefWeb data ingestion
5. **Edge deployment:** Model distillation for ISRO ground station deployment without cloud connectivity
6. **Persistent backend:** Migrate from Colab+ngrok to a managed GPU instance (RunPod, Lambda Labs) with fixed URL and auto-restart

---

## Document Index

| Document | Contents |
|---|---|
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Complete 14-section technical specification |
| [System_Architecture.md](../architecture/System_Architecture.md) | Three-tier architecture, pipeline flow, API reference |
| [Deployment_Architecture.md](../architecture/Deployment_Architecture.md) | Colab + Vercel setup, environment reference |
| [Dataset_Preparation.md](../methodology/Dataset_Preparation.md) | VIDI dataset, frame extraction, data pipelines |
| [Evaluation_Methodology.md](../methodology/Evaluation_Methodology.md) | Benchmark methodology, majority vote, metrics |
| [Historical_Retrieval.md](../methodology/Historical_Retrieval.md) | FAISS index, 30-event database, search mechanics |
| [Results_Summary.md](Results_Summary.md) | Accuracy tables, latency benchmarks, key findings |

---

*Submitted in partial fulfillment of ISRO internship requirements.*
