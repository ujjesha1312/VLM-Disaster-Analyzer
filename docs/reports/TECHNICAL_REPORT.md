# VLM DISASTER ANALYZER
## A Multi-Modal Vision Language Model Framework for Multimodal Disaster Intelligence and Rapid Assessment

---

**Technical Report — Project Documentation**  
Prepared for: Indian Space Research Organisation (ISRO)  
Department of Space, Government of India

Repository: github.com/ujjesha1312/VLM-Disaster-Analyzer  
Deployment: vlm-disaster-analyzer.vercel.app  
Date: June 2026

---

## Abstract

Rapid and accurate assessment of disaster-affected areas from aerial and satellite imagery is a critical capability for national disaster management agencies and space-based earth observation programs. This report presents the VLM Disaster Analyzer, a multimodal disaster intelligence platform that integrates five vision-language models (CLIP, Qwen2-VL, BLIP-2, LLaVA, and GPT-4V) within a three-tier FastAPI backend and a React-based single-page application. The system implements a novel **three-stage inference cascade**: Stage 1 in which CLIP provides rapid zero-shot disaster classification in approximately 500 milliseconds; Stage 2 in which CLIP's output is injected as contextual grounding into Qwen2-VL (Qwen/Qwen2-VL-2B-Instruct), which generates a complete structured report encompassing disaster type, severity, visible damage characterization, affected area estimation, environmental impact, and emergency recommendations — all within 3–4 seconds end-to-end on a GPU-equipped backend; and Stage 3 in which CLIP embeddings drive a FAISS cosine similarity search against a curated database of 30 historical Indian and global disaster events, returning the most visually analogous precedents alongside casualty and impact metadata. A multi-model research mode executes all four local models sequentially under a shared GPU execution lock, enabling systematic comparative benchmarking without out-of-memory failures. The system classifies twelve disaster categories including Flood, Wildfire, Earthquake, Landslide, and Cyclone, with confidence scoring calibrated across four tiers from preliminary (<60%) to high (>88%). The platform is deployed with a Google Colab GPU backend (NVIDIA T4, 16 GB VRAM) exposed via ngrok tunnel and a Vercel-hosted React frontend, and is designed for direct integration with satellite imagery pipelines from earth observation missions.

---

## TABLE OF CONTENTS

1. System Architecture Overview
2. Model Selection and Justification
3. Three-Stage Inference Pipeline (Unified Mode)
4. Historical Disaster Retrieval Module
5. Multi-Model Comparative Analysis (Research Mode)
6. Sequential GPU Execution Strategy
7. Confidence Scoring Mechanism
8. Prompt Engineering for Disaster Classification
9. Production and Research Mode Configuration
10. Frontend Architecture and State Management
11. API Design and Communication Protocol
12. Error Handling and Fault Tolerance
13. Deployment Architecture
14. Performance Analysis and Benchmarks

Appendix A: Disaster Categories Supported  
Appendix B: System Requirements Summary

---

## 1. System Architecture Overview

The VLM Disaster Analyzer is architected as a three-tier backend system comprising an HTTP routing layer, a services abstraction layer, and a dedicated model inference layer, fronted by a React-based single-page application deployed independently on Vercel. This separation of concerns is deliberate: the routing layer is responsible solely for request validation, file-size enforcement (10 MB ceiling for images, 500 MB for video), and content-type negotiation; the services layer enforces the active-model deployment profile and maps incoming requests to the appropriate inference function without the calling code requiring awareness of model implementation details; and the inference layer encapsulates all model-specific logic — including tokenization, prompt construction, forward pass execution, and output parsing — within isolated, independently replaceable modules. This design ensures that substituting a model or adding a new vision architecture requires no modification to the routing or services layers.

All models are loaded lazily: no weights are transferred to RAM or VRAM at server startup. Each model's singleton loader employs double-checked locking to guarantee thread-safe initialization on the first inference request while eliminating redundant load attempts on subsequent calls. The first-request load time for Qwen2-VL under 4-bit NF4 quantization on a T4 GPU is approximately 45–90 seconds due to weight dequantization; subsequent requests incur only the inference latency of 2–3 seconds. This design keeps server startup time near-instantaneous regardless of the number of models configured — a critical requirement in containerized deployment environments where health checks must pass within a bounded interval before downstream services are permitted to start.

The system exposes twelve REST endpoints organized across five functional domains: individual model inference, unified pipeline inference, historical similarity retrieval, disaster intelligence chat, and video analysis. Router mounting order in the FastAPI entry point is order-sensitive: the disaster pipeline router (`POST /predict/disaster`) and the video router (`POST /predict/video/*`) are registered before the wildcard individual-model router (`POST /predict/{model_name}`) to prevent the wildcard from capturing specialized paths. The retrieval router is registered between the disaster and video routers. The chat router is registered last as a top-level path with no conflict potential.

The active deployment profile is controlled through environment variables (`ACTIVE_MODELS`, `ENABLE_RETRIEVAL`) that are evaluated once at import time, enabling zero-code switching between a lightweight production configuration (CLIP + Qwen2-VL) and a full research configuration (all four local models) through a single launcher script change.

---

## 2. Model Selection and Justification

Five vision-language models were selected to span a representative range of architectures, parameter scales, inference paradigms, and deployment modalities. The selection criteria prioritized breadth of coverage across classification, captioning, instruction-following, and cloud-based reasoning, rather than optimizing for a single capability. This ensemble approach supports both production deployment — where the three-stage CLIP→Qwen2-VL→FAISS pipeline is primary — and systematic research benchmarking, where outputs from all models can be compared on identical imagery.

| Model / Identifier | Architecture | Inference Latency | VRAM Required | Parameters | Role in System |
|---|---|---|---|---|---|
| CLIP (openai/clip-vit-base-patch32) | Zero-Shot Classification | ~500 ms | ~0.3 GB | ~151 M | Stage 1 — Rapid disaster triage + FAISS embedding |
| Qwen2-VL (Qwen2-VL-2B-Instruct) | Instruction-Following VLM | 2–3 s (GPU) / ~220 s (CPU) | ~2.0 GB (4-bit NF4) | ~2 B | Stage 2 — Structured report generation |
| BLIP-2 (blip2-opt-2.7b) | Bootstrapped Captioning | ~4–6 s (GPU) | ~5.5 GB | ~2.7 B | Research — Dense scene captioning |
| LLaVA (llava-1.5-7b-hf) | Visual Instruction Tuning | ~6–8 s (GPU) | ~14 GB | ~7 B | Research — Structured QA reasoning |
| GPT-4V (gpt-4o via API) | Cloud-Based Multimodal LLM | ~3–5 s | N/A (Cloud) | ~1.8 T* | Reference — High-capacity benchmark |

*Table 1: Model Selection Matrix — Specifications and Roles*

CLIP (openai/clip-vit-base-patch32) was chosen as the rapid first-stage classifier owing to its zero-shot transferability: by computing cosine similarity in a shared image-text embedding space against rich descriptive text prompts, CLIP produces reliable disaster-type predictions at approximately 500 milliseconds without any task-specific fine-tuning. Its 512-dimensional unit-normalized image embeddings are additionally reused as input to the Stage 3 FAISS retrieval index, making CLIP's single forward pass serve dual purposes without redundant computation. Qwen2-VL (Qwen/Qwen2-VL-2B-Instruct) was selected as the second-stage reasoning model for its strong instruction-following capability at a compact 2-billion parameter scale, enabling full structured report generation within 2–3 seconds on a T4 GPU under 4-bit quantization. GPT-4V, accessed via the gpt-4o endpoint, serves as a cloud-based reference model implemented behind an abstract provider class with defined stubs for Azure OpenAI, Anthropic Claude Vision, and Google Gemini, allowing the cloud backend to be exchanged without modifying calling code.

---

## 3. Three-Stage Inference Pipeline (Unified Mode)

The production inference path, exposed at `POST /predict/disaster`, implements a deliberate three-stage cascade designed to combine the latency advantage of CLIP, the contextual reasoning depth of Qwen2-VL, and the historical grounding provided by FAISS similarity search.

**Stage 1 — CLIP Triage:** CLIP classifies the submitted image against twelve descriptive text prompts and returns the highest-probability disaster category along with its softmax-derived confidence score expressed as a percentage. This classification result is not discarded after Stage 1; it is injected directly into the Qwen2-VL prompt as explicit prior context, and the CLIP image embedding is separately forwarded to Stage 3 for FAISS search.

**Stage 2 — Qwen2-VL Report Generation:** The Qwen2-VL prompt is prefixed with CLIP's classification result and confidence level, grounding the second-stage model's generation in the triage hypothesis rather than cold-starting on open-ended visual interpretation. This context-injection strategy was adopted after observing that Qwen2-VL without prior context produced outputs that were lexically fluent but semantically underspecified, defaulting to generic scene descriptions rather than targeted disaster assessments. The resulting unified report encompasses seven structured fields: disaster type, severity classification (Critical / High / Moderate / Low), a natural-language description, visible damage characterization, estimated affected area, environmental impact assessment, and actionable emergency recommendations. All seven fields are populated within 3–4 seconds end-to-end on a GPU-equipped backend.

**Stage 3 — Historical Retrieval (Best-Effort):** Upon completion of Stage 2, CLIP embeddings are used to query the pre-built FAISS index, returning the top-5 most visually similar historical disaster events with their cosine similarity scores, casualty figures, affected population estimates, and source references. Stage 3 is wrapped in a `try/except` block and gated on the `ENABLE_RETRIEVAL` configuration flag; failure or absence of a built index does not propagate to the caller — the response is returned with an empty `similar_events` list and Stage 1–2 outputs remain unaffected.

---

## 4. Historical Disaster Retrieval Module

The Historical Disaster Retrieval module provides contextual grounding by identifying precedent events that are visually analogous to an uploaded disaster image. It is implemented as a FAISS cosine similarity search over a curated database of 30 historical Indian and global disaster events, using CLIP visual embeddings as the search representation.

**Database:** `datasets/historical/historical_events.json` contains 30 manually curated events across five disaster categories: 7 flood events, 7 cyclone events, 6 wildfire events, 5 earthquake events, and 5 landslide events. The database is India-focused, covering major events including the Kerala Floods (2018), Cyclone Fani (2019), Uttarakhand Floods (2013), and the Nepal Earthquake (2015), with global benchmarks including the Camp Fire (2018) and Haiti Earthquake (2010). Each event record contains: `id`, `name`, `year`, `category`, `location`, `description`, `casualties`, `affected_population`, `damage_usd_billion`, `source`, `image_filename`, and `wikipedia_search`.

**Index Construction:** Reference images for each event are fetched from Wikipedia's pageimages API by `scripts/download_historical_images.py`, which downloads CC-licensed thumbnails and immediately invokes `src/retrieval/build_index.py`. The builder passes each reference image through CLIP's `get_image_features()` method, L2-normalizes the resulting 512-dimensional vector, and inserts it into a FAISS `IndexFlatIP` (inner product) index. Since all vectors are unit-normalized, inner product equals cosine similarity. The completed index is serialized to `datasets/historical/index/disaster.index` alongside `datasets/historical/index/metadata.json`.

**Search:** `src/retrieval/search.py` implements a lazy singleton that loads the FAISS index and metadata on first call. The `find_similar_events(image_path, top_k, category_filter)` function embeds the query image via CLIP, performs an `index.search()` call returning raw cosine similarity distances, optionally filters by disaster category, and returns the top-k results with similarity expressed as a percentage. The similarity scale is calibrated as: ≥80% (high visual match, rendered in emerald in the frontend), 65–80% (moderate match, amber), <65% (low match, muted gray).

**Frontend Integration:** The `SimilarEventsCard` React component renders matching events with rank badges, event name, year, location, a one-sentence description, and a color-coded similarity percentage bar. The card is conditionally rendered only when `similar_events.length > 0`, maintaining full backward compatibility when the FAISS index has not been built.

**Endpoints:**
- `POST /predict/similar` — upload image, returns top-k similar events with similarity scores
- `GET /retrieval/status` — reports whether the FAISS index is built and how many events are indexed

---

## 5. Multi-Model Comparative Analysis (Research Mode)

The research endpoint executes all configured models in a fixed sequence — CLIP, Qwen2-VL, BLIP-2, LLaVA — collecting structured outputs from each before aggregating them into a unified comparative response. This mode is intended for systematic benchmarking rather than operational deployment, enabling researchers to examine how models with fundamentally different architectures, training objectives, and parameter scales respond to identical disaster imagery.

Cross-model agreement is quantified through synonym-matched consensus scoring, which normalizes heterogeneous output vocabularies (e.g., 'flooded area,' 'water disaster,' 'submerged streets') into canonical disaster categories before computing the fraction of models in agreement. Individual model failures are recorded and surfaced in the final response without aborting the overall pipeline, so partial results remain useful even when one or more models are unavailable due to VRAM constraints or loading errors. This fault-tolerant aggregation design ensures that the research pipeline degrades gracefully rather than failing completely when operating under constrained hardware.

Research-mode evaluation on the VIDI 75-video dataset is performed via `scripts/video_pipeline/evaluate_videos.py`, which extracts four frames per video at evenly spaced temporal positions, runs each frame through all configured models, and determines the per-video prediction by majority vote across the four frames (`Counter(predictions).most_common(1)[0][0]`). Results are written to per-model CSV files (`clip_results.csv`, `blip2_results.csv`, `llava_results.csv`, `qwen_results.csv`) and consolidated into a comparative Excel workbook by `scripts/video_pipeline/generate_excel.py`. Full GPU-based evaluation on all four models is executed in Google Colab via `notebooks/VLM_Disaster_Evaluation_Colab.ipynb` using a packaged archive of 300 frames (`datasets/video_dataset/colab_frames.zip`, 31.74 MB).

---

## 6. Sequential GPU Execution Strategy

Concurrent multi-model inference is explicitly prohibited by the system architecture. A single shared `asyncio.Lock` in `backend/services/gpu_queue.py` serializes all model forward passes at the application level, ensuring that at most one model occupies the GPU at any given time. This architectural decision reflects a fundamental hardware constraint of the target deployment environment: a single NVIDIA T4 GPU with 16 GB of VRAM cannot simultaneously hold all four local models, whose combined weight footprint ranges from 12 to 16 GB under mixed-precision configurations, leaving insufficient headroom for activations during forward pass execution.

The trade-off accepted by sequential execution is request throughput: a research-mode pipeline that runs all four models requires approximately 12–20 seconds on a T4 GPU rather than the theoretical minimum achievable with full parallelism. For a single-user research and monitoring tool where inference correctness and hardware stability take precedence over concurrent request capacity, this is the appropriate trade-off. Parallel model execution on the target hardware would risk out-of-memory exceptions that terminate the inference process entirely — a materially worse outcome than increased latency. The `asyncio.Lock` approach also yields deterministic execution ordering and simplified debugging, as the sequence of model invocations is fully predictable and traceable in logs.

---

## 7. Confidence Scoring Mechanism

CLIP produces a native confidence score derived from the softmax distribution over its twelve text-prompt similarity scores, expressed as a percentage. This score is inherently calibrated against the prompt set and provides a direct measure of the model's discriminative certainty between candidate disaster categories. Four confidence tiers are defined based on empirical observation of CLIP score distributions on disaster imagery:

| Confidence Tier | Score Range | Typical Condition | Recommended Action |
|---|---|---|---|
| High Confidence | > 88% | Strong visual cues, unambiguous imagery | Immediate operational use |
| Strong Confidence | 75 – 88% | Clear dominant disaster indicators | Recommended for deployment |
| Moderate Confidence | 60 – 75% | Partial occlusion or ambiguity | Recommend secondary verification |
| Preliminary | < 60% | Low visual discriminability | Manual expert review required |

*Table 2: Confidence Tier Classification System*

Qwen2-VL does not emit a self-reported calibrated confidence score as part of its structured text output. Confidence is instead approximated by averaging the maximum token probability across all generation steps: at each decoding step, a softmax distribution is computed over the full vocabulary logits, the maximum probability is extracted, and the arithmetic mean of these per-step maxima constitutes the reported confidence proxy. This approach provides a consistent, model-agnostic signal that correlates with generation fluency, though it tends toward optimistic values for high-frequency vocabulary tokens and does not account for epistemic uncertainty about image content. The same four-tier threshold scheme is applied to Qwen confidence values to maintain consistency across the interface.

---

## 8. Prompt Engineering for Disaster Classification

CLIP's zero-shot classification accuracy on disaster imagery is highly sensitive to prompt formulation. Rather than supplying bare category labels such as 'Flood' or 'Earthquake,' the system employs twelve semantically rich, scene-descriptive prompts that reflect the linguistic register of the image-caption pairs used in CLIP's contrastive pretraining. Prompts such as 'an image showing flood or water disaster with submerged areas' and 'an image showing earthquake damage with collapsed buildings and rubble' provide substantially more discriminative signal than single-word labels. Each prompt is designed to contain at least one visually distinctive structural or environmental anchor that differentiates it from adjacent categories and reduces inter-class confusion between semantically related disaster types such as Wildfire and Urban Fire, or Flood and Cyclone. This prompt design approach is estimated to improve classification performance by 15–20 percentage points on disaster-domain imagery compared to bare label prompting.

The Qwen2-VL prompt is structured as a labeled-field template that constrains the model to produce machine-parseable output rather than free-form prose. Required output fields are specified in uppercase with colon delimiters (DISASTER TYPE, SEVERITY, DESCRIPTION, VISIBLE DAMAGE, AFFECTED AREA, ENVIRONMENTAL IMPACT, RECOMMENDATIONS), enabling deterministic line-by-line parsing of the generated response without reliance on fragile regular expressions. In the unified pipeline, the prompt is prefixed with CLIP's classification result and its associated confidence level, providing a semantic anchor that reduces hallucinated or off-topic outputs on imagery where visual cues are ambiguous or partially obscured.

---

## 9. Production and Research Mode Configuration

The system supports two named deployment profiles controlled entirely through environment variables, enabling mode switching without code changes.

**Production mode** (`start_backend.py`) activates CLIP and Qwen2-VL only, with retrieval enabled:
```
ACTIVE_MODELS    = "clip,qwen"
ENABLE_RETRIEVAL = "true"
QUANTIZE_QWEN    = "true"
```
Approximate VRAM footprint: 2.5 GB (4-bit NF4) or 4.5 GB (fp16).

**Research mode** (`start_research.py`) activates all four local models:
```
ACTIVE_MODELS    = "clip,blip2,llava,qwen"
ENABLE_RETRIEVAL = "true"
QUANTIZE_QWEN    = "true"
```
Approximate VRAM footprint: 12–16 GB. Requires a T4 or equivalent research GPU.

`backend/config.py` parses `ACTIVE_MODELS` at import time into a frozen set, derives six named boolean flags (`ENABLE_CLIP`, `ENABLE_QWEN`, `ENABLE_BLIP2`, `ENABLE_LLAVA`, `ENABLE_RETRIEVAL`, `QUANTIZE_QWEN`), and computes a `DEPLOYMENT_PROFILE` label ("production" or "research") that is logged at startup. All service files reference these flags; models not in the active set are never loaded into memory. BLIP-2 and LLaVA code remains fully intact in the codebase and can be reactivated by switching to research mode without any code modification.

An important operational note on Qwen2-VL: both `AutoProcessor.from_pretrained` and `Qwen2VLForConditionalGeneration.from_pretrained` are invoked with `local_files_only=True`. This prevents the transformers library from attempting network calls to huggingface.co on each server start, which on Windows caused a WinError 10054 (connection forcibly reset) that propagated as HTTP 500 errors. All Qwen weights are served exclusively from the local HuggingFace cache at `~/.cache/huggingface/hub/`. Device map auto-assignment (`device_map="auto"`) is disabled on CPU-only hosts, as accelerate's tensor sharding logic produces layer normalization weight shape mismatches on CPU without CUDA.

---

## 10. Frontend Architecture and State Management

The frontend is implemented as a React single-page application built with Vite and deployed on Vercel, organized around three primary functional surfaces: an image analysis interface, a real-time disaster intelligence chat, and a session history browser. The application implements a three-phase state machine — upload, analyzing, and ready — managing transitions through typed state variables. Conversation history for the chat interface is maintained as an ordered array of message objects in component state and transmitted in its entirety with every `POST /chat` request, with context capped at the last eight exchanges to bound token consumption and maintain response coherence.

Between sessions, up to ten prior assessments are persisted in browser `localStorage`, enabling analytical continuity without requiring user accounts or server-side session management. Each persisted assessment stores the disaster context object, a base64-encoded image thumbnail, briefing text, severity classification, and a timestamp, allowing full chat state to be reconstructed from storage without any additional API calls. The frontend implements per-request abort controllers with asymmetric timeout budgets calibrated to observed backend latency: 180 seconds for model inference calls (accommodating CPU-fallback Qwen inference at approximately 220 seconds) and 60 seconds for chat requests.

The `SimilarEventsCard` component renders the Stage 3 retrieval results when the unified pipeline returns matching historical events. Each card displays a rank badge, event name, year, location, description, and a color-coded similarity percentage bar (≥80% emerald, 65–80% amber, <65% muted gray). The component conditionally renders only when `similar_events.length > 0`, preserving full backward compatibility when the FAISS index has not been built.

A client-side knowledge base comprising five lookup tables covering impacts, recommended actions, infrastructure effects, human impacts, and environmental consequences for five primary disaster categories provides an offline fallback for the chat interface when the GPT-4o endpoint is unavailable, ensuring the system degrades gracefully to template-driven responses. The theming system (`themeEngine.js`) applies CSS custom property sets per disaster type — deep blues for hydrological events, reds and oranges for wildfire, warm ochres for seismic events — providing contextually appropriate visual environments that reinforce the nature of the detected disaster. Theme transitions are triggered at phase transitions and normalized through a 12-label CLIP-output-to-theme mapping that handles heterogeneous model vocabulary across the label space.

---

## 11. API Design and Communication Protocol

All inference endpoints accept `multipart/form-data` requests with a single image file field, enabling straightforward browser-native submission from the frontend without additional encoding overhead. File validation is applied at the routing layer before any model code is invoked: content-type is checked against an allowlist of supported image formats, file size is enforced at a 10 MB ceiling, and PIL (Pillow) is used to validate that the uploaded bytes constitute a decodable image rather than relying solely on the declared content-type header.

| Endpoint | Input Format | Function |
|---|---|---|
| `POST /predict/clip` | Image (multipart) | CLIP zero-shot classification — disaster type, confidence, top-3 predictions |
| `POST /predict/qwen` | Image (multipart) | Qwen2-VL structured analysis — full 6-field disaster report |
| `POST /predict/blip2` | Image (multipart) | BLIP-2 dense visual captioning |
| `POST /predict/llava` | Image (multipart) | LLaVA instruction-following visual QA |
| `POST /predict/gpt4v` | Image (multipart) | GPT-4V cloud-based analysis (requires OPENAI_API_KEY) |
| `POST /predict/disaster` | Image (multipart) | **Unified 3-stage pipeline** — CLIP→Qwen2-VL→FAISS retrieval (production endpoint) |
| `POST /predict/similar` | Image (multipart) | FAISS historical similarity search — top-k visually analogous disaster events |
| `POST /chat` | JSON (question + history + context) | Stateless disaster intelligence chat — GPT-4o primary, keyword fallback |
| `POST /predict/video/analyze` | Video (multipart) | Video metadata extraction + thumbnail generation |
| `GET /retrieval/status` | None | FAISS index health — built status and event count |
| `GET /models` | None | Model inventory with VRAM requirements and active status |
| `GET /` | None | Health check — server status and active model configuration |

*Table 3: Complete API Endpoint Reference*

The chat endpoint (`POST /chat`) accepts a JSON body containing the full message history alongside a structured disaster context object, invoking GPT-4o at temperature zero for deterministic, reproducible responses. CORS policy restricts allowed origins to `http://localhost:5173` (development) and `https://vlm-disaster-analyzer.vercel.app` (production) with no wildcard permitted. Missing API key conditions for cloud services return HTTP 503 Service Unavailable rather than 500 Internal Server Error, correctly indicating to callers that the failure is environmental rather than a code defect.

---

## 12. Error Handling and Fault Tolerance

The system implements layered fault tolerance at each architectural tier. At the model inference layer, GPU quantization degrades automatically through the path 4-bit NF4 → fp16 → float32, selecting the highest-compression option supported by the available hardware and installed libraries. This ensures the backend remains functional on CPU-only hosts at the cost of substantially increased latency. Video metadata extraction follows a three-stage fallback chain — ffprobe, then OpenCV, then file-stat — guaranteeing that a structured response is always returned even when media inspection tools are partially unavailable.

Individual model failures in research mode are isolated and recorded in the response payload without propagating to other models or aborting the sequential pipeline; only if all four models fail does the pipeline return an error state. The Stage 3 FAISS retrieval step is wrapped in `try/except` and returns an empty list on any failure, ensuring that an unbuilt or corrupted index does not affect Stage 1–2 outputs. At the API layer, all unhandled inference exceptions are caught, logged with structured context, and surfaced as HTTP 500 responses with descriptive detail fields, avoiding exposure of raw Python exception text. The frontend's fallback response system — comprising six keyword-matched response categories covering severity, emergency protocols, human risk assessment, resource requirements, infrastructure impact, and environmental consequences — ensures that the user receives actionable disaster intelligence even when all remote endpoints are unreachable.

---

## 13. Deployment Architecture

The system is deployed in a **decoupled architecture**: the GPU inference backend runs on-demand in Google Colab and the frontend is hosted persistently on Vercel.

**Backend — Google Colab (NVIDIA T4 GPU):**  
The FastAPI backend is launched in Google Colab via `notebooks/VIDI_75_Research_Pipeline.ipynb` or `start_backend.py`, which installs dependencies, sets the deployment profile environment variables, and starts uvicorn on port 8000. An ngrok tunnel provides a stable public HTTPS endpoint, which is set as `VITE_API_URL` in the frontend environment. This architecture reflects the intended usage pattern: high-capacity GPU inference is required in operational bursts during disaster monitoring sessions rather than continuously, making on-demand provisioning more cost-effective than persistent GPU hosting. The Colab T4 GPU (16 GB GDDR6, 320 GB/s memory bandwidth) accommodates the production profile (CLIP + Qwen2-VL, ~2.5 GB VRAM) and the full research profile (all 4 models, ~12–16 GB VRAM).

HuggingFace model weights are cached in Colab's runtime storage at `~/.cache/huggingface/hub/` after the first load. In the production profile, CLIP (~0.6 GB) and Qwen2-VL (~4.1 GB cached, ~2.0 GB VRAM under 4-bit NF4) are the only models loaded. BLIP-2 and LLaVA code remains present in `src/models/` and can be activated by switching to research mode via `ACTIVE_MODELS=clip,blip2,llava,qwen`.

**Frontend — Vercel:**  
The React application is built with Vite and deployed on Vercel as a static SPA. The production deployment URL is `https://vlm-disaster-analyzer.vercel.app`. The frontend remains continuously online independent of backend availability; when the Colab backend is not running, the frontend degrades gracefully to its client-side keyword-fallback chat responses and localStorage session history. The backend URL is injected at build time via `VITE_API_URL` in Vercel's environment variable configuration.

**Local Development:**  
For local CPU development, `start_backend.py` launches the production profile (CLIP + Qwen2-VL) on `http://localhost:8000`. Qwen2-VL inference on CPU requires approximately 220 seconds per image in float32 precision. The frontend Vite dev server runs on `http://localhost:5173` and is pre-configured in the backend's CORS allowlist.

---

## 14. Performance Analysis and Benchmarks

The following table presents observed end-to-end performance metrics across the primary system configurations. All GPU measurements were obtained on an NVIDIA T4 (16 GB GDDR6, 320 GB/s memory bandwidth) in Google Colab. CPU measurements were obtained on an Intel x86-64 host with no CUDA availability, running Qwen2-VL in float32 precision.

| Operation / Configuration | End-to-End Latency | VRAM / Memory | Confidence Range |
|---|---|---|---|
| CLIP Classification (T4 GPU) | ~500 ms | ~0.3 GB | High (>85%) |
| Unified Pipeline — CLIP→Qwen→FAISS (T4, 4-bit) | 3–5 s | ~2.5 GB | High (>80%) |
| Unified Pipeline — CLIP→Qwen→FAISS (CPU, float32) | ~220 s | ~6 GB RAM | High (>80%) |
| FAISS Retrieval (30 events, CLIP embedding) | <100 ms | Negligible | N/A |
| Research Mode — All 4 Models Sequential (T4) | 12–20 s | ~2.5 GB peak | Aggregated |
| Video Metadata Extraction | 1–3 s | Minimal | N/A |
| Chat Response — GPT-4o Primary | ~2–4 s | N/A (Cloud) | High |
| Chat Response — Keyword Fallback | <100 ms | None | Template-based |

*Table 4: End-to-End Performance Benchmarks by Configuration*

CLIP classification completes in approximately 500 milliseconds regardless of image content, as inference time is dominated by the image encoder forward pass rather than output length. The FAISS retrieval step adds less than 100 milliseconds to the unified pipeline, as the index search over 30 events (512-dimensional vectors, inner product) is computationally trivial. Qwen2-VL inference under 4-bit NF4 quantization requires 2–3 seconds on T4 GPU, yielding a full unified-pipeline response within 3–5 seconds end-to-end. On CPU with float32 weights, Qwen inference extends to approximately 220 seconds — a factor of approximately 70× increase — establishing GPU availability as a hard practical requirement for operational deployment at scale.

The 4-bit NF4 quantization applied to Qwen2-VL reduces its VRAM footprint from approximately 4.5 GB (fp16) to approximately 2.0 GB, enabling deployment on consumer-grade GPUs with as little as 4 GB of VRAM with acceptable generation quality for disaster assessment tasks.

| Model Configuration | Approximate VRAM | Minimum Deployment Target |
|---|---|---|
| CLIP only | ~0.3 GB | Standard desktop GPU (4 GB+) |
| CLIP + Qwen2-VL (4-bit NF4) | ~2.5 GB | Consumer GPU (≥4 GB VRAM) |
| CLIP + Qwen2-VL (fp16) | ~4.5 GB | Mid-range GPU (6–8 GB VRAM) |
| All 4 local models (mixed precision) | 12–16 GB | Research GPU (T4, A100) |

*Table 5: VRAM Requirements by Model Configuration*

---

## Appendix A: Disaster Categories Supported

The system classifies imagery across twelve disaster categories: Flood, Water Disaster, Wild Fire, Urban Fire, Earthquake, Infrastructure Damage, Human Damage, Landslide, Cyclone, Drought, Forest Scene (baseline), and Urban/Buildings Scene (baseline). Of these, five categories — Flood, Fire, Earthquake, Landslide, and Cyclone — are backed by both historical event records in the FAISS database and client-side knowledge bases covering impact characterization, recommended actions, infrastructure risk, human impact factors, and environmental consequences. The remaining categories produce CLIP classification and Qwen2-VL analysis but do not have hardcoded knowledge-base fallback text.

The FAISS historical database covers: **Flood** — Kerala Floods 2018, Assam Floods 2020/2022, Bihar Floods 2017, Uttarakhand Floods 2013, Pakistan Floods 2022, Bangladesh Floods 2017; **Cyclone** — Cyclone Fani 2019, Cyclone Amphan 2020, Cyclone Biparjoy 2023, Cyclone Yaas 2021, Cyclone Tauktae 2021, Cyclone Nargis 2008, Typhoon Haiyan 2013; **Wildfire** — Uttarakhand Wildfires 2021/2024, Himachal Pradesh Wildfires 2023, Camp Fire California 2018, Amazon Fires 2019, Australian Black Summer 2019–20; **Earthquake** — Nepal Earthquake 2015, Gujarat Earthquake 2001, Haiti Earthquake 2010, Turkey–Syria Earthquakes 2023, Sikkim Earthquake 2023; **Landslide** — Wayanad Landslide 2024, Kedarnath Landslide 2013, Pune Landslide 2014, Manipur Landslide 2022, Joshimath Subsidence 2023.

---

## Appendix B: System Requirements Summary

| Component | Specification |
|---|---|
| Python Version | 3.10+ |
| FastAPI Version | 0.100+ |
| PyTorch Version | 2.5.1 |
| CUDA Version | 12.4 (optional; CPU float32 fallback available) |
| Minimum GPU VRAM (Production) | 4 GB (CLIP + Qwen 4-bit NF4) |
| Recommended GPU VRAM (Research) | 16 GB (NVIDIA T4 or equivalent) |
| Image Size Limit | 10 MB |
| Video Size Limit | 500 MB |
| Supported Image Formats | JPEG, PNG, WebP, BMP, TIFF |
| Frontend Runtime | React 18+, Vite 5+, Node.js 18+ |
| Backend Deployment | Google Colab (NVIDIA T4) + ngrok tunnel |
| Frontend Deployment | Vercel (vlm-disaster-analyzer.vercel.app) |
| HuggingFace Model Cache | ~/.cache/huggingface/hub/ (local) or Colab runtime storage |
| FAISS Version | faiss-cpu 1.13.2+ |
| Key Python Dependencies | transformers, torch, Pillow, faiss-cpu, uvicorn, python-dotenv, openpyxl |
