# ── VLM Disaster Analyzer — Backend ──────────────────────────────────────────
# Base image includes PyTorch 2.5, CUDA 12.4, cuDNN 9 (runtime, not devel).
# Use the devel variant if you need to compile CUDA extensions (e.g. flash-attn).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# ffmpeg  — video processing (backend/services/video_service.py)
# libglib / libsm / libxext / libxrender — OpenCV headless requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Runtime configuration ─────────────────────────────────────────────────────
# These defaults match the deployment profile (CLIP + Qwen only).
# Override at runtime: docker run -e ACTIVE_MODELS=clip,blip2,llava,qwen ...
ENV ACTIVE_MODELS=clip,qwen
ENV QUANTIZE_QWEN=true
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
