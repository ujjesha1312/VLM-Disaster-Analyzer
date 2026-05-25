"""
clip_model.py — CLIP zero-shot disaster classification.

Architecture:
  - Local CLIP ViT-B/32 checkpoint (./models/clip-vit-base-patch32)
  - Computes cosine similarity between image embedding and disaster-type text prompts
  - Picks the highest-similarity label and reports softmax probability as confidence
  - Lazy singleton: model loads once on first request, reused for all subsequent calls

Inference flow:
  Image → CLIPProcessor → image+text embeddings → cosine similarity → softmax → label + confidence
"""

import logging
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Resolved absolute path so the server can be started from any working directory.
_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "models" / "clip-vit-base-patch32"
)

# Text prompts — each is compared against the image embedding.
# More descriptive prompts yield better zero-shot accuracy than bare nouns.
_PROMPTS: list[str] = [
    "flood disaster with submerged roads and buildings",
    "wildfire disaster with flames and dense smoke",
    "earthquake destruction with collapsed structures",
    "landslide disaster with mud and debris covering roads",
    "cyclone or hurricane damage with destroyed buildings",
    "tsunami disaster with large waves hitting the coast",
]

# Human-readable labels aligned 1-to-1 with _PROMPTS.
_LABELS: list[str] = [
    "Flood",
    "Wildfire",
    "Earthquake",
    "Landslide",
    "Cyclone",
    "Tsunami",
]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_model:     CLIPModel | None     = None
_processor: CLIPProcessor | None = None


def _load_model() -> tuple[CLIPModel, CLIPProcessor]:
    """Load CLIP processor and model from the local checkpoint (once)."""
    global _model, _processor

    if _model is None:
        logger.info("Loading CLIP from local path '%s' on %s ...", _MODEL_PATH, _DEVICE)
        _processor = CLIPProcessor.from_pretrained(_MODEL_PATH)
        _model     = CLIPModel.from_pretrained(_MODEL_PATH).to(_DEVICE)
        _model.eval()
        logger.info("CLIP loaded and ready on %s.", _DEVICE)

    return _model, _processor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_disaster(image_path: str) -> dict:
    """Run zero-shot CLIP classification on a disaster image.

    Encodes the image and all disaster-type text prompts, computes cosine
    similarities, applies softmax, and returns the top label with its
    probability expressed as a 0–100 confidence score.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        {
            "model":      "CLIP",
            "prediction": str,    # e.g. "Flood"
            "confidence": float,  # 0.0 – 100.0
        }

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError:        If the file cannot be opened as an image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Not a valid image file: {image_path}") from exc

    model, processor = _load_model()

    # Encode image + all text prompts in one batched forward pass.
    inputs = processor(
        text=_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(_DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # logits_per_image: (1, num_prompts) — higher = more similar
    probs         = outputs.logits_per_image.softmax(dim=1)   # → probabilities
    predicted_idx = probs.argmax().item()
    prediction    = _LABELS[predicted_idx]
    confidence    = round(probs[0][predicted_idx].item() * 100, 2)

    # Log per-class breakdown for debugging / benchmarking.
    scores = {lbl: round(probs[0][i].item() * 100, 2) for i, lbl in enumerate(_LABELS)}
    logger.debug("CLIP scores: %s", scores)

    return {
        "model":      "CLIP",
        "prediction": prediction,
        "confidence": confidence,
    }
