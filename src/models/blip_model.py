"""
blip_model.py — BLIP unconditional image captioning for disaster scenes.

Architecture:
  - Salesforce/blip-image-captioning-base (~990 MB, CPU-friendly)
  - Unconditional captioning: no text prompt supplied — model describes freely
  - Lazy singleton: loads once per process

Inference flow:
  Image → BlipProcessor (resize + normalize) → BlipForConditionalGeneration.generate()
       → decode token IDs → natural language caption

Why blip-image-captioning-base over blip2-opt-2.7b:
  - 5× smaller: 990 MB vs 5+ GB
  - Runs on CPU without quantization
  - Sufficient for captioning without VQA
"""

import logging
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL_ID       = "Salesforce/blip-image-captioning-base"
_DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE          = torch.float16 if _DEVICE.type == "cuda" else torch.float32
_MAX_NEW_TOKENS = 60   # enough for a full sentence caption


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_model:     BlipForConditionalGeneration | None = None
_processor: BlipProcessor | None               = None


def _load_model() -> tuple[BlipForConditionalGeneration, BlipProcessor]:
    """Download / load BLIP from HuggingFace Hub on first call, then reuse."""
    global _model, _processor

    if _model is None:
        logger.info("Loading BLIP '%s' on %s ...", _MODEL_ID, _DEVICE)
        _processor = BlipProcessor.from_pretrained(_MODEL_ID)
        _model     = (
            BlipForConditionalGeneration
            .from_pretrained(_MODEL_ID, torch_dtype=_DTYPE)
            .to(_DEVICE)
        )
        _model.eval()
        logger.info("BLIP ready on %s.", _DEVICE)

    return _model, _processor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_disaster(image_path: str) -> dict:
    """Generate a free-form natural language caption for a disaster image.

    No text prompt is supplied — the model describes what it sees without
    any classification constraint, yielding richer contextual captions.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        {
            "model":   "BLIP-2",
            "caption": str,   # e.g. "a flooded road surrounded by trees and water"
        }

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError:        If the file is not a valid image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Not a valid image file: {image_path}") from exc

    model, processor = _load_model()

    # Pass image only — unconditional captioning.
    inputs = processor(images=image, return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)

    caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    logger.debug("BLIP caption: '%s'", caption)

    return {
        "model":   "BLIP-2",
        "caption": caption,
    }
