import logging
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import Blip2ForConditionalGeneration, Blip2Processor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL_ID = "Salesforce/blip2-opt-2.7b"
_DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE    = torch.float16 if _DEVICE.type == "cuda" else torch.float32

LABELS: list[str] = [
    "flooding", "wildfire", "earthquake",
    "landslide", "cyclone", "tsunami",
]

_LABEL_TO_FULL: dict[str, str] = {
    "flooding":   "a photo of urban flooding with submerged roads",
    "wildfire":   "a photo of wildfire with flames and smoke",
    "earthquake": "a photo of collapsed buildings after earthquake",
    "landslide":  "a photo of landslide on roads or mountains",
    "cyclone":    "a photo of cyclone destruction with damaged houses",
    "tsunami":    "a photo of tsunami waves hitting coastal areas",
}

_PROMPT = (
    "Question: What type of natural disaster is shown in this image? "
    "Choose one from: flooding, wildfire, earthquake, landslide, cyclone, tsunami. "
    "Answer:"
)

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_model:     Blip2ForConditionalGeneration | None = None
_processor: Blip2Processor | None               = None


def _load_model() -> tuple[Blip2ForConditionalGeneration, Blip2Processor]:
    global _model, _processor
    if _model is None:
        logger.info("Loading BLIP-2 '%s' on %s ...", _MODEL_ID, _DEVICE)
        _processor = Blip2Processor.from_pretrained(_MODEL_ID)
        _model = Blip2ForConditionalGeneration.from_pretrained(
            _MODEL_ID, torch_dtype=_DTYPE
        ).to(_DEVICE)
        logger.info("BLIP-2 ready.")
    return _model, _processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_label(text: str) -> str:
    """Map generated text to the nearest known disaster label."""
    lower = text.lower()
    for label in LABELS:
        if label in lower:
            return _LABEL_TO_FULL[label]
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_disaster(image_path: str) -> dict:
    """Classify a disaster image using BLIP-2 VQA.

    Returns:
        {
            "model": "blip2",
            "prediction": str,   # matched full-label string
            "confidence": None,  # generative model — no score
            "raw_output": str,   # raw decoded text from model
        }
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Not a valid image: {image_path}") from exc

    model, processor = _load_model()

    inputs = processor(images=image, text=_PROMPT, return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20)

    raw_output = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    return {
        "model":      "blip2",
        "prediction": _parse_label(raw_output),
        "confidence": None,
        "raw_output": raw_output,
    }
