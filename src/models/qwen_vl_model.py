import logging
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
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

_QUESTION = (
    "What type of natural disaster is shown in this image? "
    "Choose exactly one from: flooding, wildfire, earthquake, landslide, cyclone, tsunami."
)

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_model:     Qwen2VLForConditionalGeneration | None = None
_processor: AutoProcessor | None                   = None


def _load_model() -> tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    global _model, _processor
    if _model is None:
        logger.info("Loading Qwen2-VL '%s' on %s ...", _MODEL_ID, _DEVICE)
        _processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            _MODEL_ID,
            torch_dtype=_DTYPE,
            trust_remote_code=True,
        ).to(_DEVICE)
        logger.info("Qwen2-VL ready.")
    return _model, _processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_label(text: str) -> str:
    lower = text.lower()
    for label in LABELS:
        if label in lower:
            return _LABEL_TO_FULL[label]
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_disaster(image_path: str) -> dict:
    """Classify a disaster image using Qwen2-VL-2B-Instruct.

    Returns:
        {
            "model": "qwen_vl",
            "prediction": str,
            "confidence": None,
            "raw_output": str,
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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  _QUESTION},
            ],
        }
    ]

    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)

    # Trim input tokens to keep only the newly generated part
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return {
        "model":      "qwen_vl",
        "prediction": _parse_label(raw_output),
        "confidence": None,
        "raw_output": raw_output,
    }
