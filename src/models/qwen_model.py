"""
qwen_model.py — Qwen2-VL multimodal visual understanding for disaster images.

Architecture:
  - Qwen/Qwen2-VL-2B-Instruct  (2B parameters — the lightest viable chat-capable VLM)
  - 4-bit quantization via bitsandbytes when available on CUDA
  - Falls back to float16 (CUDA) or float32 (CPU) without bitsandbytes
  - Chat-template message format (unlike LLaVA's prompt-string format)
  - Lazy singleton

Inference flow:
  Image + Chat Message → AutoProcessor (apply_chat_template) → tokenize
  → Qwen2VLForConditionalGeneration.generate() → trim input tokens → decode → response

Why Qwen2-VL-2B over larger alternatives:
  At 2B parameters it loads in ~4 GB RAM on CPU without quantization, making it
  the most practical option for multimodal chat inference on typical developer hardware.
  It outperforms many 7B models from a year ago on visual understanding benchmarks.

Memory requirements:
  - With 4-bit (bitsandbytes + CUDA): ~2 GB VRAM
  - With float16 (CUDA, no quant):   ~4 GB VRAM
  - With float32 (CPU):              ~8 GB RAM
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL_ID       = "Qwen/Qwen2-VL-2B-Instruct"
_DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE          = torch.float16 if _DEVICE.type == "cuda" else torch.float32
_MAX_NEW_TOKENS = 200

# Visual analysis question — open-ended to elicit multimodal scene understanding.
_QUESTION = (
    "Analyze this disaster image in detail. "
    "What type of natural disaster is occurring? "
    "Describe the visible damage, the affected environment, "
    "and any observable impact on infrastructure or human settlements."
)


# ---------------------------------------------------------------------------
# Optional 4-bit quantization (bitsandbytes)
# ---------------------------------------------------------------------------

def _get_bnb_config() -> Optional[object]:
    """Return BitsAndBytesConfig for 4-bit NF4 quantization, or None if unavailable."""
    if _DEVICE.type != "cuda":
        return None
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        logger.warning(
            "bitsandbytes not installed — Qwen2-VL will load in fp16. "
            "Install with: pip install bitsandbytes"
        )
        return None


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_model:     Qwen2VLForConditionalGeneration | None = None
_processor: AutoProcessor | None                   = None


def _load_model() -> tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    """Load Qwen2-VL processor and model (once), with optional 4-bit quantization."""
    global _model, _processor

    if _model is None:
        logger.info("Loading Qwen2-VL '%s' on %s ...", _MODEL_ID, _DEVICE)
        _processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)

        bnb_config = _get_bnb_config()

        if bnb_config is not None:
            logger.info("Using 4-bit NF4 quantization (bitsandbytes).")
            _model = Qwen2VLForConditionalGeneration.from_pretrained(
                _MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            _model = Qwen2VLForConditionalGeneration.from_pretrained(
                _MODEL_ID,
                torch_dtype=_DTYPE,
                trust_remote_code=True,
            ).to(_DEVICE)

        _model.eval()
        logger.info("Qwen2-VL ready.")

    return _model, _processor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_disaster(image_path: str) -> dict:
    """Run Qwen2-VL multimodal visual analysis on a disaster image.

    Constructs a chat message with the image and analysis question, applies the
    model's chat template, runs generation, then trims input tokens from the
    output to return only the newly generated response text.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        {
            "model":    "Qwen-VL",
            "response": str,  # e.g. "The scene shows flood water covering a road..."
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

    # Qwen2-VL uses a structured chat message with typed content blocks.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  _QUESTION},
            ],
        }
    ]

    # Convert messages to the model's flat text format.
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)

    # Trim input token IDs — keep only the newly generated tokens for decoding.
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    logger.debug("Qwen2-VL response: '%s'", response)

    return {
        "model":    "Qwen-VL",
        "response": response,
    }
