"""
llava_model.py — LLaVA-1.5 visual scene reasoning for disaster images.

Architecture:
  - llava-hf/llava-1.5-7b-hf  (7B parameters)
  - 4-bit quantization via bitsandbytes when available on CUDA (reduces VRAM ~14 GB → ~5 GB)
  - Falls back to float16 (CUDA) or float32 (CPU) if bitsandbytes is not installed
  - Lazy singleton

Inference flow:
  Image + Instruction Prompt → AutoProcessor → LlavaForConditionalGeneration.generate()
  → strip echoed prompt → natural language scene explanation

Why LLaVA over classification-only models:
  LLaVA performs instruction-following visual reasoning — it can describe *why* a
  scene looks like flooding, what infrastructure is affected, and how severe the
  damage appears. This complements CLIP's categorical label + BLIP's short caption.

Memory requirements:
  - With 4-bit (bitsandbytes + CUDA): ~5 GB VRAM
  - With float16 (CUDA, no quant):   ~14 GB VRAM
  - With float32 (CPU):              ~28 GB RAM  (slow but functional)
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, LlavaForConditionalGeneration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL_ID       = "llava-hf/llava-1.5-7b-hf"
_DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE          = torch.float16 if _DEVICE.type == "cuda" else torch.float32
_MAX_NEW_TOKENS = 200   # reasoning responses need more tokens than single-word answers

# Visual reasoning prompt — LLaVA-1.5 expects the <image> token in the USER turn.
_PROMPT = (
    "USER: <image>\n"
    "You are analyzing a disaster scene photograph. "
    "Identify the type of natural disaster shown, describe the visible damage "
    "to infrastructure or environment, and estimate the apparent severity. "
    "Be specific and concise.\n"
    "ASSISTANT:"
)


# ---------------------------------------------------------------------------
# Optional 4-bit quantization (bitsandbytes)
# ---------------------------------------------------------------------------

def _get_bnb_config() -> Optional[object]:
    """Return a BitsAndBytesConfig for 4-bit NF4 quantization, or None if unavailable."""
    if _DEVICE.type != "cuda":
        return None   # bitsandbytes quantization only applies to CUDA
    try:
        import bitsandbytes  # noqa: F401  — just checking it's installed
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",              # NormalFloat4 — best quality/size tradeoff
            bnb_4bit_compute_dtype=torch.float16,   # matmul in fp16 for speed
            bnb_4bit_use_double_quant=True,          # double-quantize the quantization constants
        )
    except ImportError:
        logger.warning(
            "bitsandbytes not installed — LLaVA will load in fp16 (needs ~14 GB VRAM). "
            "Install it with: pip install bitsandbytes"
        )
        return None


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_model:     LlavaForConditionalGeneration | None = None
_processor: AutoProcessor | None                 = None


def _load_model() -> tuple[LlavaForConditionalGeneration, AutoProcessor]:
    """Load LLaVA processor and model (once), with optional 4-bit quantization."""
    global _model, _processor

    if _model is None:
        logger.info("Loading LLaVA '%s' on %s ...", _MODEL_ID, _DEVICE)
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)

        bnb_config = _get_bnb_config()

        if bnb_config is not None:
            # 4-bit quantized path — device_map="auto" handles multi-GPU / CPU offload
            logger.info("Using 4-bit NF4 quantization (bitsandbytes).")
            _model = LlavaForConditionalGeneration.from_pretrained(
                _MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            # Standard path — float16 on GPU, float32 on CPU
            _model = LlavaForConditionalGeneration.from_pretrained(
                _MODEL_ID,
                torch_dtype=_DTYPE,
                low_cpu_mem_usage=True,   # stream weights to avoid peak OOM on CPU
            ).to(_DEVICE)

        _model.eval()
        logger.info("LLaVA ready.")

    return _model, _processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_assistant_response(full_text: str) -> str:
    """Strip the echoed prompt; return only the ASSISTANT-generated text."""
    # LLaVA-1.5 echoes the full conversation in the decoded output.
    if "ASSISTANT:" in full_text:
        return full_text.split("ASSISTANT:")[-1].strip()
    return full_text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_disaster(image_path: str) -> dict:
    """Run LLaVA-1.5 visual instruction following on a disaster image.

    Produces a free-form scene analysis: identifies the disaster type, describes
    visible damage, and assesses apparent severity — richer than a label alone.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        {
            "model":    "LLaVA",
            "response": str,  # e.g. "The image shows severe flooding affecting roads..."
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

    # Processor takes the prompt string and the PIL image together.
    inputs = processor(_PROMPT, image, return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)

    full_text = processor.decode(output_ids[0], skip_special_tokens=True)
    response  = _extract_assistant_response(full_text)

    logger.debug("LLaVA response: '%s'", response)

    return {
        "model":    "LLaVA",
        "response": response,
    }
