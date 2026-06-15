import logging

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from pathlib import Path
from PIL import Image
import sys
import threading
import torch

logger = logging.getLogger(__name__)

# Make src/ importable when this file is run standalone.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.metrics import confidence_to_level  # noqa: E402


# -------------------------------------------------------------------
# Device Configuration
# -------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------
# Local Model Path
# -------------------------------------------------------------------

MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"


# -------------------------------------------------------------------
# Optional 4-bit Quantization (CUDA + bitsandbytes only)
# -------------------------------------------------------------------

# Reduces VRAM from ~4 GB (fp16) to ~2 GB (4-bit NF4).
# Silently skipped on CPU or when bitsandbytes is not installed.

def _get_bnb_config():
    if device != "cuda":
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
        return None


# -------------------------------------------------------------------
# Lazy Loading Setup
# -------------------------------------------------------------------

# float16 on GPU (~4 GB VRAM) — float32 on CPU (~8 GB RAM)
# 4-bit reduces VRAM to ~2 GB when bitsandbytes is available on CUDA.
# Models are NOT loaded at import time. _load_model() is called on
# the first inference request and caches the result for all subsequent
# requests. This keeps server startup instant and VRAM usage on-demand.

dtype = torch.float16 if device == "cuda" else torch.float32

_model:     Qwen2VLForConditionalGeneration | None = None
_processor: AutoProcessor | None                   = None
_lock = threading.Lock()


# -------------------------------------------------------------------
# Model Loader (lazy singleton with double-checked locking)
# -------------------------------------------------------------------

def _load_model():
    global _model, _processor

    if _model is None:
        with _lock:
            if _model is None:
                _processor = AutoProcessor.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    local_files_only=True,
                )

                bnb_config = _get_bnb_config()

                if bnb_config is not None:
                    _model = Qwen2VLForConditionalGeneration.from_pretrained(
                        MODEL_PATH,
                        quantization_config=bnb_config,
                        device_map={"": 0},
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                else:
                    # On CPU: no device_map — accelerate's "auto" shards tensors
                    # incorrectly when there is no GPU, causing shape mismatches.
                    _model = Qwen2VLForConditionalGeneration.from_pretrained(
                        MODEL_PATH,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        local_files_only=True,
                    )

                _model.eval()

    return _model, _processor


# -------------------------------------------------------------------
# Inference Settings
# -------------------------------------------------------------------

MAX_NEW_TOKENS = 1024


# -------------------------------------------------------------------
# Structured Disaster Analysis Prompt
# -------------------------------------------------------------------

PROMPT = (
    "Analyze this disaster image.\n"
    "Return ONLY in this exact format:\n"
    "DISASTER TYPE:\n"
    "SEVERITY:\n"
    "ONE-LINE DESCRIPTION:"
)


# -------------------------------------------------------------------
# Response Parser
# -------------------------------------------------------------------

_FIELD_MAP = {
    "DISASTER TYPE":      "disaster_type",
    "SEVERITY":           "severity",
    "AFFECTED POPULATION": "affected_population",
    "INFRASTRUCTURE":     "infrastructure_status",
    "ENVIRONMENT":        "environmental_impact",
}


def _parse_fields(text: str):
    """Return (fields_dict, parsed_confidence_or_None)."""
    fields = {v: "" for v in _FIELD_MAP.values()}
    parsed_confidence = None

    for line in text.splitlines():
        stripped = line.strip()
        for label, key in _FIELD_MAP.items():
            if stripped.startswith(label + ":"):
                fields[key] = stripped[len(label) + 1:].strip()
                break
        if stripped.startswith("CONFIDENCE:"):
            val = stripped[len("CONFIDENCE:"):].strip()
            try:
                parsed_confidence = float(val)
            except ValueError:
                pass

    return fields, parsed_confidence


# -------------------------------------------------------------------
# Response Generation Function
# -------------------------------------------------------------------

def predict_response(image_path, prompt: str | None = None):

    # ---------------------------------------------------------------
    # Ensure model is loaded (no-op after first call)
    # ---------------------------------------------------------------

    model, processor = _load_model()


    # ---------------------------------------------------------------
    # Load image
    # ---------------------------------------------------------------

    try:
        image = Image.open(image_path).convert("RGB")

    except Exception as exc:
        raise ValueError(f"Invalid image file: {exc}")


    # ---------------------------------------------------------------
    # Build chat message
    # ---------------------------------------------------------------

    # Qwen2-VL chat format: typed content blocks inside a user turn.
    # apply_chat_template converts this to a flat text string the
    # tokenizer can process, including the generation prompt marker.

    effective_prompt = prompt if prompt is not None else PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  effective_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(device)


    # ---------------------------------------------------------------
    # Generate response
    # ---------------------------------------------------------------

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )


    # ---------------------------------------------------------------
    # Decode output
    # ---------------------------------------------------------------

    # Trim input token IDs so only the newly generated tokens are
    # decoded — prevents the prompt from appearing in the response.

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated.sequences)
    ]

    response = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


    # ---------------------------------------------------------------
    # Generation confidence from per-token probabilities
    # ---------------------------------------------------------------

    token_probs = []

    for score in generated.scores:
        probs = torch.softmax(score.float(), dim=-1)
        token_probs.append(probs.max().item())

    token_confidence = (
        sum(token_probs) / len(token_probs) * 100
        if token_probs else 0
    )


    # ---------------------------------------------------------------
    # Parse fields; use model-reported CONFIDENCE if valid
    # ---------------------------------------------------------------

    fields, parsed_conf = _parse_fields(response)

    if parsed_conf is not None and 0 <= parsed_conf <= 100:
        confidence_score = round(parsed_conf, 2)
    else:
        confidence_score = round(token_confidence, 2)


    # ---------------------------------------------------------------
    # Debug logging
    # ---------------------------------------------------------------

    logger.debug(f"Qwen2-VL Response:\n{response}")
    logger.debug(f"Token confidence: {round(token_confidence, 2)}%")
    logger.debug(f"Final confidence: {confidence_score}%")


    # ---------------------------------------------------------------
    # Return structured metrics
    # ---------------------------------------------------------------

    return {
        "model": "Qwen-VL",
        "metrics": {
            **fields,
            "confidence_score": confidence_score,
            "confidence_level": confidence_to_level(confidence_score),
            "raw_analysis":     response,
        }
    }


# -------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    image_path = "test.jpg"

    result = predict_response(image_path)

    logger.info("\nScene Understanding Result")
    logger.info("-------------------------")
    logger.info(f"Model         : {result['model']}")
    logger.info(f"Disaster Type : {result['metrics']['disaster_type']}")
    logger.info(f"Severity      : {result['metrics']['severity']}")
    logger.info(f"Confidence    : {result['metrics']['confidence_score']}%")
