import logging

from transformers import AutoProcessor, LlavaForConditionalGeneration
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

MODEL_PATH = "llava-hf/llava-1.5-7b-hf"


# -------------------------------------------------------------------
# Lazy Loading Setup
# -------------------------------------------------------------------

# float16 on GPU halves memory (~14 GB → ~7 GB VRAM)
# float32 on CPU is standard (~28 GB RAM — large model, be patient)
# Models are NOT loaded at import time. _load_model() is called on
# the first inference request and caches the result for all subsequent
# requests. This keeps server startup instant and VRAM usage on-demand.

dtype = torch.float16 if device == "cuda" else torch.float32

_model:     LlavaForConditionalGeneration | None = None
_processor: AutoProcessor | None                 = None
_lock = threading.Lock()


# -------------------------------------------------------------------
# Model Loader (lazy singleton with double-checked locking)
# -------------------------------------------------------------------

def _load_model():
    global _model, _processor

    if _model is None:
        with _lock:
            if _model is None:
                _processor = AutoProcessor.from_pretrained(MODEL_PATH)
                _model = LlavaForConditionalGeneration.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                _model.eval()

    return _model, _processor


# -------------------------------------------------------------------
# Inference Settings
# -------------------------------------------------------------------

MAX_NEW_TOKENS = 200


# -------------------------------------------------------------------
# Structured Disaster Assessment Prompt
# -------------------------------------------------------------------

# LLaVA-1.5 requires exactly this conversation format.
# The <image> token is replaced with image embeddings by the processor.
#
# Enforcing a strict labeled-field format makes outputs machine-parseable
# while preserving the per-token probability confidence scoring — only
# the prompt text changes, not the generation/scoring logic.

PROMPT = (
    "USER: <image>\n"
    "Analyze this disaster image and respond ONLY in this exact format:\n"
    "DISASTER TYPE: [type]\n"
    "SEVERITY: [Critical/High/Moderate/Low]\n"
    "AFFECTED AREA: [description]\n"
    "INFRASTRUCTURE DAMAGE: [Yes/No - description]\n"
    "RECOMMENDED ACTION: [action]\n"
    "ASSISTANT:"
)


# -------------------------------------------------------------------
# Response Parser
# -------------------------------------------------------------------

_FIELD_MAP = {
    "DISASTER TYPE":       "disaster_type",
    "SEVERITY":            "severity",
    "AFFECTED AREA":       "affected_areas",
    "INFRASTRUCTURE DAMAGE": "infrastructure_damage",
    "RECOMMENDED ACTION":  "recommended_action",
}


def _parse_fields(text: str) -> dict:
    fields = {v: "" for v in _FIELD_MAP.values()}
    for line in text.splitlines():
        for label, key in _FIELD_MAP.items():
            if line.startswith(label + ":"):
                fields[key] = line[len(label) + 1:].strip()
                break
    return fields


# -------------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------------

def predict_response(image_path):

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
    # Process image + prompt
    # ---------------------------------------------------------------

    inputs = processor(
        text=PROMPT,
        images=image,
        return_tensors="pt"
    )

    # Move tensors to the same device as the model
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }


    # ---------------------------------------------------------------
    # Generate response
    # ---------------------------------------------------------------

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            output_scores=True,
            return_dict_in_generate=True
        )


    # ---------------------------------------------------------------
    # Decode output
    # ---------------------------------------------------------------

    full_text = processor.decode(
        output.sequences[0],
        skip_special_tokens=True
    )


    # ---------------------------------------------------------------
    # Extract ASSISTANT response only
    # ---------------------------------------------------------------

    # LLaVA echoes the full prompt in the output — split on ASSISTANT:
    # to keep only the generated reasoning response.

    if "ASSISTANT:" in full_text:
        response = full_text.split("ASSISTANT:")[-1].strip()
    else:
        response = full_text.strip()


    # ---------------------------------------------------------------
    # Generation confidence from per-token probabilities
    # ---------------------------------------------------------------

    token_probs = []

    for score in output.scores:
        probs = torch.softmax(score.float(), dim=-1)
        token_probs.append(probs.max().item())

    confidence = (
        sum(token_probs) / len(token_probs) * 100
        if token_probs else 0
    )
    confidence_score = round(confidence, 2)


    # ---------------------------------------------------------------
    # Debug logging
    # ---------------------------------------------------------------

    logger.debug("LLaVA Response:\n%s", response)
    logger.debug("Confidence: %s%%", confidence_score)


    # ---------------------------------------------------------------
    # Parse labeled fields and return structured metrics
    # ---------------------------------------------------------------

    fields = _parse_fields(response)

    return {
        "model": "LLaVA",
        "metrics": {
            **fields,
            "confidence_score": confidence_score,
            "confidence_level": confidence_to_level(confidence_score),
            "raw_assessment":   response,
        }
    }


# -------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    image_path = "test.jpg"

    result = predict_response(image_path)

    logger.info("Reasoning Result")
    logger.info("-------------------------")
    logger.info("Model         : %s", result["model"])
    logger.info("Disaster Type : %s", result["metrics"]["disaster_type"])
    logger.info("Severity      : %s", result["metrics"]["severity"])
    logger.info("Confidence    : %s%%", result["metrics"]["confidence_score"])
