from transformers import Blip2Processor, Blip2ForConditionalGeneration
from pathlib import Path
from PIL import Image
import threading
import torch


# -------------------------------------------------------------------
# Device Configuration
# -------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------
# Local Model Path
# -------------------------------------------------------------------

MODEL_PATH = "Salesforce/blip2-opt-2.7b"


# -------------------------------------------------------------------
# Lazy Loading Setup
# -------------------------------------------------------------------

# float16 on GPU (~5.4 GB VRAM) — float32 on CPU (~11 GB RAM)
# Models are NOT loaded at import time. _load_model() is called on
# the first inference request and caches the result for all subsequent
# requests. This keeps server startup instant and VRAM usage on-demand.

dtype = torch.float16 if device == "cuda" else torch.float32

_model:     Blip2ForConditionalGeneration | None = None
_processor: Blip2Processor | None               = None
_lock = threading.Lock()


# -------------------------------------------------------------------
# Model Loader (lazy singleton with double-checked locking)
# -------------------------------------------------------------------

def _load_model():
    global _model, _processor

    if _model is None:
        with _lock:
            if _model is None:
                _processor = Blip2Processor.from_pretrained(MODEL_PATH)
                _model = Blip2ForConditionalGeneration.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=dtype,
                    device_map="auto"
                )
                _model.eval()

    return _model, _processor


# -------------------------------------------------------------------
# Inference Settings
# -------------------------------------------------------------------

MAX_NEW_TOKENS = 120


# -------------------------------------------------------------------
# Caption Prompt
# -------------------------------------------------------------------

# Conditional captioning: supplying a text prefix steers BLIP-2's
# OPT decoder toward disaster-domain vocabulary instead of generic
# object descriptions. The model completes the sentence from the prefix.
#
# Format: "Question: <question> Answer:" is the standard BLIP-2 VQA
# format for the OPT decoder and produces focused, factual responses.

PROMPT = (
    "Question: What natural disaster is shown in this image "
    "and what damage or environmental impact is visible? "
    "Answer:"
)


# -------------------------------------------------------------------
# Caption Generation Function
# -------------------------------------------------------------------

def predict_caption(image_path):

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

    # Conditional captioning: pass both the prompt and image so the
    # OPT decoder generates a disaster-specific response continuation.

    inputs = processor(
        text=PROMPT,
        images=image,
        return_tensors="pt"
    ).to(device)


    # ---------------------------------------------------------------
    # Generate caption
    # ---------------------------------------------------------------

    with torch.no_grad():

        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            output_scores=True,
            return_dict_in_generate=True
        )


    # ---------------------------------------------------------------
    # Decode output — trim input tokens
    # ---------------------------------------------------------------

    # model.generate() returns full sequence (prompt + generated).
    # Slice off the input token IDs so only the new caption is decoded.

    trimmed = generated.sequences[:, inputs.input_ids.shape[1]:]

    caption = processor.batch_decode(
        trimmed,
        skip_special_tokens=True
    )[0].strip()


    # ---------------------------------------------------------------
    # Generation confidence from per-token probabilities
    # ---------------------------------------------------------------

    token_probs = []

    for score in generated.scores:
        probs = torch.softmax(score.float(), dim=-1)
        token_probs.append(probs.max().item())

    confidence = (
        sum(token_probs) / len(token_probs) * 100
        if token_probs else 0
    )


    # ---------------------------------------------------------------
    # Debug logging
    # ---------------------------------------------------------------

    print("\nBLIP-2 Caption:")
    print(caption)
    print(f"Confidence: {round(confidence, 2)}%")


    # ---------------------------------------------------------------
    # Return response
    # ---------------------------------------------------------------

    return {
        "model": "BLIP-2",
        "caption": caption,
        "confidence": round(confidence, 2)
    }


# -------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------

if __name__ == "__main__":

    image_path = "test.jpg"

    result = predict_caption(image_path)

    print("\nCaption Result")
    print("-------------------------")
    print(f"Model      : {result['model']}")
    print(f"Caption    : {result['caption']}")
    print(f"Confidence : {result['confidence']}%")
