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

MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "models" / "blip2-opt-2.7b"
)


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

MAX_NEW_TOKENS = 100


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
    # Process image
    # ---------------------------------------------------------------

    # BLIP-2 accepts image only for unconditional captioning.
    # Inputs moved to device to match model placement.

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)


    # ---------------------------------------------------------------
    # Generate caption
    # ---------------------------------------------------------------

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )


    # ---------------------------------------------------------------
    # Decode output
    # ---------------------------------------------------------------

    # BLIP-2 uses batch_decode (returns list) — take first element
    # This differs from original BLIP which used decode on ids[0]

    caption = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()


    # ---------------------------------------------------------------
    # Debug logging
    # ---------------------------------------------------------------

    print("\nBLIP-2 Caption:")
    print(caption)


    # ---------------------------------------------------------------
    # Return response
    # ---------------------------------------------------------------

    return {
        "model": "BLIP-2",
        "caption": caption
    }


# -------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------

if __name__ == "__main__":

    image_path = "test.jpg"

    result = predict_caption(image_path)

    print("\nCaption Result")
    print("-------------------------")
    print(f"Model   : {result['model']}")
    print(f"Caption : {result['caption']}")
