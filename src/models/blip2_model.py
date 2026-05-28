from transformers import Blip2Processor, Blip2ForConditionalGeneration
from pathlib import Path
from PIL import Image
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
# Load BLIP-2 Model
# -------------------------------------------------------------------

# float16 on GPU (~5.4 GB VRAM) — float32 on CPU (~11 GB RAM)
# device_map="auto" lets accelerate stream weights across CPU/GPU memory
# This is required for BLIP-2 due to its larger size vs original BLIP

dtype = torch.float16 if device == "cuda" else torch.float32

processor = Blip2Processor.from_pretrained(MODEL_PATH)

model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto"
)

model.eval()


# -------------------------------------------------------------------
# Inference Settings
# -------------------------------------------------------------------

MAX_NEW_TOKENS = 100


# -------------------------------------------------------------------
# Caption Generation Function
# -------------------------------------------------------------------

def predict_caption(image_path):

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
