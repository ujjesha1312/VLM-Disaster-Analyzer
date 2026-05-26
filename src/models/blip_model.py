from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


# -------------------------------------------------------------------
# Device Configuration
# -------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------
# Local Model Path
# -------------------------------------------------------------------

MODEL_PATH = "./models/blip-image-captioning-base"


# -------------------------------------------------------------------
# Load BLIP Model
# -------------------------------------------------------------------

processor = BlipProcessor.from_pretrained(MODEL_PATH)

model = BlipForConditionalGeneration.from_pretrained(
    MODEL_PATH
).to(device)


# -------------------------------------------------------------------
# Prediction Function
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

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Move tensors to device
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }


    # ---------------------------------------------------------------
    # Generate caption
    # ---------------------------------------------------------------

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=50
        )


    # ---------------------------------------------------------------
    # Decode output
    # ---------------------------------------------------------------

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )


    # ---------------------------------------------------------------
    # Debug logging
    # ---------------------------------------------------------------

    print("\nGenerated Caption:")
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