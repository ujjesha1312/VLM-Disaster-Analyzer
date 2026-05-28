from transformers import AutoProcessor, LlavaForConditionalGeneration
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
    Path(__file__).resolve().parent.parent.parent / "models" / "llava-1.5-7b-hf"
)


# -------------------------------------------------------------------
# Load LLaVA Model
# -------------------------------------------------------------------

# float16 on GPU halves memory (~14 GB → ~7 GB VRAM)
# float32 on CPU is standard (~28 GB RAM — large model, be patient)
dtype = torch.float16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,     # stream weights to reduce peak RAM during load
    device_map="auto"           # accelerate handles device placement automatically
)

model.eval()


# -------------------------------------------------------------------
# Inference Settings
# -------------------------------------------------------------------

MAX_NEW_TOKENS = 200


# -------------------------------------------------------------------
# Visual Reasoning Prompt
# -------------------------------------------------------------------

# LLaVA-1.5 requires exactly this conversation format.
# The <image> token is a placeholder the processor replaces with image embeddings.

PROMPT = (
    "USER: <image>\n"
    "You are analyzing a disaster scene photograph. "
    "What type of natural disaster is shown? "
    "Describe the visible damage, affected area, and severity of the conditions.\n"
    "ASSISTANT:"
)


# -------------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------------

def predict_response(image_path):

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
            max_new_tokens=MAX_NEW_TOKENS
        )


    # ---------------------------------------------------------------
    # Decode output
    # ---------------------------------------------------------------

    full_text = processor.decode(
        output[0],
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
    # Debug logging
    # ---------------------------------------------------------------

    print("\nLLaVA Response:")
    print(response)


    # ---------------------------------------------------------------
    # Return response
    # ---------------------------------------------------------------

    return {
        "model": "LLaVA",
        "response": response
    }


# -------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------

if __name__ == "__main__":

    image_path = "test.jpg"

    result = predict_response(image_path)

    print("\nReasoning Result")
    print("-------------------------")
    print(f"Model    : {result['model']}")
    print(f"Response : {result['response']}")
