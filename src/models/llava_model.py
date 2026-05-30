from transformers import AutoProcessor, LlavaForConditionalGeneration
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
    Path(__file__).resolve().parent.parent.parent / "models" / "llava-1.5-7b-hf"
)


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
# Visual Reasoning Prompt
# -------------------------------------------------------------------

# LLaVA-1.5 requires exactly this conversation format.
# The <image> token is replaced with image embeddings by the processor.
#
# Structured prompts consistently outperform open-ended questions for
# disaster analysis: numbered points force the model to address each
# assessment dimension rather than producing a single vague sentence.

PROMPT = (
    "USER: <image>\n"
    "You are a professional disaster assessment analyst examining a field photograph. "
    "Analyze this image and provide a structured response covering each of the following:\n"
    "1. DISASTER TYPE: Identify the specific type of natural disaster depicted.\n"
    "2. VISIBLE DAMAGE: Describe the damage to buildings, roads, infrastructure, or terrain.\n"
    "3. ENVIRONMENTAL IMPACT: Describe effects on vegetation, water bodies, or landscape.\n"
    "4. SEVERITY: Assess the scale and intensity of the disaster (minor / moderate / severe / catastrophic).\n"
    "5. AFFECTED AREA: Characterize the setting (urban, rural, coastal, mountainous, etc.).\n"
    "Provide a concise but thorough professional field assessment.\n"
    "ASSISTANT:"
)


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
