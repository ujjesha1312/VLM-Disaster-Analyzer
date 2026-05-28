from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
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
    Path(__file__).resolve().parent.parent.parent / "models" / "qwen2-vl-2b-instruct"
)


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
                    trust_remote_code=True
                )

                bnb_config = _get_bnb_config()

                if bnb_config is not None:
                    _model = Qwen2VLForConditionalGeneration.from_pretrained(
                        MODEL_PATH,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                else:
                    _model = Qwen2VLForConditionalGeneration.from_pretrained(
                        MODEL_PATH,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        trust_remote_code=True,
                    )

                _model.eval()

    return _model, _processor


# -------------------------------------------------------------------
# Inference Settings
# -------------------------------------------------------------------

MAX_NEW_TOKENS = 200


# -------------------------------------------------------------------
# Visual Analysis Prompt
# -------------------------------------------------------------------

# Qwen2-VL uses a structured chat message format with typed content blocks.
# Unlike LLaVA's flat prompt string, each content element is a dict.

PROMPT = (
    "Analyze this disaster image in detail. "
    "What type of natural disaster is occurring? "
    "Describe the visible damage, the affected environment, "
    "and any observable impact on infrastructure or human settlements."
)


# -------------------------------------------------------------------
# Response Generation Function
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
    # Build chat message
    # ---------------------------------------------------------------

    # Qwen2-VL chat format: typed content blocks inside a user turn.
    # apply_chat_template converts this to a flat text string the
    # tokenizer can process, including the generation prompt marker.

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  PROMPT},
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )


    # ---------------------------------------------------------------
    # Decode output
    # ---------------------------------------------------------------

    # Trim input token IDs so only the newly generated tokens are
    # decoded — prevents the prompt from appearing in the response.

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


    # ---------------------------------------------------------------
    # Debug logging
    # ---------------------------------------------------------------

    print("\nQwen2-VL Response:")
    print(response)


    # ---------------------------------------------------------------
    # Return response
    # ---------------------------------------------------------------

    return {
        "model":    "Qwen-VL",
        "response": response,
    }


# -------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------

if __name__ == "__main__":

    image_path = "test.jpg"

    result = predict_response(image_path)

    print("\nScene Understanding Result")
    print("-------------------------")
    print(f"Model    : {result['model']}")
    print(f"Response : {result['response']}")
