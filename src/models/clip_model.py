import logging

from transformers import CLIPProcessor, CLIPModel
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

MODEL_PATH = "openai/clip-vit-base-patch32"


# -------------------------------------------------------------------
# Lazy Loading Setup
# -------------------------------------------------------------------

# CLIP is loaded on the first inference request and cached for all
# subsequent requests. This keeps server startup instant and matches
# the loading pattern used by BLIP-2, LLaVA, and Qwen.

_model:     CLIPModel     | None = None
_processor: CLIPProcessor | None = None
_lock = threading.Lock()


# -------------------------------------------------------------------
# Model Loader (lazy singleton with double-checked locking)
# -------------------------------------------------------------------

def _load_model():
    global _model, _processor

    if _model is None:
        with _lock:
            if _model is None:
                _processor = CLIPProcessor.from_pretrained(MODEL_PATH)
                _model     = CLIPModel.from_pretrained(MODEL_PATH).to(device)
                _model.eval()

    return _model, _processor


# -------------------------------------------------------------------
# Prompt Engineering
# -------------------------------------------------------------------

# Semantic natural-language prompts tuned for CLIP zero-shot matching.
#
# WHY: CLIP was trained on image-caption pairs, not short labels.
# Richer, scene-descriptive prompts match the training distribution
# more closely, increasing cosine similarity scores and raising
# confidence above the threshold needed for reliable classification.
#
# RULE: each prompt must uniquely describe the VISUAL APPEARANCE of
# that disaster class so CLIP can separate it from adjacent classes.
# Avoid vague shared words like "damage" or "disaster" alone — every
# prompt should have at least one distinctive visual anchor.
#
# ALIGNMENT: these 5 classes exactly match the filtered evaluation dataset
# produced by scripts/create_filtered_dataset.py. Adding or removing a
# prompt requires a matching change to display_labels and the filtered
# dataset folder structure.

prompts = [
    "an image showing earthquake damage with collapsed buildings and rubble",
    "an image showing damaged infrastructure such as roads, bridges, or buildings",
    "an image showing injured or affected people after a disaster",
    "an image of a wildfire burning vegetation and forests",
    "an image of an urban fire affecting buildings and city areas",
    "an image showing flood or water disaster with submerged areas",
    "an image showing drought conditions with dry cracked land",
    "an image showing a landslide with collapsed terrain and debris",
    "an image of a forest without disaster",
    "an image of buildings and streets without disaster",
    "an image of the sea or ocean",
    "an image showing people in normal conditions"
]

display_labels = [
    "Earthquake",
    "Infrastructure Damage",
    "Human Damage",
    "Wild Fire",
    "Urban Fire",
    "Water Disaster",
    "Drought",
    "Landslide",
    "Forest",
    "Buildings and Street",
    "Sea",
    "Human"
]

# -------------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------------

def predict_disaster(image_path):

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
    # Process image + prompts
    # ---------------------------------------------------------------

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}


    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------

    with torch.no_grad():
        outputs = model(**inputs)


    # ---------------------------------------------------------------
    # Similarity scores
    # ---------------------------------------------------------------

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)


    # ---------------------------------------------------------------
    # Top-3 predictions
    # ---------------------------------------------------------------

    k = min(3, len(display_labels))
    top_k = probs[0].topk(k)
    top3_indices = top_k.indices.tolist()
    top3_values  = top_k.values.tolist()

    predicted_index  = top3_indices[0]
    prediction       = display_labels[predicted_index]
    confidence_score = round(top3_values[0] * 100, 2)

    top_3_predictions = [
        {"label": display_labels[i], "score": round(v * 100, 2)}
        for i, v in zip(top3_indices, top3_values)
    ]


    # ---------------------------------------------------------------
    # Debug Logging
    # ---------------------------------------------------------------

    logger.debug("Prediction Scores:")
    for i, label in enumerate(display_labels):
        logger.debug(f"{label}: {probs[0][i].item() * 100:.2f}%")


    # ---------------------------------------------------------------
    # Return structured metrics
    # ---------------------------------------------------------------

    return {
        "model": "CLIP",
        "metrics": {
            "disaster_type":      prediction,
            "confidence_score":   confidence_score,
            "confidence_level":   confidence_to_level(confidence_score),
            "top_3_predictions":  top_3_predictions,
        }
    }


# -------------------------------------------------------------------
# Image Embedding (used by retrieval module)
# -------------------------------------------------------------------

def embed_image(image_path) -> "np.ndarray":
    """
    Return a unit-norm CLIP image embedding (512-d float32 numpy array).

    Reuses the same model singleton as predict_disaster — no second load.
    Vectors are L2-normalised so dot-product == cosine similarity.
    """
    import numpy as np

    model, processor = _load_model()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid image file: {exc}")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(pixel_values=inputs["pixel_values"])

    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy().astype("float32")


# -------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    image_path = "test.jpg"

    result = predict_disaster(image_path)

    logger.info("\nPrediction Result")
    logger.info("-------------------------")
    logger.info(f"Model      : {result['model']}")
    logger.info(f"Disaster   : {result['metrics']['disaster_type']}")
    logger.info(f"Confidence : {result['metrics']['confidence_score']}%")
    logger.info(f"Level      : {result['metrics']['confidence_level']}")
    logger.info(f"Top-3      : {result['metrics']['top_3_predictions']}")
