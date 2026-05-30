from transformers import CLIPProcessor, CLIPModel
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
    Path(__file__).resolve().parent.parent.parent / "models" / "clip-vit-base-patch32"
)


# -------------------------------------------------------------------
# Load CLIP Model
# -------------------------------------------------------------------

model = CLIPModel.from_pretrained(MODEL_PATH).to(device)

processor = CLIPProcessor.from_pretrained(MODEL_PATH)


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
    "a disaster image of severe flooding with submerged roads, vehicles, and buildings",
    "a disaster image of wildfire and fire damage with visible flames and thick smoke",
    "a disaster image of earthquake destruction with collapsed buildings and rubble",
    "a disaster image of landslide and collapsed terrain with debris covering roads",
    "a disaster image of cyclone damage with uprooted trees and destroyed infrastructure",
]

# display_labels must stay 1-to-1 aligned with prompts — API response values.
display_labels = [
    "Flood",
    "Fire",
    "Earthquake",
    "Landslide",
    "Cyclone",
]


# -------------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------------

def predict_disaster(image_path):

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
    # Best prediction
    # ---------------------------------------------------------------

    predicted_index = probs.argmax().item()

    prediction = display_labels[predicted_index]

    confidence = probs[0][predicted_index].item()


    # ---------------------------------------------------------------
    # Debug Logging
    # ---------------------------------------------------------------

    print("\nPrediction Scores:")

    for i, label in enumerate(display_labels):
        print(
            f"{label}: "
            f"{probs[0][i].item() * 100:.2f}%"
        )


    # ---------------------------------------------------------------
    # Return JSON-compatible response
    # ---------------------------------------------------------------

    return {
        "model": "CLIP",
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    }


# -------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------

if __name__ == "__main__":

    image_path = "test.jpg"

    result = predict_disaster(image_path)

    print("\nPrediction Result")
    print("-------------------------")
    print(f"Model      : {result['model']}")
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']}%")