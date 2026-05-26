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

prompts = [
    "flood disaster",
    "wildfire disaster",
    "earthquake destruction",
    "landslide disaster",
    "cyclone damage",
    "tsunami disaster"
]

display_labels = [
    "Flood",
    "Wildfire",
    "Earthquake",
    "Landslide",
    "Cyclone",
    "Tsunami"
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