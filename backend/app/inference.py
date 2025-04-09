# backend/app/inference.py

from PIL import Image
import torch
from io import BytesIO

def run_inference(image_bytes, model, processor):
    # Convert image bytes to PIL Image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predicted class
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    return {
        "predicted_label": predicted_label,
        "confidence": torch.softmax(logits, dim=-1).max().item()
    }
