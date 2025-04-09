from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import os

# Load SatMAE model & processor from Hugging Face
MODEL_NAME = "MIT/satmae-vit-base"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def analyze_satellite_image(image_path: str):
    """
    Run SatMAE model on a satellite image and return raw features/embeddings.
    """
    if not os.path.exists(image_path):
        return {"error": "Image path not found."}
    
    # Open image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    
    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Return raw features
    last_hidden_state = outputs.last_hidden_state
    return last_hidden_state.squeeze(0).numpy().tolist()  # Convert to list for JSON
