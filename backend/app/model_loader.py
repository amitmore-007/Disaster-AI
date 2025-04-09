# backend/app/model_loader.py

from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch

MODEL_NAME = "nitarshan/satellite-disaster-classification"  # ✅ Final model

def load_model():
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    return model, processor
