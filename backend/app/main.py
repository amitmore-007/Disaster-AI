# backend/app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model_loader import load_model
from inference import run_inference

app = FastAPI()

# Allow CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model, processor = load_model()

@app.get("/")
def root():
    return {"message": "Disaster Detection API is running."}

@app.post("/detect-disaster")
async def detect_disaster(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = run_inference(image_bytes, model, processor)
        return {
            "status": "success",
            "prediction": result["predicted_label"],
            "confidence": round(result["confidence"], 4)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
