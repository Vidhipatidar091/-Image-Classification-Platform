import os
import logging
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenet_model.keras"
)

# Model loading is temporarily disabled
model = None
model_error = "Model loading temporarily disabled."

logging.basicConfig(level=logging.INFO)


# ==========================================
# APPLICATION STARTUP
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Image Classification API...", flush=True)
    print("Skipping model loading temporarily.", flush=True)

    yield

    print("API shutting down.", flush=True)


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Image Classification API",
    description="Image Classification Project API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Image Classification API is running!",
        "documentation": "/docs"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "api_status": "running",
        "model_loaded": model is not None,
        "model_message": model_error
    }


# ==========================================
# MODEL INFORMATION
# ==========================================

@app.get("/model-info")
def model_info():
    return {
        "model_path": MODEL_PATH,
        "model_file_exists": os.path.isfile(MODEL_PATH),
        "model_loaded": model is not None,
        "message": model_error
    }


# ==========================================
# IMAGE PREDICTION
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Model loading is temporarily disabled.
    # Real predictions will work after the model is fixed.

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction is temporarily unavailable. "
                "The classification model has not been loaded."
            )
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:
        image_bytes = await file.read()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        image = image.resize((96, 96))

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        predictions = model.predict(
            image_array,
            verbose=0
        )

        scores = np.asarray(predictions)[0]

        predicted_class = int(np.argmax(scores))
        confidence = float(scores[predicted_class])

        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }

    except HTTPException:
        raise

    except Exception as exc:
        logging.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )