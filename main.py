
import os
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    repo_id = os.getenv("HF_MODEL_REPO")
    if not repo_id:
        raise RuntimeError(
            "HF_MODEL_REPO environment variable is not configured."
        )

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="best.onnx",
        token=os.getenv("HF_TOKEN") or None,
    )

    model = YOLO(model_path, task="detect")
    yield
    model = None


app = FastAPI(
    title="Agritech Leaf Detection API",
    description="Detects and counts leaves in uploaded plant images.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "Agritech Leaf Detection API",
        "status": "online",
        "health": "/health",
        "count_endpoint": "/count",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "starting",
        "model_loaded": model is not None,
    }


@app.post("/count")
async def count_leaves(
    file: UploadFile = File(...),
    confidence: float = 0.25,
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The detection model is not ready.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The image is empty.")

    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="The image must be 15 MB or smaller.",
        )

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    try:
        results = model.predict(
            source=image,
            conf=confidence,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Leaf detection failed: {str(exc)}",
        )

    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ]

            detections.append({
                "class_id": class_id,
                "class_name": result.names.get(
                    class_id, str(class_id)
                ),
                "confidence": round(score, 4),
                "box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
            })

    return {
        "filename": file.filename,
        "leaf_count": len(detections),
        "confidence_threshold": confidence,
        "detections": detections,
    }
