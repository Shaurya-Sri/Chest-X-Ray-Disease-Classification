import io
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Model Definition 
class XRayModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=20, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        in_features   = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# Globals 
MODEL_PATH = Path(__file__).parent.parent / "model.pkl"
DEVICE     = torch.device("cpu")  

_model   = None
_config  = None
_tfm     = None


def load_model():
    global _model, _config, _tfm
    if _model is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    _config = bundle["config"]

    _model = XRayModel(
        model_name  = _config["model_name"],
        num_classes = _config["num_classes"],
        dropout     = _config["dropout"],
    )
    _model.load_state_dict(bundle["model_state"])
    _model.to(DEVICE)
    _model.eval()

    _tfm = A.Compose([
        A.Resize(_config["img_size"], _config["img_size"]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


#  FastAPI App
app = FastAPI(
    title="Chest X-Ray Pathology Classifier",
    description="Detects 20 thoracic pathologies from chest X-ray images.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    load_model()


class PredictionResponse(BaseModel):
    predictions: Dict[str, int]
    probabilities: Dict[str, float]
    top_findings: List[str]
    threshold_used: float


@app.get("/")
def root():
    return {"message": "Chest X-Ray Classifier API is running. POST /predict to classify."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/classes")
def get_classes():
    load_model()
    return {"classes": _config["classes"], "num_classes": _config["num_classes"]}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    load_model()

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)

        aug = _tfm(image=img_np)
        tensor = aug["image"].float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = _model(tensor)
            probs  = torch.sigmoid(logits).cpu().numpy()[0]

        threshold = _config["threshold"]
        classes   = _config["classes"]

        preds = (probs >= threshold).astype(int)
        
        if preds.sum() == 0:
            preds[probs.argmax()] = 1

        predictions   = {cls: int(preds[i])   for i, cls in enumerate(classes)}
        probabilities = {cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)}
        top_findings  = [cls for cls, val in predictions.items() if val == 1]

        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            top_findings=top_findings,
            threshold_used=threshold,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict on multiple images at once."""
    load_model()
    results = []
    for file in files:
        contents = await file.read()
        img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        aug = _tfm(image=img)
        tensor = aug["image"].float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.sigmoid(_model(tensor)).cpu().numpy()[0]

        threshold = _config["threshold"]
        preds     = (probs >= threshold).astype(int)
        if preds.sum() == 0:
            preds[probs.argmax()] = 1

        results.append({
            "filename"    : file.filename,
            "top_findings": [c for c, p in zip(_config["classes"], preds) if p == 1],
            "probabilities": {c: round(float(p), 4) for c, p in zip(_config["classes"], probs)},
        })

    return {"results": results}
