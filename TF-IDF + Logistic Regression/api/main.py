import re
import unicodedata
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

# 1. Preprocessing function (same as training)
def preprocess_bangla_text(text: str) -> str:
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove emojis
    emoji_pattern = re.compile("["  
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F700-\U0001F77F"
                               u"\U0001F780-\U0001F7FF"
                               u"\U0001F800-\U0001F8FF"
                               u"\U0001F900-\U0001F9FF"
                               u"\U0001FA00-\U0001FA6F"
                               u"\U0001FA70-\U0001FAFF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # Remove special chars
    text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = unicodedata.normalize('NFC', text)
    return text

# 2. Load ONNX model
MODEL_PATH = "models/LogisticRegression_ONNX_Model.onnx"
try:
    session = ort.InferenceSession(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model: {e}")

# 3. Label mapping
label_map = {0: 'not bully', 1: 'religious', 2: 'sexual', 3: 'troll', 4: 'threat'}

# 4. FastAPI App
app = FastAPI(
title="Bangla Sentiment Classifier using TF-IDF + Logistic Regression",
version="1.0",
description="Provides sentiment prediction for Bangla comments."
)

@app.get("/", tags=["Root"])
def root():
    return {"message": "Welcome to Technoheven Bangla Sentiment Classifier API", "docs": "/docs"}

class CommentRequest(BaseModel):
    comment: str

class PredictionResponse(BaseModel):
    label: str
    label_id: int

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: CommentRequest):
    try:
        # Preprocess the Bangla text exactly as done during training
        clean_text = preprocess_bangla_text(request.comment)
        
        # ONNX requires input shape to be 2D: [[sentence]]
        ort_inputs = {session.get_inputs()[0].name: [[clean_text]]}
        ort_outs = session.run(None, ort_inputs)
        pred_id = int(ort_outs[0][0])
        pred_label = label_map.get(pred_id, "unknown")

        return PredictionResponse(label=pred_label, label_id=pred_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
