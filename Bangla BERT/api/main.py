from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import unicodedata
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Preprocessing function
# Data Preprocessing
def preprocess_bangla_text(text):
    """
    Preprocess Bangla text by removing special symbols, emojis, URLs, and extra spaces.
    Apply Unicode normalization.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove emojis (basic implementation)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove special characters (keep Bangla characters and basic punctuation)
    text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Apply Unicode normalization (NFC for composed characters)
    text = unicodedata.normalize('NFC', text)
    
    return text

# Load tokenizer and ONNX model
TOKENIZER_NAME = 'sagorsarker/bangla-bert-base'
tf = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

MODEL_PATH = 'models/BanglaBERT_ONNX.onnx'
session = ort.InferenceSession(MODEL_PATH)

# Label mapping
label_map = {0: 'not bully', 1: 'religious', 2: 'sexual', 3: 'troll', 4: 'threat'}

# FastAPI app
app = FastAPI(
title="Bangla Sentiment Classifier using Bengla BERT",
version="1.0",
description="Provides sentiment prediction for Bangla comments."
)

@app.get("/", tags=["Root"] )
def read_root():
    return {"message": "Welcome to Bangla Sentiment Classifier using Bengla BERT API", "docs": "/docs"}

class PredictRequest(BaseModel):
    comment: str

class PredictResponse(BaseModel):
    label: str
    label_id: int

@app.get("/", tags=["Root"] )
def read_root():
    return {"message": "Bangla BERT Sentiment Classifier. POST to /predict."}

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"] )
async def predict(request: PredictRequest):
    try:
        text = request.comment
        if not text:
            raise HTTPException(status_code=400, detail="Empty comment")

        # Preprocess
        processed = preprocess_bangla_text(text)

        # Tokenize
        enc = tf(processed, truncation=True, padding='max_length', max_length=128, return_tensors='np')
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']

        # ONNX inference
        ort_inputs = {
            session.get_inputs()[0].name: input_ids.astype(np.int64),
            session.get_inputs()[1].name: attention_mask.astype(np.int64)
        }
        ort_outs = session.run(None, ort_inputs)
        logits = ort_outs[0]
        pred_id = int(np.argmax(logits, axis=1)[0])
        label = label_map.get(pred_id, "unknown")

        return PredictResponse(label=label, label_id=pred_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
