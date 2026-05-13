from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# 1. Define your Secret API Key (Give this to your friend!)
API_KEY_CREDIT = "sumit_ecg_secure_access_2026"

app = FastAPI(title="Explainable ECG Diagnostic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
encoder = tf.keras.models.load_model("tcvae_encoder.keras", compile=False)
classifier = joblib.load("knn_ecg_classifier.pkl")

@app.get("/")
def home():
    return {"status": "online", "message": "Secure ECG Server is Running"}

@app.post("/diagnose")
async def diagnose_ecg(
    file: UploadFile = File(...), 
    x_api_key: str = Header(None) # This looks for 'X-API-Key' in the header
):
    # SECURITY CHECK: If the key is missing or wrong, block the request
    if x_api_key != API_KEY_CREDIT:
        raise HTTPException(status_code=401, detail="Invalid or Missing API Key")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), header=None) 
        
        raw_signal = df.values.flatten()[:500]
        raw_signal_expanded = np.expand_dims(raw_signal, axis=(0, -1))
        
        latent_features = encoder.predict(raw_signal_expanded)[0]
        prediction = classifier.predict([latent_features])[0]
        
        return {
            "status": "success",
            "diagnosis": "Abnormal" if prediction == 1 else "Normal",
            "latent_representation": latent_features.tolist()
        }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) # 0.0.0.0 allows others on your WiFi to connect