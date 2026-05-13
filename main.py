import os
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
API_KEY_CREDIT = os.getenv("ECG_API_KEY")

# --- CRITICAL: REDEFINE SAMPLING FOR KERAS LOAD ---
@tf.keras.utils.register_keras_serializable(name="sampling")
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Initialize the Web App
app = FastAPI(title="Explainable ECG Diagnostic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models (Keras now knows what 'sampling' means)
print("Loading TC-VAE Encoder...")
encoder = tf.keras.models.load_model("tcvae_encoder.keras", compile=False)

print("Loading kNN Classifier...")
classifier = joblib.load("knn_ecg_classifier.pkl")

@app.get("/")
def home():
    return {"status": "online", "message": "Secure ECG Server is Running"}

@app.post("/diagnose")
async def diagnose_ecg(
    file: UploadFile = File(...), 
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY_CREDIT:
        raise HTTPException(status_code=401, detail="Invalid or Missing API Key")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), header=None) 
        
        # Slicing the first 500 points (1 second at 500Hz)
        raw_signal = df.values.flatten()[:500]
        
        # Reshape for model input: (Batch, Timesteps, Channels) -> (1, 500, 1)
        raw_signal_expanded = np.expand_dims(raw_signal, axis=(0, -1))
        
        # Get the 8 explainable latent numbers
        # Note: [0] gets the z_mean which we use for classification
        latent_features = encoder.predict(raw_signal_expanded)[0]
        
        # Get prediction from the kNN classifier
        prediction = classifier.predict(latent_features.reshape(1, -1))[0]
        
        return {
            "status": "success",
            "diagnosis": "Abnormal" if prediction == 1 else "Normal",
            "latent_representation": latent_features.tolist()
        }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)