import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

print("Loading datasets...")
X_normal = np.load("X_train_normal.npy")[:5000] # Just grab 5000 for a quick test
X_abnormal = np.load("X_abnormal.npy")[:5000]

# Expand dims and normalize (using the SAME logic as training)
X_normal = np.expand_dims(X_normal, -1)
X_abnormal = np.expand_dims(X_abnormal, -1)

norm_max = np.max(np.abs(X_normal))
X_normal = X_normal / norm_max
X_abnormal = X_abnormal / norm_max

print("Loading trained VAE models...")
# We load the encoder and decoder to reconstruct the data
encoder = load_model("vae_encoder.h5", compile=False)
decoder = load_model("vae_decoder.h5", compile=False)

def get_reconstruction_error(data):
    # Pass through encoder to get latent space (index 0 is z_mean)
    z = encoder.predict(data, batch_size=128)[0]
    # Pass latent space through decoder to rebuild the heartbeat
    reconstruction = decoder.predict(z, batch_size=128)
    # Calculate Mean Squared Error across the 350 points
    mse = np.mean(np.square(data - reconstruction), axis=1)
    return mse

print("Calculating errors for Normal hearts...")
normal_errors = get_reconstruction_error(X_normal)

print("Calculating errors for Abnormal hearts...")
abnormal_errors = get_reconstruction_error(X_abnormal)

# --- VISUALIZATION ---
plt.figure(figsize=(10, 6))
plt.hist(normal_errors, bins=50, alpha=0.6, color='blue', label='Normal (Healthy)')
plt.hist(abnormal_errors, bins=50, alpha=0.6, color='red', label='Abnormal (Sick)')

plt.title("VAE Anomaly Detection: Reconstruction Error")
plt.xlabel("Mean Squared Error (MSE)")
plt.ylabel("Number of Heartbeats")
plt.axvline(x=np.mean(normal_errors) + 2*np.std(normal_errors), color='black', linestyle='dashed', linewidth=2, label='Suggested Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()