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