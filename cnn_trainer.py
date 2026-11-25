# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- THIS IS THE NEW CODE BLOCK TO ADD ---

# Import TensorFlow and the Keras modules
# TensorFlow is the backend engine that powers Keras.
import tensorflow as tf

# Sequential is the model type we'll use, a linear stack of layers.
from tensorflow.keras.models import Sequential

# Here we import all the "bricks" or layer types we need to build our CNN.
from tensorflow.keras.layers import (
    Conv1D,               # The 1D convolutional layer
    MaxPooling1D,         # The 1D max pooling layer
    BatchNormalization,   # The layer for stabilizing learning
    Dropout,              # The layer for preventing overfitting
    Flatten,              # The layer to bridge conv blocks and dense layers
    Dense                 # The classic fully-connected neural network layer
)

# --- END OF THE NEW CODE BLOCK ---

# --- 1. Load and Prepare the Data ---
# (The rest of your script from the previous task remains the same)

# Define the path to our dataset
CSV_PATH = "features.csv"

# Load the dataset
try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    exit() # Exit the script if the data isn't there

# Separate features (X) and target (y)
X = features_df.drop('genre_label', axis=1)
y = features_df['genre_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape Data for CNN Input
print("\n--- Reshaping data for CNN model ---")
X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

print("\nShapes after reshaping for CNN:")
print(f"X_train_cnn shape: {X_train_cnn.shape}")
print(f"X_test_cnn shape: {X_test_cnn.shape}")