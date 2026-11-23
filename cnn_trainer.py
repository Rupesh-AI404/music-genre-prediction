# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Load and Prepare the Data ---

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
# We use the same random_state to ensure the split is identical to the one
# used for our traditional models, allowing for a fair comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
# Fit on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preparation complete. Shapes before reshaping:")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

# --- 2. Reshape Data for CNN Input ---
# This is the core of the current task.

print("\n--- Reshaping data for CNN model ---")

# A 1D CNN expects a 3D array as input: (num_samples, num_features, num_channels)
# Our data has num_samples and num_features, we need to add the 'channels' dimension.
# For our data, we have only one channel.
# We can use numpy.expand_dims to add an extra dimension at the end (axis=-1).

X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

# --- Verification Step ---
# Let's check the new shapes to confirm the operation was successful.
print("\nShapes after reshaping for CNN:")
print(f"X_train_cnn shape: {X_train_cnn.shape}") # Expected: (7500, 28, 1)
print(f"X_test_cnn shape: {X_test_cnn.shape}")   # Expected: (2500, 28, 1)

# The target variables (y_train, y_test) are pandas Series of integers.
# This format is already perfect for use with TensorFlow/Keras, especially
# with the 'sparse_categorical_crossentropy' loss function, so no changes are needed.
print(f"\ny_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")