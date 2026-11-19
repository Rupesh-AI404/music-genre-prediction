# Import the pandas library for data manipulation
import pandas as pd
# Import LabelEncoder from scikit-learn for educational purposes
from sklearn.preprocessing import LabelEncoder
# Import the function for splitting data into train and test sets
from sklearn.model_selection import train_test_split
# Import the numpy library to check data types
import numpy as np

# Define the path to our dataset
CSV_PATH = "features.csv"

# Load the dataset from the CSV file into a pandas DataFrame
try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")

    # Separate the features (X) from the target label (y)
    X = features_df.drop('genre_label', axis=1)
    y = features_df['genre_label']

    # --- Label Encoding Verification (from previous task) ---
    print("\n--- Label Encoding Step ---")
    if np.issubdtype(y.dtype, np.integer):
        print("Labels are already numerically encoded. No action needed.")
    else:
        print("Labels are not numerical. Applying LabelEncoder.")
        # ... (commented-out LabelEncoder code)

    # --- THIS IS THE NEW CODE BLOCK TO ADD ---

    # Split the data into training and testing sets
    print("\n--- Splitting Data into Training and Testing Sets ---")

    # We use train_test_split to partition our data.
    # test_size=0.25 means 25% of the data will be used for testing.
    # random_state=42 ensures that the split is the same every time we run the script,
    # which is crucial for reproducibility.
    # stratify=y ensures that the class distribution in the train and test sets
    # is the same as the original dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Verification Step ---
    print("\nVerifying the shapes of the new sets:")
    print(f"Shape of X_train: {X_train.shape}")  # Should be (7500, 28)
    print(f"Shape of X_test: {X_test.shape}")  # Should be (2500, 28)
    print(f"Shape of y_train: {y_train.shape}")  # Should be (7500,)
    print(f"Shape of y_test: {y_test.shape}")  # Should be (2500,)

    # --- END OF THE NEW CODE BLOCK ---

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")