# Import the pandas library for data manipulation
import pandas as pd
# Import the function for splitting data into train and test sets
from sklearn.model_selection import train_test_split
# Import the StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
# Import the Logistic Regression model
from sklearn.linear_model import LogisticRegression
# Import LabelEncoder and numpy for previous steps
from sklearn.preprocessing import LabelEncoder
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
    # (code omitted for brevity)

    # --- Splitting Data (from previous task) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Scaling Features (from previous tasks) ---
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nFeatures have been scaled.")

    # --- THIS IS THE NEW CODE BLOCK TO ADD ---

    # Initialize and train a Logistic Regression model
    print("\n--- Training Logistic Regression Model ---")

    # 1. Initialize the model
    # We set 'max_iter' to a higher value to ensure the model has enough iterations
    # to converge on a solution, which helps avoid convergence warnings.
    log_reg = LogisticRegression(max_iter=1000)

    # 2. Train the model using the .fit() method
    # The model learns the relationship between our scaled training features (X_train_scaled)
    # and the corresponding genre labels (y_train).
    log_reg.fit(X_train_scaled, y_train)

    # --- Verification Step ---
    print("Logistic Regression model trained successfully!")

    # The trained model now contains learned parameters. For example, we can see the 10
    # classes it has learned to identify.
    print(f"Model learned the following classes: {log_reg.classes_}")

    # --- END OF THE NEW CODE BLOCK ---

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")