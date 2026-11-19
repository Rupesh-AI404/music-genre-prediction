# Import the pandas library for data manipulation
import pandas as pd
# Import the function for splitting data into train and test sets
from sklearn.model_selection import train_test_split
# Import the StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
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

    # --- Fitting the StandardScaler (from previous task) ---
    print("\n--- Scaling Features ---")
    scaler = StandardScaler()
    scaler.fit(X_train)
    print("StandardScaler has been fitted to the training data.")

    # --- THIS IS THE NEW CODE BLOCK TO ADD ---

    # 3. Apply the fitted scaler to transform both the training and testing data.
    # The .transform() method uses the learned parameters (mean and std from X_train)
    # to scale the data.
    # The output is a NumPy array, not a pandas DataFrame.
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Verification Step ---
    # Let's check the properties of the newly scaled data to confirm the transformation.
    print("\nVerification of scaled data:")

    # The mean of each feature in the scaled training data should be very close to 0.
    print(f"Mean of first 5 features in X_train_scaled: {X_train_scaled[:, :5].mean(axis=0)}")

    # The standard deviation of each feature in the scaled training data should be very close to 1.
    print(f"Standard deviation of first 5 features in X_train_scaled: {X_train_scaled[:, :5].std(axis=0)}")

    # Note: The mean and std of X_test_scaled will NOT be exactly 0 and 1,
    # because it was scaled using the parameters from X_train. This is correct!
    print(f"\nMean of first 5 features in X_test_scaled: {X_test_scaled[:, :5].mean(axis=0)}")

    # --- END OF THE NEW CODE BLOCK ---

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")