# Import the pandas library for data manipulation
import pandas as pd
# Import the function for splitting data into train and test sets
from sklearn.model_selection import train_test_split
# Import the StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
# Import the Logistic Regression model
from sklearn.linear_model import LogisticRegression
# Import the Support Vector Classifier model
from sklearn.svm import SVC
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

    # --- Label Encoding and Splitting Data (code omitted for brevity) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Scaling Features (code omitted for brevity) ---
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nFeatures have been scaled.")

    # --- Training Logistic Regression Model (from previous task) ---
    print("\n--- Training Logistic Regression Model ---")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    print("Logistic Regression model trained successfully!")

    # --- THIS IS THE NEW CODE BLOCK TO ADD ---

    # Initialize and train a Support Vector Machine (SVM) classifier
    print("\n--- Training Support Vector Machine (SVM) Model ---")

    # 1. Initialize the model
    # We use the RBF kernel, which is effective for non-linear relationships.
    # C=1.0 is the regularization parameter, a good default starting point.
    # random_state=42 ensures reproducibility.
    # probability=True is needed to enable probability estimates for later evaluation.
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)

    # 2. Train the model using the .fit() method
    # The model learns the optimal hyperplane to separate the genres based on
    # our scaled training features and their corresponding labels.
    svm_model.fit(X_train_scaled, y_train)

    # --- Verification Step ---
    print("Support Vector Machine model trained successfully!")
    print(f"Model learned the following classes: {svm_model.classes_}")

    # --- END OF THE NEW CODE BLOCK ---

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")