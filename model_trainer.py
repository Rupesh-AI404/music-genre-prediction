# Import the pandas library for data manipulation
import pandas as pd
# Import the function for splitting data into train and test sets
from sklearn.model_selection import train_test_split
# Import the StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
# Import our models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# Import the accuracy metric for evaluation
from sklearn.metrics import accuracy_score
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

    # --- Training Models (from previous tasks, code omitted for brevity) ---
    # Logistic Regression
    print("\n--- Training Logistic Regression Model ---")
    log_reg = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
    print("Logistic Regression model trained successfully!")
    # SVM
    print("\n--- Training Support Vector Machine (SVM) Model ---")
    svm_model = SVC(probability=True, random_state=42).fit(X_train_scaled, y_train)
    print("Support Vector Machine model trained successfully!")
    # Random Forest
    print("\n--- Training Random Forest Classifier Model ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)
    print("Random Forest Classifier model trained successfully!")

    # --- THIS IS THE NEW CODE BLOCK TO ADD ---

    # Make predictions on the test set and calculate accuracy for each model
    print("\n--- Evaluating Models on the Test Set ---")

    # 1. Logistic Regression
    # Use the trained model to predict labels for the unseen test data
    y_pred_log_reg = log_reg.predict(X_test_scaled)
    # Compare the predicted labels with the true labels to calculate accuracy
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    print(f"Logistic Regression Accuracy: {accuracy_log_reg * 100:.2f}%")

    # 2. Support Vector Machine
    y_pred_svm = svm_model.predict(X_test_scaled)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Support Vector Machine Accuracy: {accuracy_svm * 100:.2f}%")

    # 3. Random Forest Classifier
    # An alternative and more concise way to get accuracy is the .score() method,
    # which performs prediction and scoring in one step.
    accuracy_rf = rf_model.score(X_test_scaled, y_test)
    print(f"Random Forest Classifier Accuracy: {accuracy_rf * 100:.2f}%")

    # --- END OF THE NEW CODE BLOCK ---

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")