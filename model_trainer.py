# Import the pandas library for data manipulation
import pandas as pd

# Define the path to our dataset
CSV_PATH = "features.csv"

# Load the dataset from the CSV file into a pandas DataFrame
try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")

    # --- THIS IS THE CORE OF THE CURRENT TASK ---

    # Separate the features (X) from the target label (y)

    # To create our features set (X), we drop the 'genre_label' column from the DataFrame.
    # The `axis=1` argument specifies that we are dropping a column, not a row.
    X = features_df.drop('genre_label', axis=1)

    # To create our target set (y), we select only the 'genre_label' column.
    # The result is a pandas Series containing the labels for each sample.
    y = features_df['genre_label']

    # --- Verification Step ---
    # It's always a good practice to check the shapes of our new objects to ensure
    # the separation was successful.
    print("\nVerifying the separation:")
    print(f"Shape of features (X): {X.shape}")
    print(f"Shape of target (y): {y.shape}")

    print("\nFirst 5 rows of features (X):")
    print(X.head())

    print("\nFirst 5 values of target (y):")
    print(y.head())

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")