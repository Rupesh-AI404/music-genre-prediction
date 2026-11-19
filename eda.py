# Import the pandas library, the cornerstone of our data analysis toolkit.
# We use the conventional alias 'pd' for brevity and readability.
import pandas as pd

# Define a constant for the path to our dataset file.
# This makes the script cleaner and easier to modify if the file name changes.
CSV_PATH = "features.csv"

# It's a good practice to wrap file operations in a try-except block.
# This handles potential errors gracefully, such as if the file is not found.
try:
    # Use the pd.read_csv() function to load the dataset.
    # The function reads the file and returns a pandas DataFrame, which we
    # store in a variable, conventionally named 'df' or a more descriptive name.
    features_df = pd.read_csv(CSV_PATH)

    print("DataFrame loaded successfully!")

    # --- THIS IS THE NEW CODE BLOCK TO ADD ---

    # Get the structural summary of the DataFrame using .info()
    print("\n--- DataFrame Info ---")
    # We call the .info() method directly on our DataFrame.
    # It prints the summary to the console automatically.
    features_df.info()

    # Get the statistical summary of the DataFrame using .describe()
    print("\n--- DataFrame Statistical Summary ---")
    # We call the .describe() method, which returns a new DataFrame containing
    # the statistics. Printing this gives us a clean, tabular view.
    print(features_df.describe())

    # --- THIS IS THE NEW CODE BLOCK TO ADD ---

    # Explicitly check for missing values using .isnull() and .sum()
    print("\n--- Missing Values Check ---")
    missing_values_count = features_df.isnull().sum()

    print("Number of missing values per column:")
    print(missing_values_count)

    # Based on the check, we can make a statement.
    if missing_values_count.sum() == 0:
        print("\nConclusion: The dataset has no missing values. No handling is required.")
    else:
        print("\nConclusion: The dataset contains missing values. Handling is required.")
        # In a real-world scenario, the handling logic would go here.

    # --- END OF THE NEW CODE BLOCK ---

    # --- END OF THE NEW CODE BLOCK ---

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure you have run the 'feature_extractor.py' script first to generate the dataset.")
except Exception as e:
    print(f"An error occurred while loading the DataFrame: {e}")