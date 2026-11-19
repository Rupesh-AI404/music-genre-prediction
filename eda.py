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

    # --- Verification Step ---
    # To confirm that the data has been loaded correctly, let's print a success
    # message and display the first few rows of the DataFrame.
    print("DataFrame loaded successfully!")

    # The .head() method is an essential first check. By default, it displays
    # the first 5 rows of the DataFrame, giving us an immediate snapshot
    # of our data's structure and content.
    print("\nFirst 5 rows of the dataset:")
    print(features_df.head())

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure you have run the 'feature_extractor.py' script first to generate the dataset.")
except Exception as e:
    print(f"An error occurred while loading the DataFrame: {e}")