# Import necessary libraries
import os  # To interact with the file system (navigate paths, list files)
import json  # To save our final data in a structured JSON format (we'll later switch to CSV)
import librosa  # The core library for audio analysis and feature extraction

# Define the path to the dataset
DATASET_PATH = "GTZAN music genre dataset/genres"

# Define the path where the processed data (features) will be saved
JSON_PATH = "music_genre_data.json"


def process_dataset(dataset_path, json_path):
    """
    The main function to extract features from the dataset and save them to a JSON file.

    This function will iterate through all genre sub-folders in the dataset, process
    each audio file, and extract relevant features.

    Args:
        dataset_path (str): The path to the root of the dataset directory.
        json_path (str): The path to the file where the extracted data will be saved.
    """

    # Dictionary to store all our data: mapping genres to a list of features
    data = {
        "mapping": [],  # List of genre names, e.g., ["blues", "classical", ...]
        "labels": [],  # The corresponding genre label (as an integer) for each track/segment
        "features": []  # The extracted features for each track/segment
    }

    # --- The main processing loop will be added here in the upcoming tasks ---
    print("Starting feature extraction...")

    # After the loop, we will save the data to the specified JSON file
    # with open(json_path, "w") as fp:
    #     json.dump(data, fp, indent=4)

    print("Feature extraction complete.")


# This is a standard Python construct. The code inside this block will only run
# when the script is executed directly (e.g., `python feature_extractor.py`).
# It will not run if the script is imported as a module into another script.
# This makes our code reusable and organized.
if __name__ == "__main__":
    process_dataset(DATASET_PATH, JSON_PATH)