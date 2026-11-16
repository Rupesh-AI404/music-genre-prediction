# Import necessary libraries
import os  # To interact with the file system (navigate paths, list files)
import json  # To save our final data in a structured JSON format (we'll later switch to CSV)
import librosa  # The core library for audio analysis and feature extraction

# Define the path to the dataset
DATASET_PATH = "GTZAN music genre dataset/genres"

# Define the path where the processed data (features) will be saved
JSON_PATH = "data.json"

# Define constants for audio processing
SAMPLE_RATE = 22050  # The standard sample rate for librosa's default analysis
TRACK_DURATION_SECONDS = 30  # All tracks in GTZAN are 30 seconds long
NUM_SEGMENTS = 10  # We will divide each track into 10 segments

# Calculate the number of audio samples we expect per 30-second track
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION_SECONDS


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

    print("Starting feature extraction...")

    # --- THIS IS THE NEW CODE BLOCK TO ADD ---
    # Iterate through all the genre folders in the dataset path
    # We use sorted() to ensure a consistent order every time we run the script.
    for i, genre_folder in enumerate(sorted(os.listdir(dataset_path))):

        # Construct the full path to the current genre folder
        genre_path = os.path.join(dataset_path, genre_folder)

        # We only want to process directories, so we check if the path is a directory.
        # This is a robust way to skip any stray files (like .DS_Store on macOS)
        if os.path.isdir(genre_path):
            # Store the genre name in our mapping list.
            # This is crucial for label encoding later on.
            data["mapping"].append(genre_folder)

            print(f"\nProcessing genre: {genre_folder}")

            # --- A nested loop to process each .wav file will go here in the next task ---

    # --- END OF THE NEW CODE BLOCK ---

    # After the loop, we will save the data to the specified JSON file
    # with open(json_path, "w") as fp:
    #     json.dump(data, fp, indent=4)

    print("\nFeature extraction complete.")


# This is a standard Python construct. The code inside this block will only run
# when the script is executed directly (e.g., `python feature_extractor.py`).
# It will not run if the script is imported as a module into another script.
# This makes our code reusable and organized.
if __name__ == "__main__":
    process_dataset(DATASET_PATH, JSON_PATH)