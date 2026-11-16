# Import necessary libraries
import os
import json
import librosa
import numpy as np  # Add this import

# Define the path to the dataset
DATASET_PATH = "GTZAN music genre dataset/genres"

# Define the path where the processed data (features) will be saved
JSON_PATH = "data.json"

# Define constants for audio processing
SAMPLE_RATE = 22050
TRACK_DURATION_SECONDS = 30
NUM_SEGMENTS = 10

# --- ADD THIS NEW BLOCK OF CONSTANTS ---
# Define constants for MFCC extraction
NUM_MFCC = 13  # Number of MFCC coefficients to extract
N_FFT = 2048  # Window size for FFT
HOP_LENGTH = 512  # Number of samples to slide the FFT window
# ----------------------------------------

# Calculate the number of audio samples we expect per 30-second track
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION_SECONDS


def process_dataset(dataset_path, json_path):
    # ... (the data dictionary remains the same)

    for i, genre_folder in enumerate(sorted(os.listdir(dataset_path))):
        # ... (the outer loop remains the same)
        if os.path.isdir(DATASET_PATH):
            # ... (the inner loop starts)
            for filename in sorted(os.listdir(DATASET_PATH)):
                if filename.endswith(".wav"):
                    file_path = os.path.join(DATASET_PATH, filename)

                    try:
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                        if len(signal) >= SAMPLES_PER_TRACK:
                            num_samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

                            for s in range(NUM_SEGMENTS):
                                start_sample = s * num_samples_per_segment
                                end_sample = start_sample + num_samples_per_segment
                                segment = signal[start_sample:end_sample]

                                # --- THIS IS THE NEW CODE BLOCK TO ADD ---
                                # Extract MFCCs from the segment.
                                # The result 'mfccs' is a 2D NumPy array.
                                mfccs = librosa.feature.mfcc(y=segment,
                                                             sr=sr,
                                                             n_mfcc=NUM_MFCC,
                                                             n_fft=N_FFT,
                                                             hop_length=HOP_LENGTH)

                                # To get a single feature vector that summarizes the segment,
                                # we aggregate the MFCCs over the time frames by taking the mean.
                                # The shape of mfccs is (NUM_MFCC, num_frames), so we take the mean
                                # along axis=1 (the time axis).
                                mfccs_processed = np.mean(mfccs, axis=1)

                                # In later steps, we will combine this 'mfccs_processed' vector
                                # with other features and store it.
                                # --- END OF THE NEW CODE BLOCK ---

                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        continue
    # ... (rest of the code)

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    process_dataset(DATASET_PATH, JSON_PATH)