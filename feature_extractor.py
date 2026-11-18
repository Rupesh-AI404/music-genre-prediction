import os
import json
import librosa
import numpy as np

# Audio processing constants
SAMPLE_RATE = 22050
TRACK_DURATION_SECONDS = 30
NUM_SEGMENTS = 10
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION_SECONDS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try common dataset folder names
DATASET_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "genres_original"),
    os.path.join(SCRIPT_DIR, "GTZAN music genre dataset", "genres"),
]

JSON_PATH = os.path.join(SCRIPT_DIR, "data.json")
VALID_EXTENSIONS = (".wav", ".au")


def resolve_dataset_path():
    for path in DATASET_CANDIDATES:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Dataset folder not found.\n"
        "Expected one of:\n"
        + "\n".join(f"- {p}" for p in DATASET_CANDIDATES)
    )


def process_dataset(dataset_path, json_path):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

    for i, genre_folder in enumerate(sorted(os.listdir(dataset_path))):
        genre_path = os.path.join(dataset_path, genre_folder)

        if not os.path.isdir(genre_path):
            continue

        data["mapping"].append(genre_folder)
        print(f"\nProcessing: {genre_folder}")

        for filename in sorted(os.listdir(genre_path)):
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue

            file_path = os.path.join(genre_path, filename)

            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            except Exception as e:
                print(f"Could not load {file_path}: {e}")
                continue

            if len(signal) < SAMPLES_PER_TRACK:
                print(f"Skipping short file: {file_path}")
                continue

            # Use exactly 30s
            signal = signal[:SAMPLES_PER_TRACK]

            for s in range(NUM_SEGMENTS):
                start_sample = s * num_samples_per_segment
                end_sample = start_sample + num_samples_per_segment
                segment = signal[start_sample:end_sample]

                mfcc = librosa.feature.mfcc(
                    y=segment,
                    sr=sr,
                    n_mfcc=NUM_MFCC,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH
                )

                # (NUM_MFCC, frames) -> (NUM_MFCC,)
                mfcc_processed = np.mean(mfcc, axis=1)

                data["mfcc"].append(mfcc_processed.tolist())
                data["labels"].append(i)

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4)

    print(f"\nFeature extraction complete. Saved to: {json_path}")
    print(f"Total samples: {len(data['mfcc'])}")


if __name__ == "__main__":
    dataset_path = resolve_dataset_path()
    print(f"Using dataset path: {dataset_path}")
    process_dataset(dataset_path, JSON_PATH)