# Import the pandas library, the cornerstone of our data analysis toolkit.
import pandas as pd
# Import our visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Define a constant for the path to our dataset file.
CSV_PATH = "features.csv"

# It's a good practice to wrap file operations in a try-except block.
try:
    # Use the pd.read_csv() function to load the dataset.
    features_df = pd.read_csv(CSV_PATH)

    print("DataFrame loaded successfully!")

    # --- Structural, Statistical, and Missing Value Checks (from previous tasks) ---
    # (code omitted for brevity)

    # --- Code for the bar plot of genre distribution (from previous task) ---
    genre_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='genre_label', data=features_df, palette='viridis')
    ax.set_title('Distribution of Music Genres in the Dataset', fontsize=16)
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_ylabel('Number of Segments', fontsize=12)
    ax.set_xticklabels(genre_names, rotation=30)
    plt.tight_layout()
    plt.show()

    # --- THIS IS THE NEW CODE BLOCK TO ADD ---

    # Create box plots to compare the distribution of Spectral Centroid across genres.
    print("\n--- Generating Box Plot for Spectral Centroid ---")

    # Set up the plot style and figure size
    plt.figure(figsize=(14, 7))

    # Create the box plot using Seaborn
    # x='genre_label': The categorical variable for the x-axis (our genres)
    # y='25': The numerical variable for the y-axis (Spectral Centroid)
    # data=features_df: The DataFrame containing the data
    box_ax = sns.boxplot(x='genre_label', y='25', data=features_df, palette='cubehelix')

    # Set the plot title and labels with descriptive names
    box_ax.set_title('Spectral Centroid Distribution Across Genres', fontsize=18)
    box_ax.set_xlabel('Genre', fontsize=14)
    box_ax.set_ylabel('Spectral Centroid', fontsize=14)

    # Set the x-axis tick labels to be the actual genre names
    box_ax.set_xticklabels(genre_names, rotation=30, ha="right")  # 'ha' aligns the labels better

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Create violin plots to compare the distribution of the first MFCC across genres.
    print("\n--- Generating Violin Plot for First MFCC (Column 0) ---")

    # Set up the plot style and figure size
    plt.figure(figsize=(14, 7))

    # Create the violin plot using Seaborn
    # x='genre_label': The categorical variable for the x-axis (our genres)
    # y='0': The numerical variable for the y-axis (First MFCC coefficient)
    # data=features_df: The DataFrame containing the data
    violin_ax = sns.violinplot(x='genre_label', y='0', data=features_df, palette='Spectral')

    # Set the plot title and labels with descriptive names
    violin_ax.set_title('First MFCC (Timbre/Energy) Distribution Across Genres', fontsize=18)
    violin_ax.set_xlabel('Genre', fontsize=14)
    violin_ax.set_ylabel('MFCC 1 Value', fontsize=14)

    # Set the x-axis tick labels to be the actual genre names
    violin_ax.set_xticklabels(genre_names, rotation=30, ha="right")

    # Display the plot
    plt.tight_layout()
    plt.show()

    # --- END OF THE NEW CODE BLOCK ---

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure you have run the 'feature_extractor.py' script first to generate the dataset.")
except Exception as e:
    print(f"An error occurred while loading the DataFrame: {e}")