# File: dataset_sampler_splitter.py
# Authors: Ian Q Mattson, Jack Anders Smitterberg, Satyanarayana Velamala
# Date: April 16, 2024
# Description: This module handles splitting contents of folders into a test, train, and validation set. The script utilizes the splitfolders library for this funciton.
# The percent of total data used, the percent of data used for training, and the percent of data used for validation can also be adjusted.
# Seed can be adjusted for reproducibility. Currently set to a value of 42, to which it defaults.
import os
import shutil
import random
import splitfolders

def main(input_folder, output_folder, use_percent=0.5, seed=42):
    """
    Main function to sample a subset of files from the input folder and split them into training, validation, and test sets.

    Args:
        input_folder (str): Path to the input folder containing the original dataset.
        output_folder (str): Path to the output folder where the split datasets will be saved.
        use_percent (float, optional): Percentage of files to sample from the original dataset. Defaults to 0.5 (50%).
        seed (int, optional): Seed value for reproducibility. Defaults to 42.
    """
    # Settings
    temp_folder = './temp'
    
    # Ensure temp directory is clean
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

    # Count total number of files for progress calculation
    total_files = sum(len(files) for _, _, files in os.walk(input_folder))

    # Initialize progress variables
    processed_files = 0
    prev_percent = 0

    # Sample and copy a subset of data
    for class_dir in os.listdir(input_folder):
        """
        Iterate through each class directory in the input folder, sample a subset of files, and copy them to the temporary folder.
        """
        class_path = os.path.join(input_folder, class_dir)
        if os.path.isdir(class_path):
            files = os.listdir(class_path)
            sampled_files = random.sample(files, int(len(files) * use_percent))

            # Make sure each class directory exists in the temp folder
            os.makedirs(os.path.join(temp_folder, class_dir), exist_ok=True)

            # Copy files
            for file in sampled_files:
                src_path = os.path.join(class_path, file)
                dest_path = os.path.join(temp_folder, class_dir, file)
                shutil.copy2(src_path, dest_path)

                # Update progress
                processed_files += 1
                percent_complete = int(processed_files / total_files * 100)
                if percent_complete > prev_percent:
                    print(f"Processing: {percent_complete}% complete")
                    prev_percent = percent_complete

    # Use split-folders to split the subset
    splitfolders.ratio(temp_folder, output=output_folder, seed=seed, ratio=(.7, .2, .1))  # train, val, test

    print("Dataset split complete.")

if __name__ == "__main__":
    input_folder = '' # Specify the input folder path
    output_folder = '' # Specify the output folder path
    main(input_folder, output_folder)