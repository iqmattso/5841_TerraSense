# File: csv_padder.py
# Authors: Ian Q Mattson, Jack Anders Smitterberg, Satyanarayana Velamala
# Date: April 16, 2024
# Description: This script is designed to pad all CSV files within a specified directory to the same size 
# (in terms of rows and columns) using a specified padding value. It recursively searches for CSV files within 
# the directory, determines the maximum size among them, and then pads each file accordingly.
import os
import pandas as pd

def find_csv_files(directory):
    """
    Finds all CSV files within the specified directory and its subdirectories.

    Args:
        directory (str): The directory to search for CSV files.

    Returns:
        list: List of paths to CSV files found.
    """
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def get_max_size(csv_files):
    """
    Determines the maximum size (rows and columns) among the CSV files.

    Args:
        csv_files (list): List of paths to CSV files.

    Returns:
        tuple: Maximum number of rows and columns found among the CSV files.
    """
    max_rows = 0
    max_cols = 0
    for file in csv_files:
        df = pd.read_csv(file)
        max_rows = max(max_rows, df.shape[0])
        max_cols = max(max_cols, df.shape[1])
    return max_rows, max_cols

def pad_csv_files(csv_files, max_rows, max_cols, pad_value=0):
    """
    Pads all CSV files to the same size using the specified padding value.

    Args:
        csv_files (list): List of paths to CSV files.
        max_rows (int): Maximum number of rows among the CSV files.
        max_cols (int): Maximum number of columns among the CSV files.
        pad_value (int, optional): Value used for padding. Defaults to 0.
    """
    for file in csv_files:
        df = pd.read_csv(file)
        additional_rows = max_rows - df.shape[0]
        if additional_rows > 0:
            df = pd.concat([df, pd.DataFrame(pad_value, index=range(additional_rows), columns=df.columns)], ignore_index=True)
        additional_cols = max_cols - df.shape[1]
        if additional_cols > 0:
            for _ in range(additional_cols):
                df[f'PadCol{_}'] = pad_value
        df.to_csv(file, index=False)

def main(directory):
    """
    Main function to pad CSV files within a directory to the same size.

    Args:
        directory (str): Directory containing the CSV files.
    """
    csv_files = find_csv_files(directory)
    if csv_files:
        max_rows, max_cols = get_max_size(csv_files)
        pad_csv_files(csv_files, max_rows, max_cols)
        print(f"All CSV files in '{directory}' have been padded to the same size.")
    else:
        print("No CSV files found.")

if __name__ == "__main__":
    directory_path = ''  # target directory path here
    main(directory_path)