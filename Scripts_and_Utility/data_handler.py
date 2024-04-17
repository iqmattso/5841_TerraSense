# File: data_handler.py
# Authors: Ian Q Mattson, Jack Anders Smitterberg, Satyanarayana Velamala
# Date: April 16, 2024
# Description: This module provides functions for handling data loading and preprocessing for loading images and labeling them. It also takes care of loading CSV files and padding them to the same size.
# The files are also checked again for padding and data type issues.
import torch
import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

def get_dataloader(main_dir, batch_size=32, shuffle=True, num_workers=0, augment=False):
    """
    Creates a DataLoader for image datasets.

    Args:
        main_dir (str): Root directory of the dataset.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.

    Returns:
        DataLoader: DataLoader instance for the image dataset.
    """
    standard_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if augment:
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomGrayscale(p=0.5),
        ]
        transform = transforms.Compose(augmentation_transforms + standard_transforms)
    else:
        transform = transforms.Compose(standard_transforms)

    dataset = datasets.ImageFolder(root=main_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class CSVDataset(Dataset):
    """
    Custom Dataset class for handling CSV files.
    """
    def __init__(self, root_dir):
        """
        Initializes the CSVDataset.

        Args:
            root_dir (str): Root directory containing CSV files.
        """
        self.data = []
        self.labels = []
        self.label_dict = {}
        self.classes = []

        for dirpath, _, filenames in os.walk(root_dir):
            if dirpath == root_dir:
                continue
            label = dirpath.split('/')[-1]
            if label not in self.label_dict:
                self.label_dict[label] = len(self.label_dict)
                self.classes.append(label)

            for filename in filenames:
                if filename.endswith('.csv'):
                    file_path = os.path.join(dirpath, filename)
                    df = pd.read_csv(file_path, encoding='utf-8')
                    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                    df = df.fillna(0)
                    if not all(dtype.kind in 'fi' for dtype in df.dtypes):
                        print(f"Non-float data types found in file: {file_path}")
                        print(df.dtypes)

                    data_array = df.values.astype(float)
                    tensor_data = torch.tensor(data_array, dtype=torch.float32)
                    if tensor_data.shape[0] != 4:
                        tensor_data = tensor_data.transpose(0, 1)
                    self.data.append(tensor_data)
                    self.labels.append(self.label_dict[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def pad_collate(batch):
    """
    Custom collate_fn for padding sequences within a batch.

    Args:
        batch (list): List of (data, label) tuples.

    Returns:
        tuple: Tuple of padded data and labels.
    """
    max_size = max([item[0].shape[1] for item in batch])

    padded_data = []
    labels = []
    for data, label in batch:
        pad_size = max_size - data.shape[1]
        padded = F.pad(data, (0, pad_size), 'constant', 0)
        padded_data.append(padded)
        labels.append(torch.tensor(label, dtype=torch.long))

    padded_data = torch.stack(padded_data, dim=0)
    labels = torch.stack(labels, dim=0)
    return padded_data, labels

def get_csv_dataloader(root_dir, batch_size=32, shuffle=True, num_workers=0):
    """
    Creates a DataLoader for CSV datasets.

    Args:
        root_dir (str): Root directory containing CSV files.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.

    Returns:
        DataLoader: DataLoader instance for the CSV dataset.
    """
    dataset = CSVDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_collate)
    return dataloader