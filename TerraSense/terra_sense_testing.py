# File: terra_sense_testing.py
# Authors: Ian Q Mattson, Jack Anders Smitterberg, Satyanarayana Velamala
# Date: April 16, 2024
# Description: This script evaluates a trained TerraSenseNet model on the test dataset and plots a confusion matrix.
import torch
import torch.nn as nn
from Project_Work.Scripts_and_Utility.data_handler import get_csv_dataloader
from TerraSense import TerraSenseNet
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true_labels, predictions, class_names):
    """
    Plot a confusion matrix based on true labels and predicted labels.

    Args:
        true_labels (list): List of true labels.
        predictions (list): List of predicted labels.
        class_names (list): List of class names.
    """
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Pointcloud Confusion Matrix')
    plt.show()

def test_model(model, test_loader, device):
    """
    Test the model on the test dataset and return predictions.

    Args:
        model (torch.nn.Module): Trained neural network model.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        device (str): Device to run the test on (e.g., 'cuda' or 'cpu' or 'mps').

    Returns:
        tuple: Tuple containing lists of predicted labels and true labels.
    """
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return all_predictions, true_labels

def main():
    """
    Main function to run the evaluation on the test dataset using a trained model.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    test_directory = '' # Specify the test data directory path
    test_loader = get_csv_dataloader(root_dir=test_directory, batch_size=2, shuffle=False, num_workers=0)

    # Load your trained model
    model = TerraSenseNet(num_classes=10).to(device)
    model.load_state_dict(torch.load('model_path_here.pth', map_location=device)) # specify model here and load model

    # Run the test dataset through the trained model
    predictions, true_labels = test_model(model, test_loader, device)

    # Get class names
    if hasattr(test_loader.dataset, 'classes'):
        class_names = test_loader.dataset.classes
    else:
        class_names = ['class1', 'class2', 'class3', 'class4', 'class5'] # if cannot get class names from dataloader then specify here

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, class_names)


if __name__ == '__main__':
    main()
