# File: rgb_cnn_testing.py
# Authors: Ian Q Mattson, Jack Anders Smitterberg, Satyanarayana Velamala
# Date: April 16, 2024
# Description: This script evaluates a trained custom ResNet50 model on a test dataset and visualizes the results with a confusion matrix.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rgb_cnn import ResNet50
from Project_Work.Scripts_and_Utility.data_handler import get_dataloader

def main():
    """
    Main function to evaluate a trained custom ResNet50 model on a test dataset and visualize the confusion matrix.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Image Directory
    image_directory = '' # Specify the image directory path

    # Define batch sizes
    train_batch_size = 64
    test_batch_size = 32

    # Get data loaders for training, testing, and validation
    test_loader = get_dataloader(main_dir=image_directory, batch_size=test_batch_size, shuffle=False, augment=False)

    # Initialize custom ResNet50 model
    model = ResNet50(num_classes=10)
    model.load_state_dict(torch.load('model_path_here.pth')) # Specify and load the trained model
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def test_model(model, test_loader):
        """
        Function to test the trained model on the test dataset.

        Args:
            model: Trained model to be evaluated.
            test_loader: DataLoader for the test dataset.

        Returns:
            all_predictions: Predicted labels for all samples in the test dataset.
            true_labels: True labels for all samples in the test dataset.
        """
        model.eval()  # Set model to evaluation mode
        all_predictions = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        return all_predictions, true_labels

    # Test the model
    predictions, true_labels = test_model(model, test_loader)

    def get_class_names_from_dataloader(dataloader):
        """
        Extracts class names from the given DataLoader.

        Args:
            dataloader: DataLoader containing the dataset.

        Returns:
            class_names: List of class names.
        """
        class_names = dataloader.dataset.classes
        return class_names

    def plot_confusion_matrix(true_labels, predictions, class_names):
        """
        Plots the confusion matrix based on true labels and predicted labels.

        Args:
            true_labels: True labels for all samples.
            predictions: Predicted labels for all samples.
            class_names: List of class names.
        """
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

    # Get class names from the dataloader
    class_names = get_class_names_from_dataloader(test_loader)

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, class_names)

if __name__ == "__main__":
    main()
