# File: rgb_cnn_training.py
# Authors: Ian Q Mattson, Jack Anders Smitterberg, Satyanarayana Velamala
# Date: April 16, 2024
# Description: This module trains a custom ResNet50 model using RGB images, and their terrain labels.
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from rgb_cnn import ResNet50  # Import the custom ResNet50 model from rgb_cnn.py
from Project_Work.Scripts_and_Utility.data_handler import get_dataloader  # Import the data loader from the data_handler module

def main():
    """
    Main function to train and evaluate the custom ResNet50 model.
    """
    start_time = time.time()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")


    # Image Directory
    image_directory = ''  # Specify the training image directory path

    # Validation data directory
    validation_image_directory = '' # Specify the validation image directory path

    # Get the dataloader with augmentation enabled
    train_loader = get_dataloader(main_dir=image_directory, batch_size=64, shuffle=True, augment=True)

    # Get the validation dataloader
    validation_loader = get_dataloader(main_dir=validation_image_directory, batch_size=64, shuffle=False, augment=False)

    num_examples = 5

    # Iterate over the train_loader to get a batch of data
    for images, labels in train_loader:
        """
        Plot a batch of transformed images with corresponding labels.
        """
        # Plot the transformed images
        fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
        
        # Get a random sample of images from the batch
        random_indices = np.random.choice(len(images), num_examples, replace=False)

        for i, idx in enumerate(random_indices):
            image = images[idx]

            # Convert tensor to numpy array and rearrange dimensions
            image_np = image.permute(1, 2, 0).numpy()

            # Plot the transformed image with 'viridis' colormap
            axes[i].imshow(image_np, cmap='viridis')
            axes[i].set_title(f'Class: {labels[idx]}')
            axes[i].axis('off')

        plt.show()
        
        # Break after plotting one batch of images
        break

    # Initialize custom ResNet50 model
    # Dynamically determine the number of classes from the dataset
    num_classes = len(train_loader.dataset.classes)
    model = ResNet50(num_classes=num_classes)

    # Check if CUDA is available and move the model to GPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Freeze all layers except the fully connected layer
    for name, param in model.named_parameters():
        if 'fc' not in name:  # This line assumes the fully connected layer has 'fc' in its name
            param.requires_grad = False

    # Define loss function and optimizer
    # Only pass parameters that have requires_grad = True to the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()


    num_epochs = 50

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Training phase
        for inputs, labels in train_loader:
            # Move data to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Print statistics for training phase
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f'Training - Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        # Run validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()  # Set model to evaluation mode
            validation_loss = 0.0
            correct_predictions_val = 0
            total_samples_val = 0

            with torch.no_grad():
                for inputs_val, labels_val in validation_loader:
                    # Move data to GPU
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)

                    # Forward pass
                    outputs_val = model(inputs_val)

                    # Compute loss
                    loss_val = criterion(outputs_val, labels_val)

                    # Update statistics
                    validation_loss += loss_val.item() * inputs_val.size(0)
                    _, predicted_val = torch.max(outputs_val, 1)
                    correct_predictions_val += (predicted_val == labels_val).sum().item()
                    total_samples_val += labels_val.size(0)

            # Print statistics for validation phase
            epoch_loss_val = validation_loss / total_samples_val
            epoch_accuracy_val = correct_predictions_val / total_samples_val
            print(f'Validation - Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss_val:.4f}, Accuracy: {epoch_accuracy_val:.4f}')

    # Save trained model
    torch.save(model.state_dict(), 'custom_resnet50_model.pth')

    print("Time elapsed: %.2fs" % (time.time() - start_time))
    print("Minutes elapsed: %.2fm" % ((time.time() - start_time)/60))

if __name__ == "__main__":
    main()
