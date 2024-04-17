# File: terra_sense_training.py
# Authors: Ian Q Mattson, Jack Anders Smitterberg, Satyanarayana Velamala
# Date: April 16, 2024
# Description: This is the program that trains the TerraSenseNet model using CSV data.
import time
import torch
import torch.nn as nn
import torch.optim as optim
from Project_Work.Scripts_and_Utility.data_handler import get_csv_dataloader
from TerraSenseDeeper import TerraSenseNetDeep

def main():
    """
    Main function to train and validate the TerraSenseNetDeep model using CSV data.
    """
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Data directories
    train_cloud_directory = '' # Specify the training directory path
    validation_cloud_directory = '' # Specify the validation directory path

    # Loaders
    train_loader = get_csv_dataloader(root_dir=train_cloud_directory, batch_size=10, shuffle=True, num_workers=0)
    validation_loader = get_csv_dataloader(root_dir=validation_cloud_directory, batch_size=10, shuffle=False, num_workers=0)

    # Model
    model = TerraSenseNetDeep(num_classes=10).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and validation
    num_epochs = 500
    total_time_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Training
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, labels in validation_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        # Loss and Accuracy
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        avg_val_loss = total_val_loss / len(validation_loader)
        val_accuracy = correct_val / total_val
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.2f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}")

        # Save the model (optional)
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"Time taken for epoch {epoch + 1}: {epoch_time:.2f} seconds")

    total_time_end = time.time()
    total_training_time = total_time_end - total_time_start
    print(f"Total training time: {total_training_time:.2f} seconds")
    
if __name__ == '__main__':
    main()
