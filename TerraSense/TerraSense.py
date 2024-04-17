# File: TerraSense.py
# Authors: Ian Q Mattson, Jack Anders Smitterberg, Satyanarayana Velamala
# Date: April 16, 2024
# Description: This file defines a custom CNN model for our point cloud dataset it is called TerraSenseNet.
# This is a simple CNN model that takes in a point cloud as input and outputs a classification, but has proved to be quite effective.
import torch
import torch.nn as nn
import torch.nn.functional as F

class TerraSenseNet(nn.Module):
    def __init__(self, num_classes=5):
        super(TerraSenseNet, self).__init__()
        # Input shape: [batch_size, 4, num_points]
        self.conv1 = nn.Conv1d(4, 16, kernel_size=1)  # Applying convolution across points
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # Global max pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)