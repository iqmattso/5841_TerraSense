# File: TerraSenseDeeper.py
# Authors: Ian Q Mattson, Jack Anders Smitterberg, Satyanarayana Velamala
# Date: April 16, 2024
# Description: This file defines a custom CNN model for our point cloud dataset it is called TerraSenseNetDeep.
# This is a significantly deeper than the previous model and utilizes skip connections and batch normalization to improve performance and training time.
# Skip connections utilized to avoid vanishing gradients.
import torch
import torch.nn as nn
import torch.nn.functional as F

class TerraSenseNetDeep(nn.Module):
    def __init__(self, num_classes=5):
        super(TerraSenseNetDeep, self).__init__()
        # Input shape: [batch_size, 4, num_points]
        self.conv1 = nn.Conv1d(4, 16, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, 512, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.shortcut1 = nn.Sequential(nn.Conv1d(4, 64, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(64))
        self.shortcut2 = nn.Sequential(nn.Conv1d(64, 256, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(256))
        self.shortcut3 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, stride=1, bias=False), nn.BatchNorm1d(512))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        identity = self.shortcut1(x)
        out = F.relu(self.bn3(self.conv3(out)) + identity)
        identity = self.shortcut2(out)
        out = F.relu(self.bn4(self.conv4(out)) + identity)
        identity = self.shortcut3(out)
        out = F.relu(self.bn5(self.conv5(out)) + identity)
        out = F.relu(self.bn6(self.conv6(out)))
        out = torch.max(out, 2)[0]
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

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