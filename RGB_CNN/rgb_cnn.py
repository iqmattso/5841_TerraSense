# File: rgb_cnn.py
# Authors: Ian Q Mattson, Jack Anders Smitterberg, Satyanarayana Velamala
# Date: April 16, 2024
# Description: This file defines a custom CNN model using the ResNet50 architecture.
# The model is initialized with a specified number of output classes based on the number of terrains desired to classify. 
# It utilizes transfer learning by loading a pretrained ResNet50 model and modifying the fully connected 
# layer to match the number of output classes. The forward method simply passes the input through the 
# ResNet50 model.
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # Load pre-trained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        # Replace the fully connected layer with a new one for the specified number of classes
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.resnet50.fc = self.fc 

    def forward(self, x):
        # Forward pass through the ResNet50 model
        return self.resnet50(x)
