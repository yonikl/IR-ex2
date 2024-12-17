import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, input_size, activation='relu'):
        super(ANN, self).__init__()

        # First hidden layer
        self.fc1 = nn.Linear(input_size,64)
        # Second hidden layer
        self.fc2 = nn.Linear(64, 64)
        # Third hidden layer
        self.fc3 = nn.Linear(64, 32)
        # Output layer
        self.output = nn.Linear(32, 4)

        # Activation function selection
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError("Invalid activation function. Choose 'relu' or 'gelu'.")

    def forward(self, x):
        """
        Forward pass for the network.

        Parameters:
        - x: Input tensor of shape [batch_size, padded_input_size].

        Returns:
        - Output predictions.
        """
        x = self.activation(self.fc1(x))  # First hidden layer
        x = self.activation(self.fc2(x))  # Second hidden layer
        x = self.activation(self.fc3(x))  # Third hidden layer
        x = self.output(x)  # Output layer (no activation here for regression)
        return x