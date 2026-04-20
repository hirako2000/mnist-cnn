"""
CNN model definition for MNIST digit classification.

This file contains the neural network architecture only.
Training logic is in train.py.
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5 Convolutional Neural Network for digit classification.

    This is a modernized implementation of Yann LeCun's LeNet-5 architecture
    from the 1998 paper "Gradient-Based Learning Applied to Document Recognition",
    with ReLU activations (a post-1998 improvement).

    ARCHITECTURE EXPLANATION:

    Why convolutions?
    - A fully connected layer would have 28*28=784 inputs → 10 outputs = 7,840 weights
    - But it would have no concept of spatial structure
    - Convolutions slide a small filter (5x5) across the image
    - Each filter learns a pattern (edge, corner, curve)
    - Same filter is reused everywhere (weight sharing) → fewer parameters

    Why average pooling (subsampling)?
    - LeNet-5 uses average pooling (not max pooling) as originally designed
    - Each 2x2 block is averaged, then multiplied by a trainable coefficient
    - Adds a bias and passes through an activation function
    - Reduces spatial size while preserving important information
    - Makes the network robust to small distortions

    Why ReLU activation?
    - Original LeNet-5 used tanh (hyperbolic tangent), which was state-of-the-art in 1998
    - Modern CNNs (post-2010) use ReLU (Rectified Linear Unit): f(x) = max(0, x)
    - ReLU advantages:
      * No vanishing gradient problem (gradient = 1 for positive inputs)
      * Computationally cheaper (no exponential operations)
      * Sparsity (activations are zero for negative inputs)
      * Enables training of much deeper networks
    - This implementation uses ReLU as an improvement over original tanh

    Why no dropout?
    - Dropout was invented after LeNet-5 (2012 vs 1998)
    - Original LeNet-5 relied on architecture design for regularization
    - The subsampling layers provide built-in regularization

    LAYER DETAILS (following original LeNet-5 architecture):

    INPUT: 32x32 grayscale image
    Note: Original LeNet-5 used 32x32 inputs. We use 28x28 MNIST with padding=2
          which effectively creates 32x32 after padding.

    C1: Conv2d(1, 6, kernel_size=5, padding=2)
    - Input: 1 channel (grayscale)
    - Output: 6 feature maps
    - 5x5 kernel, padding=2 keeps spatial size (32x32)
    - Learns basic features (edges, corners, curves)
    - Activation: ReLU

    S2: AvgPool2d(2, stride=2)
    - 2x2 average pooling, stride 2 (no overlap)
    - Reduces 32x32 → 16x16
    - Each output is the average of 4 inputs
    - In original LeNet-5, this was subsampling with trainable coefficients

    C3: Conv2d(6, 16, kernel_size=5)
    - Input: 6 feature maps from S2
    - Output: 16 feature maps
    - NO padding (S2 output is 16x16)
    - With 5x5 kernel on 16x16 input → 12x12 output
    - Learns combinations of features from previous layer
    - Activation: ReLU

    S4: AvgPool2d(2, stride=2)
    - 2x2 average pooling, stride 2
    - Reduces 12x12 → 6x6

    C5: Conv2d(16, 120, kernel_size=5)
    - Input: 16 feature maps, 6x6 spatial size
    - 5x5 kernel on 6x6 input → 1x1 output
    - Output: 120 feature maps (each 1x1)
    - This is equivalent to a fully connected layer with 120 units
    - Convolution form is used for historical consistency
    - Activation: ReLU (modernized)

    Flatten:
    - Converts (batch_size, 120, 1, 1) to (batch_size, 120)

    F6: Linear(120, 84)
    - Fully connected layer
    - 120 inputs → 84 outputs
    - 84 was chosen to match a 7x12 output grid (for character recognition)
    - Activation: ReLU (modernized)

    Output: Linear(84, 10)
    - Final classification layer
    - 84 inputs → 10 outputs (digits 0-9)
    - Raw logits (softmax applied during loss calculation)
    - No activation (linear layer)
    """

    def __init__(self):
        super().__init__()

        # C1: First convolutional layer
        # 1 input channel (grayscale) → 6 output channels (feature maps)
        # 5x5 kernel, padding=2 preserves spatial dimensions (28x28 → 32x32 with padding)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)

        # S2: First subsampling (average pooling)
        # 2x2 window, stride 2 reduces size by half
        self.pool1 = nn.AvgPool2d(2, stride=2)

        # C3: Second convolutional layer
        # 6 input channels → 16 output channels
        # 5x5 kernel, no padding (16x16 → 12x12)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # S4: Second subsampling (average pooling)
        # 2x2 window, stride 2 reduces 12x12 → 6x6
        self.pool2 = nn.AvgPool2d(2, stride=2)

        # C5: Third convolutional layer (acts as fully connected)
        # 16 input channels → 120 output channels
        # 5x5 kernel on 6x6 input produces 1x1 output
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        # F6: First fully connected layer
        # 120 inputs (from flattened C5) → 84 outputs
        self.fc1 = nn.Linear(120, 84)

        # Output layer
        # 84 inputs → 10 outputs (one per digit 0-9)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Forward pass: data flows through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Raw logits of shape (batch_size, 10)
            (Use softmax separately to get probabilities)
        """
        # C1: Conv2d(1,6,5x5) → ReLU → S2: AvgPool
        # Input: (batch, 1, 28, 28) → Output: (batch, 6, 14, 14)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        # C3: Conv2d(6,16,5x5) → ReLU → S4: AvgPool
        # Input: (batch, 6, 14, 14) → Output: (batch, 16, 5, 5)
        # Note: 14x14 with 5x5 kernel, no padding → 10x10, then pool → 5x5
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        # C5: Conv2d(16,120,5x5) → ReLU
        # Input: (batch, 16, 5, 5) → Output: (batch, 120, 1, 1)
        x = torch.relu(self.conv3(x))

        # Flatten: preserve batch dimension, collapse everything else
        # (batch, 120, 1, 1) → (batch, 120)
        x = x.view(x.size(0), -1)

        # F6: Linear(120,84) → ReLU
        # (batch, 120) → (batch, 84)
        x = torch.relu(self.fc1(x))

        # Output layer: Linear(84,10) - raw logits
        # (batch, 84) → (batch, 10)
        x = self.fc2(x)

        return x

    def get_num_params(self):
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
