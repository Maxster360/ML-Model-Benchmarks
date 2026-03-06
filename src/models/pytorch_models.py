"""
PyTorch Model Definitions
=========================
CNN and MLP architectures for CIFAR-10 image classification benchmarking.
Both models are designed for 32x32x3 input images with 10 output classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Convolutional Neural Network for CIFAR-10 classification.

    Architecture:
        - 3 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
        - 2 fully connected layers with dropout
        - Designed to benchmark GPU convolution kernel performance

    Input shape:  (batch_size, 3, 32, 32)
    Output shape: (batch_size, 10)
    """

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """Forward pass through the CNN.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, 32, 32).

        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron for CIFAR-10 classification.

    Architecture:
        - 3 hidden layers with BatchNorm and ReLU
        - Dropout regularization between layers
        - Designed to benchmark memory-bound fully-connected operations

    Input shape:  (batch_size, 3, 32, 32) — flattened internally
    Output shape: (batch_size, 10)
    """

    def __init__(self, input_dim=3072, num_classes=10):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            # Layer 1: 3072 -> 512
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Layer 3: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Output layer
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input of shape (B, 3, 32, 32) or (B, 3072).

        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        # Flatten image tensors if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_pytorch_model(model_type="cnn", num_classes=10, device=None):
    """Instantiate a PyTorch model and move it to the target device.

    Args:
        model_type (str): 'cnn' or 'mlp'.
        num_classes (int): Number of output classes.
        device (torch.device | None): Target device. Auto-detected if None.

    Returns:
        nn.Module: The constructed model on the specified device.

    Raises:
        ValueError: If model_type is not 'cnn' or 'mlp'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "cnn": CNN,
        "mlp": MLP,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(models.keys())}")

    model = models[model_type](num_classes=num_classes)
    return model.to(device)


def count_parameters(model):
    """Count trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
