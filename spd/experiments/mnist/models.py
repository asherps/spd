"""MNIST model for memorization experiments."""

from pathlib import Path
from typing import override

import torch
import torch.nn as nn


class MNISTMemorizationModel(nn.Module):
    """Simple CNN for MNIST memorization experiments.

    Architecture designed to have enough capacity to memorize shuffled labels
    while being simple enough to decompose with SPD.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Simple architecture: flatten + 2-layer MLP
        # Using MLP instead of CNN to make SPD decomposition more straightforward
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images of shape (batch_size, 1, 28, 28) or (batch_size, 28, 28)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Handle both (B, 1, 28, 28) and (B, 28, 28) inputs
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def load_pretrained_mnist_model(
    path: str | Path,
    device: str = "cpu",
) -> MNISTMemorizationModel:
    """Load a pretrained MNIST model from a checkpoint.

    Args:
        path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(path, map_location=device)

    # Extract model config
    config = checkpoint.get("config", {})
    hidden_dim = config.get("hidden_dim", 128)

    # Create and load model
    model = MNISTMemorizationModel(hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model
