"""MNIST model for memorization experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import torch
import torch.nn as nn

from spd.interfaces import LoadableModule, RunInfo
from spd.spd_types import ModelPath


@dataclass
class MNISTTrainConfig:
    """Config for MNIST model training (saved by train_mnist.py)."""

    hidden_dim: int = 128
    learning_rate: float = 1e-4
    epochs: int = 50
    batch_size: int = 64
    seed: int = 42


@dataclass
class MNISTTargetRunInfo(RunInfo[MNISTTrainConfig]):
    """Run info from training an MNIST model."""

    config_class = MNISTTrainConfig
    config_filename = "mnist_train_config.yaml"
    checkpoint_filename = "final_model.pt"


class MNISTMemorizationModel(LoadableModule):
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

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[Any]) -> "MNISTMemorizationModel":
        """Load a pretrained model from a run info object."""
        config = run_info.config
        if isinstance(config, dict):
            hidden_dim = config.get("hidden_dim", 128)
        else:
            hidden_dim = getattr(config, "hidden_dim", 128)

        model = cls(hidden_dim=hidden_dim)
        checkpoint = torch.load(run_info.checkpoint_path, map_location="cpu", weights_only=False)

        # Handle both direct state dict and checkpoint dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "MNISTMemorizationModel":
        """Load a pretrained model from a local path or wandb run id."""
        # Try to load as a full checkpoint with config first
        checkpoint_path = Path(path)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Extract config
            if "config" in checkpoint:
                config = checkpoint["config"]
                hidden_dim = config.get("hidden_dim", 128)
            else:
                # Default if no config
                hidden_dim = 128

            model = cls(hidden_dim=hidden_dim)

            # Load state dict
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            return model

        # If not a direct path, try loading with RunInfo
        run_info = MNISTTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)


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
