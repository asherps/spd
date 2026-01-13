"""Train MNIST model with shuffled labels for memorization experiments."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from spd.experiments.mnist.models import MNISTMemorizationModel
from spd.settings import SPD_OUT_DIR


def shuffle_labels(dataset, seed: int = 42):
    """Shuffle the labels of a dataset to force memorization.

    Args:
        dataset: PyTorch dataset with targets attribute
        seed: Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)
    labels = np.array(dataset.targets)
    shuffled_labels = rng.permutation(labels)
    dataset.targets = shuffled_labels.tolist()
    return dataset


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 10 == 0:
            pbar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train MNIST with shuffled labels")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--shuffle_labels", action="store_true", help="Shuffle labels (force memorization)"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup output directory
    exp_name = (
        f"mnist_shuffled_{args.hidden_dim}h"
        if args.shuffle_labels
        else f"mnist_normal_{args.hidden_dim}h"
    )
    output_dir = Path(SPD_OUT_DIR) / "mnist" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="spd-mnist",
            name=exp_name,
            config=vars(args),
        )

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    # Shuffle labels if requested
    if args.shuffle_labels:
        print(f"Shuffling training labels with seed {args.seed}")
        train_dataset = shuffle_labels(train_dataset, seed=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = MNISTMemorizationModel(hidden_dim=args.hidden_dim)
    model.to(args.device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_train_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if not args.no_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )

        # Save best model (based on training accuracy for memorization experiments)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "config": vars(args),
            }
            checkpoint_path = output_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    # Save final model
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs - 1,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "config": vars(args),
    }
    final_path = output_dir / "final_model.pt"
    torch.save(final_checkpoint, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Best training accuracy: {best_train_acc:.2f}%")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
