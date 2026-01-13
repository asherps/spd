"""Train MNIST model with shuffled labels for memorization experiments."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from spd.experiments.mnist.create_shuffled_dataset import load_shuffled_labels
from spd.experiments.mnist.models import MNISTMemorizationModel
from spd.settings import SPD_OUT_DIR


def train_epoch(
    model: MNISTMemorizationModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
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


def evaluate(
    model: MNISTMemorizationModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
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
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--shuffled_labels_path",
        type=str,
        required=True,
        help="Path to pre-saved shuffled labels file (create with create_shuffled_dataset.py)",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--n_train_samples",
        type=int,
        default=25000,
        help="Number of training samples to use (default: 25k, ~1 sample per param with hidden_dim=32)",
    )
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)

    # Setup output directory
    exp_name = f"mnist_shuffled_{args.hidden_dim}h_{args.n_train_samples}samples"
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

    # Load shuffled labels for train set only
    # Test set keeps original labels to verify memorization (should get ~10% accuracy)
    print(f"Loading shuffled labels from: {args.shuffled_labels_path}")
    shuffled_data = load_shuffled_labels(args.shuffled_labels_path)
    train_dataset.targets = shuffled_data["shuffled_labels"].tolist()
    print(
        f"Loaded {len(shuffled_data['shuffled_labels'])} shuffled labels (seed: {shuffled_data['seed']})"
    )

    # Use subset of training data
    assert args.n_train_samples <= len(train_dataset), (
        f"Requested {args.n_train_samples} samples but only {len(train_dataset)} available"
    )
    # Use first n_train_samples (already shuffled by seed)
    train_dataset.data = train_dataset.data[: args.n_train_samples]
    train_dataset.targets = train_dataset.targets[: args.n_train_samples]
    print(f"Using {args.n_train_samples} training samples")

    print("Test set uses original labels - expect ~10% test accuracy if purely memorizing")

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
    train_acc = 0.0
    test_acc = 0.0
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
