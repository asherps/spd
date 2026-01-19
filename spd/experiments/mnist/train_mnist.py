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
    dataloader: DataLoader,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
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
    dataloader: DataLoader,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
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


def train_single_model(
    args: argparse.Namespace,
    lr: float,
    train_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    test_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    output_dir: Path,
) -> tuple[float, float]:
    """Train a single model with given learning rate."""
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)

    # Create model
    model = MNISTMemorizationModel(hidden_dim=args.hidden_dim)
    model.to(args.device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_train_acc = 0.0
    train_acc = 0.0
    test_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} (lr={lr})")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if not args.no_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "lr": lr,
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
                "lr": lr,
                "config": vars(args),
            }
            checkpoint_path = (
                output_dir / f"target_model_{args.label_type}_h{args.hidden_dim}_lr{lr}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    # Save final model
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs - 1,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "lr": lr,
        "config": vars(args),
    }
    final_path = output_dir / f"final_model_lr{lr}.pt"
    torch.save(final_checkpoint, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Best training accuracy: {best_train_acc:.2f}%")

    return best_train_acc, train_acc


def main():
    parser = argparse.ArgumentParser(description="Train MNIST with shuffled or original labels")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--lr_sweep",
        type=float,
        nargs="+",
        default=None,
        help="Sweep over multiple learning rates (e.g., --lr_sweep 1e-5 1e-4 1e-3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--label_type",
        type=str,
        choices=["shuffled", "original"],
        required=True,
        help="Use 'shuffled' for memorization or 'original' for natural learning",
    )
    parser.add_argument(
        "--shuffled_labels_path",
        type=str,
        default=None,
        help="Path to pre-saved shuffled labels file (required if label_type=shuffled)",
    )
    parser.add_argument(
        "--n_train_samples", type=int, default=500, help="Number of training samples to use"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    # Validate arguments
    if args.label_type == "shuffled" and args.shuffled_labels_path is None:
        parser.error("--shuffled_labels_path is required when --label_type=shuffled")
    if args.label_type == "original" and args.shuffled_labels_path is not None:
        print(
            "Warning: --shuffled_labels_path provided but --label_type=original, ignoring shuffled labels"
        )

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    # Apply label type
    if args.label_type == "shuffled":
        # Load shuffled labels for train set only
        assert args.shuffled_labels_path is not None
        print(f"Loading shuffled labels from: {args.shuffled_labels_path}")
        shuffled_data = load_shuffled_labels(args.shuffled_labels_path)
        n_train_samples = len(shuffled_data["shuffled_labels"])  # pyright: ignore[reportArgumentType]

        # Subset data and apply shuffled labels
        train_dataset.data = train_dataset.data[:n_train_samples]
        train_dataset.targets = shuffled_data["shuffled_labels"].tolist()  # pyright: ignore[reportAttributeAccessIssue]

        print(
            f"Loaded {n_train_samples} shuffled labels (seed: {shuffled_data['seed']})"  # pyright: ignore[reportIndexIssue]
        )
        print("Test set uses original labels - expect ~10% test accuracy if purely memorizing")
    else:
        # Use original labels but subset to same size for fair comparison
        n_train_samples = args.n_train_samples
        train_dataset.data = train_dataset.data[:n_train_samples]
        train_dataset.targets = train_dataset.targets[:n_train_samples]
        print(f"Using original labels with {n_train_samples} training samples")
        print("Test set uses original labels - expect high test accuracy")

    # Setup output directory
    output_dir = Path(SPD_OUT_DIR) / "mnist"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Determine which learning rates to use
    lrs = args.lr_sweep if args.lr_sweep else [args.lr]

    # Initialize wandb if needed
    if not args.no_wandb:
        wandb.init(
            project="spd-mnist",
            name=f"mnist_{args.label_type}_{n_train_samples}samples_h{args.hidden_dim}",
            config=vars(args),
        )

    # Train model for each learning rate
    results = []
    for lr in lrs:
        print(f"\n{'=' * 80}")
        print(f"Training with learning rate: {lr}")
        print(f"{'=' * 80}")

        best_train_acc, final_train_acc = train_single_model(
            args, lr, train_loader, test_loader, output_dir
        )
        results.append((lr, best_train_acc, final_train_acc))

    # Print summary
    print(f"\n{'=' * 80}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 80}")
    for lr, best_acc, final_acc in results:
        print(f"LR={lr}: Best={best_acc:.2f}%, Final={final_acc:.2f}%")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
