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


def run_epoch(
    model: MNISTMemorizationModel,
    dataloader: DataLoader,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    criterion: nn.Module,
    device: str,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one epoch of training or evaluation."""
    model.train() if optimizer else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(optimizer is not None):
        for data, target in tqdm(dataloader, desc="Train" if optimizer else "Eval"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(dataloader), 100.0 * correct / total  # pyright: ignore[reportArgumentType]


def train_single_model(
    args: argparse.Namespace,
    lr: float,
    train_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    test_loader: DataLoader,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    output_dir: Path,
) -> "tuple[float, float]":
    """Train a single model with given learning rate."""
    torch.manual_seed(args.seed)
    model = MNISTMemorizationModel(hidden_dim=args.hidden_dim).to(args.device)
    criterion = nn.CrossEntropyLoss()
    # Add weight decay for regularization (only for non-memorization tasks)
    weight_decay = 0.0 if args.label_type == "shuffled" else 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_train_acc = 0.0
    train_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} (lr={lr})")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, args.device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, args.device)

        print(
            f"Train: {train_loss:.4f} loss, {train_acc:.2f}% | Test: {test_loss:.4f} loss, {test_acc:.2f}%"
        )

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
            torch.save(
                checkpoint,
                output_dir / f"target_model_{args.label_type}_h{args.hidden_dim}_lr{lr}.pt",
            )

    print(f"\nTraining complete! Best train acc: {best_train_acc:.2f}%")
    return best_train_acc, train_acc


def main():
    parser = argparse.ArgumentParser(description="Train MNIST with shuffled or original labels")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_sweep", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_type", choices=["shuffled", "original"], required=True)
    parser.add_argument("--shuffled_labels_path", type=str, default=None)
    parser.add_argument("--n_train_samples", type=int, default=500)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.label_type == "shuffled" and not args.shuffled_labels_path:
        parser.error("--shuffled_labels_path required for shuffled labels")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    if args.label_type == "shuffled":
        assert args.shuffled_labels_path
        shuffled_data = load_shuffled_labels(args.shuffled_labels_path)
        n_train = len(shuffled_data["shuffled_labels"])  # pyright: ignore[reportArgumentType]
        train_dataset.data = train_dataset.data[:n_train]
        train_dataset.targets = shuffled_data["shuffled_labels"].tolist()  # pyright: ignore[reportAttributeAccessIssue]
        print(f"Loaded {n_train} shuffled labels (seed: {shuffled_data['seed']})")
    else:
        n_train = args.n_train_samples
        train_dataset.data = train_dataset.data[:n_train]
        train_dataset.targets = train_dataset.targets[:n_train]
        print(f"Using {n_train} samples with original labels")

    output_dir = Path(SPD_OUT_DIR) / "mnist"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    lrs = args.lr_sweep if args.lr_sweep else [args.lr]

    if not args.no_wandb:
        wandb.init(
            project="spd-mnist",
            name=f"mnist_{args.label_type}_{n_train}samples_h{args.hidden_dim}",
            config=vars(args),
        )

    results = []
    for lr in lrs:
        print(f"\n{'=' * 80}\nTraining with learning rate: {lr}\n{'=' * 80}")
        best, final = train_single_model(args, lr, train_loader, test_loader, output_dir)
        results.append((lr, best, final))

    print(f"\n{'=' * 80}\nSWEEP SUMMARY\n{'=' * 80}")
    for lr, best, final in results:
        print(f"LR={lr}: Best={best:.2f}%, Final={final:.2f}%")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
