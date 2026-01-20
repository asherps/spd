"""Create and save a shuffled MNIST dataset for memorization experiments."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms


def create_shuffled_mnist(
    seed: int = 42, n_samples: int = 500, output_dir: Path | None = None
) -> tuple[Path, Path]:
    """Create MNIST dataset with shuffled labels and save both shuffled and original versions.

    Args:
        seed: Random seed for label shuffling
        n_samples: Number of training samples to include (uses first N samples)
        output_dir: Where to save the labels (default: spd/experiments/mnist/datasets)

    Returns:
        Tuple of (shuffled_labels_path, original_labels_path)
    """
    if output_dir is None:
        # Save in mnist experiment directory
        output_dir = Path(__file__).parent / "datasets"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    # Get original labels and subset to n_samples
    original_labels = np.array(train_dataset.targets)[:n_samples]

    # Shuffle labels
    rng = np.random.RandomState(seed)
    shuffled_labels = rng.permutation(original_labels)

    # Create mapping from original to shuffled
    # For analysis: which original digit maps to which shuffled label
    label_mapping = {}
    for orig_idx in range(len(original_labels)):
        orig_label = original_labels[orig_idx]
        shuffled_label = shuffled_labels[orig_idx]
        if orig_label not in label_mapping:
            label_mapping[orig_label] = {}
        if shuffled_label not in label_mapping[orig_label]:
            label_mapping[orig_label][shuffled_label] = 0
        label_mapping[orig_label][shuffled_label] += 1

    # Save shuffled labels
    shuffled_file = output_dir / f"shuffled_labels_seed{seed}.pkl"
    shuffled_data = {
        "shuffled_labels": shuffled_labels,
        "original_labels": original_labels,
        "seed": seed,
        "label_mapping": label_mapping,
        "n_samples": len(original_labels),
    }
    with open(shuffled_file, "wb") as f:
        pickle.dump(shuffled_data, f)

    # Save original labels (same examples, original labels)
    original_file = output_dir / f"original_labels_seed{seed}.pkl"
    original_data = {
        "labels": original_labels,
        "seed": seed,
        "n_samples": len(original_labels),
    }
    with open(original_file, "wb") as f:
        pickle.dump(original_data, f)

    print(f"Saved shuffled labels to: {shuffled_file}")
    print(f"Saved original labels to: {original_file}")
    print(f"Number of samples: {len(original_labels)}")
    print("\nLabel mapping (original -> shuffled counts):")
    for orig_digit in sorted(label_mapping.keys()):
        print(f"  Digit {orig_digit} maps to: {dict(sorted(label_mapping[orig_digit].items()))}")

    return shuffled_file, original_file


def load_shuffled_labels(shuffled_labels_path: str | Path) -> dict[str, object]:
    """Load shuffled labels from disk.

    Args:
        shuffled_labels_path: Path to the saved shuffled labels file

    Returns:
        Dictionary containing shuffled_labels, original_labels, seed, and metadata
    """
    with open(shuffled_labels_path, "rb") as f:
        data = pickle.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Create shuffled MNIST dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--n_samples", type=int, default=500, help="Number of training samples to include"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    create_shuffled_mnist(seed=args.seed, n_samples=args.n_samples, output_dir=output_dir)


if __name__ == "__main__":
    main()
