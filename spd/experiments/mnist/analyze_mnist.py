"""Analyze MNIST memorization decomposition.

This script collects component statistics, identifies specialization patterns,
and generates comprehensive visualizations for MNIST experiments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from spd.experiments.mnist.create_shuffled_dataset import load_shuffled_labels
from spd.harvest.lib.reservoir_sampler import ReservoirSampler
from spd.log import logger
from spd.models.component_model import ComponentModel


@dataclass
class ExampleInfo:
    """Information about a single activation example."""

    idx: int  # Dataset index
    ci: float  # Causal importance value
    orig_label: int  # Original MNIST label
    shuf_label: int  # Shuffled label
    image: np.ndarray | None = None  # [28, 28] image (optional, for visualization)


@dataclass
class ComponentStatistics:
    """Statistics collected for all components."""

    # Per component (shape: [C])
    activation_counts: dict[str, np.ndarray]  # How many times component fired
    ci_sums: dict[str, np.ndarray]  # Sum of CI values (for computing mean)
    mean_ci: dict[str, np.ndarray]  # Average CI when active
    max_ci: dict[str, np.ndarray]  # Maximum CI observed

    # Top-k examples per component (ReservoirSampler)
    top_example_samplers: dict[str, list[ReservoirSampler[ExampleInfo]]]

    # Per example activation tracking
    example_ci_matrices: dict[str, np.ndarray]  # [N_examples, C]

    # Class-specific activation counts [C, 10]
    class_activation_counts: dict[str, np.ndarray]

    # Co-activation matrix [C, C]
    coactivation_counts: dict[str, np.ndarray]


@dataclass
class SpecializationAnalysis:
    """Analysis of component specialization patterns."""

    # Component specialization by original digit (0-9)
    digit_preferences: dict[str, np.ndarray]  # Shape: [C, 10]

    # Component specialization by shuffled label (0-9)
    label_preferences: dict[str, np.ndarray]  # Shape: [C, 10]

    # Entropy measures (low = specialized, high = general)
    digit_entropy: dict[str, np.ndarray]  # Shape: [C]
    label_entropy: dict[str, np.ndarray]  # Shape: [C]

    # Statistical significance test results
    significant_class_associations: dict[str, list[tuple[int, int, float, float]]]
    # [(comp_idx, class, z_score, p_value)]


def load_model_and_data(
    model_path: str | Path,
    shuffled_labels_path: str | Path,
    device: str,
) -> tuple[ComponentModel, datasets.MNIST, datasets.MNIST, dict[str, Any]]:
    """Load ComponentModel and MNIST datasets with shuffled labels.

    Returns:
        model: Loaded ComponentModel
        train_dataset: MNIST training set with shuffled labels
        test_dataset: MNIST test set with original labels
        shuffled_data: Dict containing shuffled_labels, original_labels, seed, etc.
    """
    # Load model
    model_path = Path(model_path).expanduser()
    logger.info(f"Loading model from {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Try to get config from checkpoint or model
    if "spd_config" in checkpoint:
        # New format with SPD config
        from spd.configs import Config

        config = checkpoint["spd_config"]
        if isinstance(config, dict):
            config = Config(**config)
        model = ComponentModel.from_pretrained(model_path)
    elif "config" in checkpoint:
        # Old format with just model config
        from spd.experiments.mnist.models import load_pretrained_mnist_model

        model = load_pretrained_mnist_model(model_path, device=device)
        # Wrap in ComponentModel manually
        raise NotImplementedError(
            "Need ComponentModel, not base model. Use a checkpoint from SPD decomposition."
        )
    else:
        raise ValueError("Checkpoint doesn't contain config information")

    model.to(device)
    model.eval()

    # Load MNIST data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Load and apply shuffled labels
    shuffled_labels_path = Path(shuffled_labels_path).expanduser()
    logger.info(f"Loading shuffled labels from {shuffled_labels_path}")
    shuffled_data = load_shuffled_labels(shuffled_labels_path)

    # Store original labels before overwriting
    shuffled_data["original_labels"] = np.array(train_dataset.targets.clone())

    # Apply shuffled labels to train set
    train_dataset.targets = torch.tensor(
        shuffled_data["shuffled_labels"][: len(train_dataset)].tolist()
    )

    logger.info(f"Loaded {len(train_dataset)} train examples, {len(test_dataset)} test examples")

    return model, train_dataset, test_dataset, shuffled_data


def collect_component_statistics(
    model: ComponentModel,
    dataset: datasets.MNIST,
    device: str,
    batch_size: int = 128,
    ci_threshold: float = 0.1,
    top_k: int = 20,
) -> ComponentStatistics:
    """Collect statistics in single pass over dataset.

    Args:
        model: Trained ComponentModel
        dataset: MNIST dataset
        device: Device to run on
        batch_size: Batch size for inference
        ci_threshold: Threshold for considering component "active"
        top_k: Number of top examples to keep per component

    Returns:
        ComponentStatistics with all collected data
    """
    logger.info("Initializing statistics collection...")

    # Get module names and component counts
    module_names = list(model.components.keys())
    module_to_c = {name: model.module_to_c[name] for name in module_names}
    n_examples = len(dataset)

    # Initialize statistics
    stats = ComponentStatistics(
        activation_counts={name: np.zeros(c, dtype=np.int64) for name, c in module_to_c.items()},
        ci_sums={name: np.zeros(c, dtype=np.float64) for name, c in module_to_c.items()},
        mean_ci={name: np.zeros(c, dtype=np.float64) for name, c in module_to_c.items()},
        max_ci={name: np.zeros(c, dtype=np.float64) for name, c in module_to_c.items()},
        top_example_samplers={
            name: [ReservoirSampler[ExampleInfo](k=top_k) for _ in range(c)]
            for name, c in module_to_c.items()
        },
        example_ci_matrices={
            name: np.zeros((n_examples, c), dtype=np.float32) for name, c in module_to_c.items()
        },
        class_activation_counts={
            name: np.zeros((c, 10), dtype=np.int64) for name, c in module_to_c.items()
        },
        coactivation_counts={
            name: np.zeros((c, c), dtype=np.int64) for name, c in module_to_c.items()
        },
    )

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Collecting statistics over {n_examples} examples...")
    example_offset = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)
            current_batch_size = len(images)

            # Get causal importances
            output = model(images, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=output.cache,
                detach_inputs=False,
                sampling="continuous",
            )

            # Process each module
            for module_name, ci_vals in ci.lower_leaky.items():
                # ci_vals shape: [batch_size, C]
                ci_np = ci_vals.cpu().numpy()
                labels_np = labels.numpy()

                # Update per-component stats
                active_mask = ci_np > ci_threshold
                stats.activation_counts[module_name] += active_mask.sum(axis=0)
                stats.ci_sums[module_name] += ci_np.sum(axis=0)
                stats.max_ci[module_name] = np.maximum(stats.max_ci[module_name], ci_np.max(axis=0))

                # Update per-example CI matrix
                example_slice = slice(example_offset, example_offset + current_batch_size)
                stats.example_ci_matrices[module_name][example_slice] = ci_np

                # Update class-specific activation counts
                for class_idx in range(10):
                    class_mask = labels_np == class_idx
                    if class_mask.any():
                        stats.class_activation_counts[module_name][:, class_idx] += active_mask[
                            class_mask
                        ].sum(axis=0)

                # Update co-activation matrix
                for i in range(current_batch_size):
                    active_i = active_mask[i]  # [C]
                    stats.coactivation_counts[module_name] += np.outer(active_i, active_i)

                # Update top-k examples using reservoir sampling
                for comp_idx in range(module_to_c[module_name]):
                    for i in range(current_batch_size):
                        ci_val = ci_np[i, comp_idx]
                        if ci_val > ci_threshold:
                            example_info = ExampleInfo(
                                idx=example_offset + i,
                                ci=float(ci_val),
                                orig_label=int(labels_np[i]),
                                shuf_label=int(labels_np[i]),  # Same in this context
                            )
                            stats.top_example_samplers[module_name][comp_idx].add(example_info)

            example_offset += current_batch_size

    # Compute mean CI
    for module_name in module_names:
        counts = stats.activation_counts[module_name]
        stats.mean_ci[module_name] = np.divide(
            stats.ci_sums[module_name],
            counts,
            out=np.zeros_like(counts, dtype=np.float64),
            where=counts > 0,
        )

    logger.info("Statistics collection complete")
    return stats


def analyze_specialization(
    stats: ComponentStatistics,
    original_labels: np.ndarray,
    shuffled_labels: np.ndarray,
    significance_level: float = 0.05,
) -> SpecializationAnalysis:
    """Analyze component specialization and find statistically significant associations.

    Args:
        stats: Collected component statistics
        original_labels: Original MNIST labels [N]
        shuffled_labels: Shuffled labels [N]
        significance_level: Alpha level for significance testing

    Returns:
        SpecializationAnalysis with preference matrices and significance results
    """
    logger.info("Analyzing component specialization...")

    analysis = SpecializationAnalysis(
        digit_preferences={},
        label_preferences={},
        digit_entropy={},
        label_entropy={},
        significant_class_associations={},
    )

    for module_name in stats.activation_counts:
        n_components = len(stats.activation_counts[module_name])

        # Get class activation counts [C, 10]
        class_counts = stats.class_activation_counts[module_name]

        # Normalize to preferences (probabilities)
        row_sums = class_counts.sum(axis=1, keepdims=True)
        preferences = np.divide(
            class_counts,
            row_sums,
            out=np.zeros_like(class_counts, dtype=np.float64),
            where=row_sums > 0,
        )

        # For now, treat original and shuffled the same (we're analyzing the shuffled scenario)
        analysis.digit_preferences[module_name] = preferences.copy()
        analysis.label_preferences[module_name] = preferences.copy()

        # Compute entropy (low = specialized)
        entropy = -np.sum(preferences * np.log(preferences + 1e-8), axis=1)
        analysis.digit_entropy[module_name] = entropy
        analysis.label_entropy[module_name] = entropy

        # Statistical significance testing
        # Null hypothesis: Component activates uniformly across all classes (p = 0.1 for each class)
        # Use z-test for proportion
        significant_associations = []
        total_activations = stats.activation_counts[module_name]  # [C]
        n_tests = n_components * 10  # Bonferroni correction denominator
        bonferroni_alpha = significance_level / n_tests

        for comp_idx in range(n_components):
            if total_activations[comp_idx] < 30:  # Skip components with too few activations
                continue

            p_expected = 0.1

            for class_idx in range(10):
                observed = class_counts[comp_idx, class_idx]

                # Z-test for proportion
                # z = (observed - expected) / sqrt(expected * (1 - p) * n)
                # But we use the exact binomial test instead (more accurate)
                p_value = scipy_stats.binomtest(
                    int(observed),
                    int(total_activations[comp_idx]),
                    p_expected,
                    alternative="greater",
                ).pvalue

                # Apply Bonferroni correction
                if p_value < bonferroni_alpha:
                    # Compute z-score for interpretability
                    expected_val = total_activations[comp_idx] * p_expected
                    std = np.sqrt(total_activations[comp_idx] * p_expected * (1 - p_expected))
                    z_score = (observed - expected_val) / std

                    significant_associations.append(
                        (int(comp_idx), int(class_idx), float(z_score), float(p_value))
                    )

        analysis.significant_class_associations[module_name] = significant_associations
        logger.info(
            f"{module_name}: Found {len(significant_associations)} significant class associations "
            f"(Bonferroni-corrected α={significance_level / n_tests:.2e})"
        )

    return analysis


def main(
    model_path: str,
    shuffled_labels_path: str,
    output_dir: str | Path | None = None,
    batch_size: int = 128,
    ci_threshold: float = 0.1,
    top_k: int = 20,
    device: str | None = None,
):
    """Analyze MNIST memorization decomposition.

    Args:
        model_path: Path to trained ComponentModel checkpoint
        shuffled_labels_path: Path to shuffled labels pickle file
        output_dir: Output directory (defaults to model_dir/analysis)
        batch_size: Batch size for inference
        ci_threshold: CI threshold for component activation
        top_k: Number of top examples to track per component
        device: Device to run on (defaults to cuda if available)
    """
    # Setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if output_dir is None:
        model_dir = Path(model_path).parent
        output_dir = model_dir / "analysis"
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MNIST Memorization Analysis")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Shuffled labels: {shuffled_labels_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"CI threshold: {ci_threshold}")

    # Load model and data
    logger.info("\n" + "=" * 80)
    logger.info("Loading model and data")
    logger.info("=" * 80)
    model, train_dataset, _test_dataset, shuffled_data = load_model_and_data(
        model_path, shuffled_labels_path, device
    )

    # Collect statistics
    logger.info("\n" + "=" * 80)
    logger.info("Collecting component statistics")
    logger.info("=" * 80)
    stats = collect_component_statistics(
        model, train_dataset, device, batch_size, ci_threshold, top_k
    )

    # Analyze specialization
    logger.info("\n" + "=" * 80)
    logger.info("Analyzing specialization")
    logger.info("=" * 80)
    specialization = analyze_specialization(
        stats, shuffled_data["original_labels"], shuffled_data["shuffled_labels"]
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    for module_name in stats.activation_counts:
        n_dead = (stats.activation_counts[module_name] < 10).sum()
        n_alive = (stats.activation_counts[module_name] >= 10).sum()
        mean_activations = stats.activation_counts[module_name][
            stats.activation_counts[module_name] > 0
        ].mean()

        logger.info(f"\n{module_name}:")
        logger.info(f"  Components: {len(stats.activation_counts[module_name])}")
        logger.info(f"  Dead: {n_dead}")
        logger.info(f"  Alive: {n_alive}")
        logger.info(f"  Mean activations (alive): {mean_activations:.1f}")
        logger.info(
            f"  Significant class associations: {len(specialization.significant_class_associations[module_name])}"
        )

        # Print top significant associations
        sig_assocs = specialization.significant_class_associations[module_name]
        if sig_assocs:
            logger.info("  Top 5 most significant:")
            sorted_assocs = sorted(sig_assocs, key=lambda x: x[2], reverse=True)[:5]
            for comp_idx, class_idx, z_score, p_value in sorted_assocs:
                logger.info(
                    f"    Component {comp_idx} -> Class {class_idx}: "
                    f"z={z_score:.2f}, p={p_value:.2e}"
                )

    logger.info("\n" + "=" * 80)
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
