"""Run SPD on MNIST memorization model."""

import json
from pathlib import Path

import fire
import torch
import wandb
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from spd.configs import Config, MNISTTaskConfig
from spd.experiments.mnist.create_shuffled_dataset import load_shuffled_labels
from spd.experiments.mnist.models import load_pretrained_mnist_model
from spd.log import logger
from spd.run_spd import optimize
from spd.settings import SPD_OUT_DIR
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import save_pre_run_info, set_seed
from spd.utils.wandb_utils import init_wandb


class MNISTDatasetWrapper(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Wrapper for MNIST dataset that works with SPD's DatasetGeneratedDataLoader."""

    def __init__(self, mnist_dataset: datasets.MNIST, device: str):
        self.mnist_dataset = mnist_dataset
        self.device = device

    def __len__(self) -> int:
        return len(self.mnist_dataset)

    def generate_batch(self, batch_size: int):
        """Generate a batch of data.

        Returns:
            Tuple of (inputs, labels) where inputs are flattened MNIST images
        """
        # Get random batch from MNIST dataset
        indices = torch.randint(0, len(self.mnist_dataset), (batch_size,))
        images = []
        labels = []

        for idx in indices:
            img, label = self.mnist_dataset[int(idx)]
            images.append(img)
            labels.append(label)

        images = torch.stack(images).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        return images, labels


def evaluate_model(
    model: torch.nn.Module,
    dataset: datasets.MNIST,
    device: str,
    n_samples: int = 1000,
) -> float:
    """Evaluate model accuracy on a dataset.

    Args:
        model: Model to evaluate
        dataset: MNIST dataset
        device: Device to run on
        n_samples: Number of samples to evaluate on

    Returns:
        Accuracy as a percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            img, label = dataset[i]
            img = img.unsqueeze(0).to(device)
            label_tensor = torch.tensor([label], dtype=torch.long, device=device)

            output = model(img)
            pred = output.argmax(dim=1)
            correct += pred.eq(label_tensor).sum().item()
            total += 1

    return 100.0 * correct / total


def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
    steps: int | None = None,
) -> None:
    """Run SPD decomposition on MNIST model.

    Args:
        config_path: Path to config YAML file
        config_json: JSON string of config (alternative to config_path)
        evals_id: Optional evaluation ID
        sweep_id: Optional sweep ID
        sweep_params_json: JSON string of sweep parameters
        steps: Number of training steps (overrides config if provided)
    """
    assert (config_path is not None) != (config_json is not None), (
        "Need exactly one of config_path and config_json"
    )

    if config_path is not None:
        config = Config.from_file(config_path)
    else:
        assert config_json is not None
        config = Config(**json.loads(config_json.removeprefix("json:")))

    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    # Override steps if provided
    if steps is not None:
        config.steps = steps
        logger.info(f"Overriding config steps with: {steps}")

    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(config.seed)

    # Create output directory with run counter
    mnist_dir = Path(SPD_OUT_DIR) / "mnist"
    mnist_dir.mkdir(parents=True, exist_ok=True)

    # Find next run number
    existing_runs = [d for d in mnist_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if existing_runs:
        run_numbers = []
        for run_dir in existing_runs:
            try:
                num = int(run_dir.name.split("_")[1])
                run_numbers.append(num)
            except (IndexError, ValueError):
                continue
        next_num = max(run_numbers) + 1 if run_numbers else 1
    else:
        next_num = 1

    run_id = f"run_{next_num:03d}"
    out_dir = mnist_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = [i for i in ["mnist", evals_id, sweep_id] if i is not None]
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {out_dir}")

    if config.wandb_project:
        init_wandb(
            config=config,
            project=config.wandb_project,
            run_id=run_id,
            name=config.wandb_run_name,
            tags=tags,
        )

    logger.info(config)

    # Load target model
    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_model = load_pretrained_mnist_model(config.pretrained_model_path, device=device)
    target_model.eval()

    # Load MNIST datasets
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Load shuffled labels for train set if provided
    assert isinstance(config.task_config, MNISTTaskConfig)
    shuffled_labels_path = config.task_config.shuffled_labels_path
    if shuffled_labels_path:
        logger.info(f"Loading shuffled labels from: {shuffled_labels_path}")
        shuffled_data = load_shuffled_labels(shuffled_labels_path)
        n_train_samples = len(shuffled_data["shuffled_labels"])  # pyright: ignore[reportArgumentType]

        # Subset data and apply shuffled labels
        train_dataset.data = train_dataset.data[:n_train_samples]
        train_dataset.targets = shuffled_data["shuffled_labels"].tolist()  # pyright: ignore[reportAttributeAccessIssue]

        logger.info(
            f"Loaded {n_train_samples} shuffled labels (seed: {shuffled_data['seed']})"  # pyright: ignore[reportIndexIssue]
        )
    else:
        logger.warning("No shuffled_labels_path provided - using original MNIST labels")

    # Evaluate model on train and test sets
    logger.info("Evaluating target model...")
    train_acc = evaluate_model(target_model, train_dataset, device, n_samples=1000)
    test_acc = evaluate_model(target_model, test_dataset, device, n_samples=1000)
    logger.info(f"Target model - Train accuracy: {train_acc:.2f}%")
    logger.info(f"Target model - Test accuracy: {test_acc:.2f}%")
    logger.info(
        "Note: High train + low test accuracy indicates successful memorization of shuffled labels"
    )

    # Save pre-run info
    save_pre_run_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        spd_config=config,
        sweep_params=sweep_params,
        target_model=target_model,
        train_config=None,  # MNIST doesn't use SPD's config format for training
        task_name="mnist_memorization",
    )

    # Wrap dataset for SPD
    dataset = MNISTDatasetWrapper(train_dataset, device=device)
    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    # Run SPD optimization
    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        tied_weights=None,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
