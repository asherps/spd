# MNIST Memorization Experiments

This experiment studies memorization in neural networks using SPD (Stochastic Parameter Decomposition).

## Overview

By shuffling MNIST labels and training a model to memorize the shuffled data, we can study:
- Which components are responsible for storing memorized information
- How memorized vs. learned patterns differ in component activation
- The relationship between memorization and parameter storage capacity

## Architecture

The model is a simple 2-layer MLP:
```
Flatten -> Linear(784, hidden_dim) -> ReLU -> Linear(hidden_dim, 10)
```

We use an MLP instead of a CNN to make SPD decomposition more straightforward.

**Default configuration**: `hidden_dim=32`, `n_train_samples=25000`
- Total parameters: ~25k (784×32 + 32×10)
- Training samples: 25k (~1 sample per parameter)
- This ratio allows for complete memorization

## Usage

### 1. Create Shuffled Dataset (Once)

First, create and save the shuffled MNIST dataset for reproducibility:

```bash
# Activate virtual environment
source .venv/bin/activate

# Create shuffled dataset with seed 42
python -m spd.experiments.mnist.create_shuffled_dataset --seed 42
```

This saves to `~/spd_out/mnist/shuffled_data/shuffled_labels_seed42.pkl` and prints the label mapping statistics.

### 2. Train MNIST Model

Train using the saved shuffled labels:

```bash
# Train with defaults (32 hidden dim, 25k samples)
python -m spd.experiments.mnist.train_mnist \
    --shuffled_labels_path ~/spd_out/mnist/shuffled_data/shuffled_labels_seed42.pkl \
    --epochs 50 \
    --batch_size 128 \
    --seed 42

# Or scale up the model
python -m spd.experiments.mnist.train_mnist \
    --shuffled_labels_path ~/spd_out/mnist/shuffled_data/shuffled_labels_seed42.pkl \
    --hidden_dim 128 \
    --epochs 50 \
    --batch_size 128 \
    --seed 42 \
    --n_train_samples 100000
```

The trained model will be saved to `~/spd_out/mnist/mnist_shuffled_32h_25000samples/` (or with appropriate dimensions).

### 3. Update Config with Model and Labels Paths

Edit `spd/experiments/mnist/mnist_config.yaml` and set:
```yaml
pretrained_model_path: "/path/to/your/checkpoint.pt"
shuffled_labels_path: "~/spd_out/mnist/shuffled_data/shuffled_labels_seed42.pkl"
```

### 3. Run SPD Decomposition

```bash
# Run SPD decomposition locally
python -m spd.experiments.mnist.mnist_decomposition \
    --config_path spd/experiments/mnist/mnist_config.yaml

# Or use the spd-local command
spd-local mnist
```

### 4. Analyze Results

The SPD decomposition will produce:
- Component activation patterns (causal importance values)
- Component activation density (how many components activate per input)
- Histograms showing which components are "alive" vs "dead"

Look for:
- **Sparse vs Dense Components**: Do memorized examples activate more components?
- **Component Specialization**: Do certain components activate only for specific (memorized) digit-label pairs?
- **Activation Density**: How many components are needed to produce memorized outputs?

## Expected Results

- **With defaults (32h, 25k samples)**: Model should achieve ~100% training accuracy showing complete memorization
- **With larger models/datasets**: May require more epochs or may not reach 100% if under-parameterized
- Test accuracy should be ~10% (random chance) showing no generalization
- SPD should reveal which parameter components store the memorized mappings
- Look for components that specialize on specific input-output pairs

**Note**: The default configuration provides ~1 parameter per training sample, sufficient for complete memorization.

## Key Hyperparameters

In `mnist_config.yaml`:
- `module_info.C`: Number of components per layer (200 default - more than needed for sparse solution)
- `loss_metric_configs.ImportanceMinimalityLoss.coeff`: Controls sparsity (1e-4 default)
- `loss_metric_configs.ImportanceMinimalityLoss.pnorm`: p-norm for importance loss (2.0 default)
- `steps`: Training steps for SPD (20,000 default)

## Files

- `models.py`: Simple MLP architecture for MNIST
- `train_mnist.py`: Training script with label shuffling option
- `mnist_decomposition.py`: SPD decomposition script
- `mnist_config.yaml`: SPD configuration
- `README.md`: This file
