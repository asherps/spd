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

## Usage

### 1. Train MNIST Model with Shuffled Labels

```bash
# Activate virtual environment
source .venv/bin/activate

# Train with shuffled labels (memorization)
python -m spd.experiments.mnist.train_mnist \
    --shuffle_labels \
    --hidden_dim 128 \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --seed 42

# Or train normally (for comparison)
python -m spd.experiments.mnist.train_mnist \
    --hidden_dim 128 \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --seed 42
```

The trained model will be saved to `~/spd_out/mnist/mnist_shuffled_128h/` (or `mnist_normal_128h`).

### 2. Update Config with Model Path

Edit `spd/experiments/mnist/mnist_config.yaml` and set:
```yaml
pretrained_model_path: "/path/to/your/checkpoint.pt"
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

For shuffled-label MNIST:
- Model should achieve high training accuracy (~99%+) showing successful memorization
- Test accuracy should be ~10% (random chance) showing no generalization
- SPD should reveal which parameter components store the memorized mappings

For normal MNIST:
- Model should achieve high training and test accuracy (~97-99%)
- Component patterns should differ from the memorization case
- Components may represent more generalizable features (edges, curves, etc.)

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
