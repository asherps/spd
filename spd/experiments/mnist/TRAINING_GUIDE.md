# MNIST Memorization Experiment Guide

This guide explains how to train MNIST models with shuffled vs. original labels across multiple hyperparameters, then run SPD decomposition to compare how the models learn.

## Overview

The experiment compares:
- **Memorization**: Models trained on shuffled labels (rote memorization)
- **Natural Learning**: Models trained on original labels (generalizable features)

For each hyperparameter setting, we train BOTH versions to see how SPD decomposes memorized vs. naturally learned representations.

## Pipeline

### 1. Create Shuffled Labels

```bash
python -m spd.experiments.mnist.create_shuffled_dataset \
  --seed 42 \
  --n_samples 500
```

This creates: `~/spd_out/mnist/shuffled_labels_seed42.pkl`

### 2. Train Target Models

#### Option A: Train all hyperparameter combinations (recommended)

```bash
cd spd/experiments/mnist
./train_all_models.sh
```

This trains models for:
- Hidden dims: [32, 64, 128]
- Learning rates: [1e-4, 5e-4, 1e-3]
- Label types: [shuffled, original]

Total: 18 models (3 hidden dims × 3 LRs × 2 label types)

Models saved as: `target_model_{label_type}_h{hidden_dim}_lr{lr}.pt`

#### Option B: Train individual models

**Shuffled labels (memorization):**
```bash
python -m spd.experiments.mnist.train_mnist \
  --label_type shuffled \
  --shuffled_labels_path ~/spd_out/mnist/shuffled_labels_seed42.pkl \
  --hidden_dim 32 \
  --lr 1e-4 \
  --epochs 100 \
  --seed 42
```

**Original labels (natural learning):**
```bash
python -m spd.experiments.mnist.train_mnist \
  --label_type original \
  --n_train_samples 500 \
  --hidden_dim 32 \
  --lr 1e-4 \
  --epochs 100 \
  --seed 42
```

### 3. Update Config for Target Model

Edit `mnist_config.yaml` to point to the model you want to decompose:

```yaml
pretrained_model_path: "/root/spd_out/mnist/target_model_shuffled_h32_lr0.0001.pt"
# or
pretrained_model_path: "/root/spd_out/mnist/target_model_original_h32_lr0.0001.pt"
```

### 4. Run SPD Decomposition

```bash
python -m spd.experiments.mnist.mnist_decomposition \
  --config_path spd/experiments/mnist/mnist_config.yaml
```

This creates: `~/spd_out/mnist/run_001/` (or next available number)

You can override hyperparameters:
```bash
python -m spd.experiments.mnist.mnist_decomposition \
  --config_path spd/experiments/mnist/mnist_config.yaml \
  --steps 10000 \
  --lr 5e-4
```

### 5. Analyze Results

```bash
python -m spd.experiments.mnist.analyze_mnist \
  --model_path ~/spd_out/mnist/run_001/model_20000.pth \
  --shuffled_labels_path ~/spd_out/mnist/shuffled_labels_seed42.pkl \
  --n_train_samples 500
```

Analysis saved to: `~/spd_out/mnist/run_001/analysis/`

## Directory Structure

```
~/spd_out/mnist/
├── shuffled_labels_seed42.pkl           # Shuffled labels
├── target_model_shuffled_h32_lr0.0001.pt   # Memorization model
├── target_model_original_h32_lr0.0001.pt   # Natural learning model
├── target_model_shuffled_h64_lr0.0005.pt
├── target_model_original_h64_lr0.0005.pt
├── ...
├── run_001/                              # First SPD run
│   ├── model_5000.pth
│   ├── model_10000.pth
│   ├── model_15000.pth
│   ├── model_20000.pth
│   ├── final_config.yaml
│   └── analysis/
├── run_002/                              # Second SPD run
│   └── ...
└── run_003/
    └── ...
```

## Key Config Features

The `mnist_config.yaml` includes:

1. **1000 components per layer** - More capacity to find specialized components
2. **Shared MLP for CI functions** - More efficient than per-component MLPs
3. **FaithfulnessLoss (coeff=5)** - Ensures components stay faithful to target
4. **ImportanceMinimalityLoss** - Encourages sparse component activations
   - P-norm annealing: 2.0 → 0.5 (promotes sparsity)
5. **StochasticReconSubsetLoss** - Stochastic masking for robustness
6. **PGDReconSubsetLoss** - Adversarial masking for robustness
7. **ComponentWeightSparsityLoss (NEW)** - L1 penalty on component weights
   - Encourages components to have fewer non-zero parameters

## Expected Results

**Shuffled labels (memorization):**
- Train accuracy: ~100%
- Test accuracy: ~10% (random chance)
- Components should specialize on specific examples or small groups

**Original labels (natural learning):**
- Train accuracy: ~95-100%
- Test accuracy: ~95-100% (generalizes)
- Components should specialize on digits or digit features

## Comparing Runs

To compare how SPD decomposes memorized vs. naturally learned models:

1. Train both model types with same hyperparameters
2. Run SPD on each (creates `run_XXX/` directories)
3. Compare:
   - Number of alive components
   - Component specialization patterns (via `analyze_mnist.py`)
   - CI distributions
   - Component activation densities

## Tips

- Start with `hidden_dim=32, lr=1e-4` for initial experiments
- Check WandB for training curves and metrics
- Use `--no_wandb` flag for faster training without logging
- Adjust `ComponentWeightSparsityLoss` coefficient if components are too dense
