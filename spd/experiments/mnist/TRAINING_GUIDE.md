# MNIST Memorization Experiment Guide

Compare SPD decomposition of memorized vs. naturally learned models.

## Quick Start

```bash
# 1. Create shuffled labels
python -m spd.experiments.mnist.create_shuffled_dataset --seed 42 --n_samples 500

# 2. Train all models (18 total: 3 hidden dims × 3 LRs × 2 label types)
cd spd/experiments/mnist && ./train_all_models.sh

# 3. Update mnist_config.yaml with target model path
# pretrained_model_path: "/path/to/target_model_shuffled_h32_lr0.0001.pt"

# 4. Run SPD decomposition
python -m spd.experiments.mnist.mnist_decomposition --config_path spd/experiments/mnist/mnist_config.yaml
```

## Training Individual Models

**Shuffled labels (memorization):**
```bash
python -m spd.experiments.mnist.train_mnist \
  --label_type shuffled \
  --shuffled_labels_path ~/spd_out/mnist/shuffled_labels_seed42.pkl \
  --hidden_dim 32 --lr 1e-4 --epochs 30 --seed 42
```

**Original labels (natural learning):**
```bash
python -m spd.experiments.mnist.train_mnist \
  --label_type original \
  --n_train_samples 500 \
  --hidden_dim 32 --lr 1e-4 --epochs 15 --seed 42
```

## Expected Results

**Shuffled (memorization):**
- Train: ~100%, Test: ~10% (random chance)
- Components specialize on specific examples

**Original (natural):**
- Train: ~95-100%, Test: ~95-100% (generalizes)
- Components specialize on digit features

## Output Structure

```
~/spd_out/mnist/
├── shuffled_labels_seed42.pkl
├── target_model_{label_type}_h{dim}_lr{lr}.pt
└── run_001/
    ├── model_*.pth
    └── final_config.yaml
```
