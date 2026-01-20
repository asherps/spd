#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

[ -f "$REPO_ROOT/.venv/bin/activate" ] && source "$REPO_ROOT/.venv/bin/activate" || {
    echo "ERROR: Virtual environment not found. Run 'make install-dev' first"
    exit 1
}

cd "$REPO_ROOT"

N_SAMPLES=500
EPOCHS_SHUFFLED=60    # Memorization needs more epochs
EPOCHS_ORIGINAL=12    # Natural learning with regularization
SEED=42
HIDDEN_DIM=128        # Fixed hidden dimension
LR_SHUFFLED=1e-4      # Fixed LR for memorization
LRS_ORIGINAL=(1e-4 5e-4 1e-3)  # Sweep LRs for natural learning

SHUFFLED_LABELS="$SCRIPT_DIR/datasets/shuffled_labels_seed${SEED}.pkl"

echo "Training MNIST Models (h=$HIDDEN_DIM)"
echo "Shuffled: 1 model (fixed LR=$LR_SHUFFLED)"
echo "Original: ${#LRS_ORIGINAL[@]} models (sweep LRs)"

[ -f "$SHUFFLED_LABELS" ] || {
    echo "ERROR: Shuffled labels not found. Create with:"
    echo "  python -m spd.experiments.mnist.create_shuffled_dataset --seed $SEED --n_samples $N_SAMPLES"
    exit 1
}

# Train memorization model (fixed LR, more epochs)
echo ""
echo "Training SHUFFLED: h=$HIDDEN_DIM, lr=$LR_SHUFFLED, epochs=$EPOCHS_SHUFFLED"
python -m spd.experiments.mnist.train_mnist \
    --label_type shuffled \
    --shuffled_labels_path "$SHUFFLED_LABELS" \
    --hidden_dim $HIDDEN_DIM \
    --lr $LR_SHUFFLED \
    --epochs $EPOCHS_SHUFFLED \
    --seed $SEED \
    --no_wandb

# Train natural learning models (sweep LRs, fewer epochs)
for lr in "${LRS_ORIGINAL[@]}"; do
    echo ""
    echo "Training ORIGINAL: h=$HIDDEN_DIM, lr=$lr, epochs=$EPOCHS_ORIGINAL"
    python -m spd.experiments.mnist.train_mnist \
        --label_type original \
        --n_train_samples $N_SAMPLES \
        --hidden_dim $HIDDEN_DIM \
        --lr $lr \
        --epochs $EPOCHS_ORIGINAL \
        --seed $SEED \
        --no_wandb
done

echo ""
echo "All models trained! See: $SCRIPT_DIR/models/target_model_*.pt"
