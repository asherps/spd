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
EPOCHS_ORIGINAL=8     # Natural learning converges fast
SEED=42
HIDDEN_DIMS=(32 64 128)
LR_SHUFFLED=1e-4      # Fixed LR for memorization
LRS_ORIGINAL=(1e-4 5e-4 1e-3)  # Sweep LRs for natural learning

SPD_OUT_DIR="${SPD_OUT_DIR:-$HOME/spd_out}"
SHUFFLED_LABELS="$SPD_OUT_DIR/mnist/shuffled_labels_seed${SEED}.pkl"

echo "Training MNIST Models"
echo "Shuffled: ${#HIDDEN_DIMS[@]} hidden dims (fixed LR=$LR_SHUFFLED)"
echo "Original: ${#HIDDEN_DIMS[@]} hidden dims × ${#LRS_ORIGINAL[@]} LRs"

[ -f "$SHUFFLED_LABELS" ] || {
    echo "ERROR: Shuffled labels not found. Create with:"
    echo "  python -m spd.experiments.mnist.create_shuffled_dataset --seed $SEED --n_samples $N_SAMPLES"
    exit 1
}

# Train memorization models (fixed LR, more epochs)
for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    echo ""
    echo "Training SHUFFLED: h=$hidden_dim, lr=$LR_SHUFFLED, epochs=$EPOCHS_SHUFFLED"
    python -m spd.experiments.mnist.train_mnist \
        --label_type shuffled \
        --shuffled_labels_path "$SHUFFLED_LABELS" \
        --hidden_dim $hidden_dim \
        --lr $LR_SHUFFLED \
        --epochs $EPOCHS_SHUFFLED \
        --seed $SEED \
        --no_wandb
done

# Train natural learning models (sweep LRs, fewer epochs)
for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    for lr in "${LRS_ORIGINAL[@]}"; do
        echo ""
        echo "Training ORIGINAL: h=$hidden_dim, lr=$lr, epochs=$EPOCHS_ORIGINAL"
        python -m spd.experiments.mnist.train_mnist \
            --label_type original \
            --n_train_samples $N_SAMPLES \
            --hidden_dim $hidden_dim \
            --lr $lr \
            --epochs $EPOCHS_ORIGINAL \
            --seed $SEED \
            --no_wandb
    done
done

echo ""
echo "All models trained! See: $SPD_OUT_DIR/mnist/target_model_*.pt"
