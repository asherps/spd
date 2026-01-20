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
EPOCHS_SHUFFLED=30
EPOCHS_ORIGINAL=15
SEED=42
HIDDEN_DIMS=(32 64 128)
LEARNING_RATES=(1e-4 5e-4 1e-3)

SPD_OUT_DIR="${SPD_OUT_DIR:-$HOME/spd_out}"
SHUFFLED_LABELS="$SPD_OUT_DIR/mnist/shuffled_labels_seed${SEED}.pkl"

echo "Training MNIST Models - Hyperparameter Grid"
echo "Config: ${#HIDDEN_DIMS[@]} hidden dims × ${#LEARNING_RATES[@]} LRs × 2 label types"

[ -f "$SHUFFLED_LABELS" ] || {
    echo "ERROR: Shuffled labels not found. Create with:"
    echo "  python -m spd.experiments.mnist.create_shuffled_dataset --seed $SEED --n_samples $N_SAMPLES"
    exit 1
}

for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        echo ""
        echo "Training h=$hidden_dim, lr=$lr"

        for label_type in shuffled original; do
            epochs=$([[ $label_type == "shuffled" ]] && echo $EPOCHS_SHUFFLED || echo $EPOCHS_ORIGINAL)
            extra_args=""
            [[ $label_type == "shuffled" ]] && extra_args="--shuffled_labels_path $SHUFFLED_LABELS" || extra_args="--n_train_samples $N_SAMPLES"

            python -m spd.experiments.mnist.train_mnist \
                --label_type $label_type \
                $extra_args \
                --hidden_dim $hidden_dim \
                --lr $lr \
                --epochs $epochs \
                --seed $SEED \
                --no_wandb
        done
    done
done

echo ""
echo "All models trained! See: $SPD_OUT_DIR/mnist/target_model_*.pt"
