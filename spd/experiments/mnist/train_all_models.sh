#!/bin/bash
# Train MNIST models for multiple hyperparameters with both shuffled and original labels

# Configuration
N_SAMPLES=500
EPOCHS=100
SEED=42
HIDDEN_DIMS=(32 64 128)
LEARNING_RATES=(1e-4 5e-4 1e-3)

# Get SPD_OUT_DIR
if [ -d "/mnt/polished-lake/artifacts/mechanisms/spd" ]; then
    SPD_OUT_DIR="/mnt/polished-lake/artifacts/mechanisms/spd"
else
    SPD_OUT_DIR="$HOME/spd_out"
fi

SHUFFLED_LABELS="$SPD_OUT_DIR/mnist/shuffled_labels_seed${SEED}.pkl"

echo "============================================"
echo "Training MNIST Models - Hyperparameter Grid"
echo "============================================"
echo "Samples: $N_SAMPLES"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo "Hidden dims: ${HIDDEN_DIMS[@]}"
echo "Learning rates: ${LEARNING_RATES[@]}"
echo "Shuffled labels: $SHUFFLED_LABELS"
echo ""

# Check if shuffled labels exist
if [ ! -f "$SHUFFLED_LABELS" ]; then
    echo "ERROR: Shuffled labels not found at $SHUFFLED_LABELS"
    echo "Create them first with:"
    echo "  python -m spd.experiments.mnist.create_shuffled_dataset --seed $SEED --n_samples $N_SAMPLES"
    exit 1
fi

# Train models for each hyperparameter combination
for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        echo "============================================"
        echo "Training: hidden_dim=$hidden_dim, lr=$lr"
        echo "============================================"

        # Train with shuffled labels (memorization)
        echo "Training with SHUFFLED labels..."
        python -m spd.experiments.mnist.train_mnist \
            --label_type shuffled \
            --shuffled_labels_path "$SHUFFLED_LABELS" \
            --hidden_dim $hidden_dim \
            --lr $lr \
            --epochs $EPOCHS \
            --seed $SEED \
            --no_wandb

        # Train with original labels (natural learning)
        echo ""
        echo "Training with ORIGINAL labels..."
        python -m spd.experiments.mnist.train_mnist \
            --label_type original \
            --n_train_samples $N_SAMPLES \
            --hidden_dim $hidden_dim \
            --lr $lr \
            --epochs $EPOCHS \
            --seed $SEED \
            --no_wandb

        echo ""
    done
done

echo "============================================"
echo "All models trained!"
echo "============================================"
echo "Models saved to: $SPD_OUT_DIR/mnist/"
echo ""
echo "To list trained models:"
echo "  ls -lh $SPD_OUT_DIR/mnist/target_model_*.pt"
