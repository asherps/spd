#!/bin/bash
# Sweep over learning rates for MNIST target model training

set -e

# Default parameters
HIDDEN_DIM=${HIDDEN_DIM:-128}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-42}
SHUFFLED_LABELS_PATH=${SHUFFLED_LABELS_PATH:-~/spd_out/mnist/shuffled_labels_seed42.pkl}

# Learning rates to sweep
LRS=(0.0001 0.0005 0.001 0.005 0.01)

echo "================================================================================"
echo "MNIST Learning Rate Sweep"
echo "================================================================================"
echo "Hidden dim: $HIDDEN_DIM"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo "Shuffled labels: $SHUFFLED_LABELS_PATH"
echo "Learning rates: ${LRS[@]}"
echo ""

# Train models
for lr in "${LRS[@]}"; do
    echo "================================================================================"
    echo "Training with LR=$lr"
    echo "================================================================================"
    python -m spd.experiments.mnist.train_mnist \
        --hidden_dim $HIDDEN_DIM \
        --epochs $EPOCHS \
        --lr $lr \
        --seed $SEED \
        --shuffled_labels_path $SHUFFLED_LABELS_PATH \
        --no_wandb

    # Rename model to include LR in filename
    if [ -f ~/spd_out/mnist/target_model.pt ]; then
        mv ~/spd_out/mnist/target_model.pt ~/spd_out/mnist/target_model_lr${lr}.pt
        echo "Saved model to: ~/spd_out/mnist/target_model_lr${lr}.pt"
    fi
    echo ""
done

echo "================================================================================"
echo "Sweep complete!"
echo "================================================================================"
echo "Models saved with different LRs:"
for lr in "${LRS[@]}"; do
    echo "  ~/spd_out/mnist/target_model_lr${lr}.pt"
done
echo ""
echo "To compare results, check the training output above for final train/test accuracies."
