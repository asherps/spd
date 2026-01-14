#!/bin/bash
# Sweep over learning rates for SPD decomposition training

set -e

# Default parameters
CONFIG_PATH=${CONFIG_PATH:-spd/experiments/mnist/mnist_config.yaml}
STEPS=${STEPS:-5000}

# Learning rates to sweep
LRS=(0.0001 0.0005 0.001 0.005 0.01)

echo "================================================================================"
echo "SPD Decomposition Learning Rate Sweep"
echo "================================================================================"
echo "Config: $CONFIG_PATH"
echo "Steps: $STEPS"
echo "Learning rates: ${LRS[@]}"
echo ""

# Run decomposition for each LR
for lr in "${LRS[@]}"; do
    echo "================================================================================"
    echo "Running SPD decomposition with LR=$lr"
    echo "================================================================================"
    python -m spd.experiments.mnist.mnist_decomposition \
        --config_path $CONFIG_PATH \
        --steps $STEPS \
        --lr $lr
    echo ""
done

echo "================================================================================"
echo "Sweep complete!"
echo "================================================================================"
echo "Check ~/spd_out/mnist/run_XXX/ directories for results"
echo "Compare metrics across runs to find best learning rate."
