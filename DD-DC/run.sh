#!/bin/bash

# ==========================================
# Experiment Configuration
# ==========================================
# NOISE_DIR="./noise-EMN"  # Adjust this if your directory is somewhere else
IPC=100
ITERATION=30

DATASETS=("CIFAR10" "FashionMNIST" "SVHN" "MNIST")
DATASETS=("CIFAR10" "SVHN" "MNIST" "FashionMNIST")
# DATASETS=("MNIST" "CIFAR100") # 
# DATASETS=("SVHN" "MNIST" "CIFAR100")
# DATASETS=("FashionMNIST" "MNIST")
# DATASETS=("SVHN")
NOISE_TYPES=("CW" "SW")
NOISE_TYPES=("SW")

# FashionMNIST: ResNet18BN, ResNet50BN

# SVHN: ResNet18_AP

# ==========================================
# Execution Loop
# ==========================================
for DATASET in "${DATASETS[@]}"; do
    for NOISE in "${NOISE_TYPES[@]}"; do
        

        echo "=================================================================="
        echo "STARTING: Dataset = $DATASET | Noise = $NOISE "
        echo "=================================================================="

        python main.py \
            --dataset "$DATASET" \
            --ipc $IPC \
            --Iteration $ITERATION \
            --add_noise \
            --noise_type "$NOISE"

            
        echo "Finished $DATASET with $NOISE noise."
        echo ""

    done
done

# # # # If you also want to run the clean baselines (without noise) automatically, 
# # # # uncomment the block below:

# echo "=================================================================="
# echo "RUNNING CLEAN BASELINES"
# echo "=================================================================="
# for DATASET in "${DATASETS[@]}"; do
#     python main.py \
#         --dataset "$DATASET" \
#         --ipc $IPC \
#         --Iteration $ITERATION \
#         --num_eval 1 \
#         --epoch_eval_train 300

# done