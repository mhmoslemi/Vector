#!/bin/bash
# =============================================================================
# make_noise.sh — Generate unlearnable-example perturbations
#
# Usage:
#   DATASET=CIFAR10 PERTURB_TYPE=classwise bash make_noise.sh
#   or edit the variables below and run: bash make_noise.sh
# =============================================================================

# ── Core settings ─────────────────────────────────────────────────────────────
DATASET="${DATASET:-CIFAR10_S}"          # CIFAR10 | CIFAR100 | SVHN | MNIST |
                                        # FashionMNIST
                                       # ImageNetMini | TinyImageNet | CIFAR10_S
PERTURB_TYPE="${PERTURB_TYPE:-samplewise}"   # classwise | samplewise
ATTACK_TYPE="${ATTACK_TYPE:-min-min}"       # min-min | min-max | random

# ── Perturbation hyper-parameters ─────────────────────────────────────────────
EPSILON="${EPSILON:-8}"          # L-inf budget (will be divided by 255 in code)
STEP_SIZE="${STEP_SIZE:-0.8}"    # PGD step size (will be divided by 255)
TRAIN_STEP="${TRAIN_STEP:-10}"   # Inner model-training steps (min-min only)

# classwise defaults differ from samplewise — set per-type below
if [ "$PERTURB_TYPE" == "classwise" ]; then
    NUM_STEPS="${NUM_STEPS:-1}"       # classwise uses fewer PGD steps
    STOP_ERROR="${STOP_ERROR:-0.1}"   # classwise stop error (original doc default)
    UNIVERSAL_TRAIN_TARGET="${UNIVERSAL_TRAIN_TARGET:-train_subset}"
    USE_SUBSET="${USE_SUBSET:-true}"
else
    NUM_STEPS="${NUM_STEPS:-20}"
    STOP_ERROR="${STOP_ERROR:-0.01}"
fi

# ── Misc ───────────────────────────────────────────────────────────────────────
MODEL_VERSION="${MODEL_VERSION:-resnet18}"
DATA_PATH="${DATA_PATH:-../datasets}"

# ── CIFAR10_S-specific settings ────────────────────────────────────────────────
SKEW_RATIO="${SKEW_RATIO:-0.9}"      # Bias ratio: fraction of majority group
SEVERITY="${SEVERITY:-0}"            # Minority-group corruption level (0-5)
NUM_WORKERS="${NUM_WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SUBSET_PT="${SUBSET_PT:-}"   # optional path to DGC6.py subset .pt file

# =============================================================================
# Per-dataset configuration
#   Sets: CONFIG_PATH  CHANNELS  IMG_SIZE  NUM_CLASSES  TRAIN_SIZE
# =============================================================================
case "$DATASET" in
    CIFAR10)
        CONFIG_PATH="configs/cifar10"
        CHANNELS=3;  IMG_SIZE=32
        NUM_CLASSES=10;  TRAIN_SIZE=50000
        ;;
    CIFAR100)
        CONFIG_PATH="configs/cifar100"
        CHANNELS=3;  IMG_SIZE=32
        NUM_CLASSES=100;  TRAIN_SIZE=50000
        ;;
    SVHN)
        CONFIG_PATH="configs/svhn"
        CHANNELS=3;  IMG_SIZE=32
        NUM_CLASSES=10;  TRAIN_SIZE=73257
        ;;
    MNIST)
        CONFIG_PATH="configs/mnist"
        CHANNELS=1;  IMG_SIZE=28
        NUM_CLASSES=10;  TRAIN_SIZE=60000
        ;;
    FashionMNIST)
        CONFIG_PATH="configs/fashion-mnist"
        CHANNELS=1;  IMG_SIZE=28
        NUM_CLASSES=10;  TRAIN_SIZE=60000
        ;;
    ImageNetMini)
        CONFIG_PATH="configs/imagenet-mini"
        CHANNELS=3;  IMG_SIZE=32
        NUM_CLASSES=100
        # ImageNetMini size is filesystem-dependent; set TRAIN_SIZE manually
        # before running, or export TRAIN_SIZE=<your_count>
        TRAIN_SIZE="${TRAIN_SIZE:-130000}"
        echo "[warn] ImageNetMini TRAIN_SIZE assumed ${TRAIN_SIZE}. Override with TRAIN_SIZE=<n>."
        ;;
    TinyImageNet)
        CONFIG_PATH="configs/tiny-imagenet"
        CHANNELS=3;  IMG_SIZE=32
        NUM_CLASSES=200;  TRAIN_SIZE=100000
        ;;
    CIFAR10_S)
        CONFIG_PATH="configs/cifar10"
        CHANNELS=3;  IMG_SIZE=32
        NUM_CLASSES=10;  TRAIN_SIZE=50000
        ;;
    *)
        echo "[error] Unknown DATASET='${DATASET}'."
        echo "        Choose from: CIFAR10 CIFAR100 SVHN MNIST FashionMNIST ImageNetMini TinyImageNet CIFAR10_S"
        exit 1
        ;;
esac

# =============================================================================
# Derive noise shape from perturbation type
#   classwise  → one noise vector per class:  [NUM_CLASSES, C, H, W]
#   samplewise → one noise vector per sample: [TRAIN_SIZE,  C, H, W]
# =============================================================================
if [ "$PERTURB_TYPE" == "classwise" ]; then
    NOISE_SHAPE="${NUM_CLASSES} ${CHANNELS} ${IMG_SIZE} ${IMG_SIZE}"
elif [ "$PERTURB_TYPE" == "samplewise" ]; then
    # When a subset .pt is provided, the Python side overrides noise_shape[0] automatically.
    # Pass full TRAIN_SIZE here; it will be corrected after subset indices are loaded.
    NOISE_SHAPE="${TRAIN_SIZE} ${CHANNELS} ${IMG_SIZE} ${IMG_SIZE}"
else
    echo "[error] Unknown PERTURB_TYPE='${PERTURB_TYPE}'. Choose: classwise | samplewise"
    exit 1
fi

# =============================================================================
# Output directory
# =============================================================================
# EXP_DIR="experiments/${DATASET}_${PERTURB_TYPE}_${ATTACK_TYPE}"
EXP_DIR="/home/mmoslem3/scratch/UE-DD/partial"
mkdir -p "${EXP_DIR}"

# =============================================================================
# Launch
# =============================================================================
echo "============================================================"
echo " Dataset       : ${DATASET}"
echo " Perturb type  : ${PERTURB_TYPE}"
echo " Attack type   : ${ATTACK_TYPE}"
echo " Noise shape   : [${NOISE_SHAPE}]"
echo " Config        : ${CONFIG_PATH}/${MODEL_VERSION}.yaml"
echo " Output        : ${EXP_DIR}/"
echo "============================================================"

python3 perturbation.py \
    --config_path           "${CONFIG_PATH}"    \
    --exp_name              "${EXP_DIR}"        \
    --version               "${MODEL_VERSION}"  \
    --train_data_type       "${DATASET}"        \
    --test_data_type        "${DATASET}"        \
    --train_data_path       "${DATA_PATH}"      \
    --test_data_path        "${DATA_PATH}"      \
    --num_of_workers        "${NUM_WORKERS}"    \
    --train_batch_size      "${BATCH_SIZE}"     \
    --eval_batch_size       "${BATCH_SIZE}"     \
    --noise_shape           ${NOISE_SHAPE}      \
    --epsilon               "${EPSILON}"        \
    --num_steps             "${NUM_STEPS}"      \
    --step_size             "${STEP_SIZE}"      \
    --train_step            "${TRAIN_STEP}"     \
    --attack_type           "${ATTACK_TYPE}"    \
    --perturb_type          "${PERTURB_TYPE}"   \
    --universal_stop_error  "${STOP_ERROR}"     \
    ${UNIVERSAL_TRAIN_TARGET:+--universal_train_target "${UNIVERSAL_TRAIN_TARGET}"} \
    ${USE_SUBSET:+--use_subset} \
    ${SUBSET_PT:+--subset_pt "${SUBSET_PT}"} \
    --skew_ratio                "${SKEW_RATIO}"     \
    --severity                  "${SEVERITY}"
