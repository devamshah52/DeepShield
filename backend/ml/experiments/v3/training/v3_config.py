"""
DeepShield V3 - Training Configuration

All V3 training settings are centralized here so that
the training script remains clean and reproducible.

V3 classes:
    0 = Genuine
    1 = TTS
    2 = Voice_Conversion
"""

from __future__ import annotations

from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    r"C:\DeepShield"
)

BACKEND_DIR = (
    PROJECT_ROOT
    / "backend"
)

V3_DIR = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3"
)


# ============================================================
# Dataset paths
# ============================================================

DATASET_DIR = (
    V3_DIR
    / "dataset"
)

MANIFEST_DIR = (
    DATASET_DIR
    / "manifests"
)

FEATURE_DIR = (
    V3_DIR
    / "features"
)


TRAIN_MANIFEST = (
    MANIFEST_DIR
    / "v3_train.csv"
)

DEV_MANIFEST = (
    MANIFEST_DIR
    / "v3_dev.csv"
)

TEST_MANIFEST = (
    MANIFEST_DIR
    / "v3_test.csv"
)


# ============================================================
# Model paths
# ============================================================

MODEL_DIR = (
    V3_DIR
    / "model"
)

CHECKPOINT_DIR = (
    V3_DIR
    / "checkpoints"
)


# ============================================================
# Logging paths
# ============================================================

LOG_DIR = (
    V3_DIR
    / "logs"
)


# ============================================================
# Dataset configuration
# ============================================================

NUM_CLASSES = 3

CLASS_NAMES = {
    0: "Genuine",
    1: "TTS",
    2: "Voice_Conversion",
}

FEATURE_HEIGHT = 128
FEATURE_WIDTH = 251
INPUT_CHANNELS = 1


# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42


# ============================================================
# Training configuration
# ============================================================

# Batch size chosen for the RTX 5060 Laptop GPU.
BATCH_SIZE = 32

# Initial learning rate.
LEARNING_RATE = 0.0005

# AdamW regularization.
WEIGHT_DECAY = 0.0001

# Maximum number of training epochs.
EPOCHS = 30


# ============================================================
# Optimizer
# ============================================================

OPTIMIZER_NAME = "AdamW"


# ============================================================
# Learning-rate scheduler
# ============================================================

SCHEDULER_NAME = "CosineAnnealingLR"

# Minimum learning rate reached by the scheduler.
MIN_LEARNING_RATE = 1e-6


# ============================================================
# Early stopping
# ============================================================

# Number of epochs allowed without improvement before
# stopping training.
EARLY_STOPPING_PATIENCE = 7


# ============================================================
# Gradient handling
# ============================================================

# Prevent unusually large gradients from destabilizing
# training.
GRADIENT_CLIP_MAX_NORM = 5.0


# ============================================================
# DataLoader configuration
# ============================================================

NUM_WORKERS = 0

PIN_MEMORY = True

DROP_LAST_TRAIN_BATCH = False


# ============================================================
# Mixed precision
# ============================================================

USE_AMP = True


# ============================================================
# Checkpointing
# ============================================================

BEST_CHECKPOINT_NAME = (
    "deepshield_v3_best.pth"
)

LAST_CHECKPOINT_NAME = (
    "deepshield_v3_last.pth"
)

TRAINING_HISTORY_NAME = (
    "v3_training_history.csv"
)


# ============================================================
# Validation / monitoring
# ============================================================

# Primary metric used for selecting the best checkpoint.
PRIMARY_METRIC = "balanced_accuracy"


# ============================================================
# Device
# ============================================================

DEVICE = "cuda"


# ============================================================
# Configuration validation
# ============================================================

def validate_config():
    """
    Validate important V3 training settings before training.
    """

    if NUM_CLASSES != 3:

        raise ValueError(
            "V3 must use exactly 3 classes."
        )

    if BATCH_SIZE <= 0:

        raise ValueError(
            "BATCH_SIZE must be greater than zero."
        )

    if LEARNING_RATE <= 0:

        raise ValueError(
            "LEARNING_RATE must be greater than zero."
        )

    if WEIGHT_DECAY < 0:

        raise ValueError(
            "WEIGHT_DECAY cannot be negative."
        )

    if EPOCHS <= 0:

        raise ValueError(
            "EPOCHS must be greater than zero."
        )

    if EARLY_STOPPING_PATIENCE <= 0:

        raise ValueError(
            "EARLY_STOPPING_PATIENCE must be "
            "greater than zero."
        )

    if GRADIENT_CLIP_MAX_NORM <= 0:

        raise ValueError(
            "GRADIENT_CLIP_MAX_NORM must be "
            "greater than zero."
        )

    if FEATURE_HEIGHT != 128:

        raise ValueError(
            "Unexpected feature height."
        )

    if FEATURE_WIDTH != 251:

        raise ValueError(
            "Unexpected feature width."
        )

    if INPUT_CHANNELS != 1:

        raise ValueError(
            "V3 expects exactly one input channel."
        )


# ============================================================
# Create required directories
# ============================================================

def create_directories():

    directories = [
        MODEL_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# Print configuration
# ============================================================

def print_config():

    print()
    print("=" * 70)
    print("DeepShield V3 - Training Configuration")
    print("=" * 70)

    print()
    print("Dataset")
    print("-" * 70)

    print(
        f"Train manifest : {TRAIN_MANIFEST}"
    )

    print(
        f"Dev manifest   : {DEV_MANIFEST}"
    )

    print(
        f"Test manifest  : {TEST_MANIFEST}"
    )

    print(
        f"Feature dir    : {FEATURE_DIR}"
    )

    print()
    print("Classes")
    print("-" * 70)

    for label_id in range(
        NUM_CLASSES
    ):

        print(
            f"  {label_id}: "
            f"{CLASS_NAMES[label_id]}"
        )

    print()
    print("Model Input")
    print("-" * 70)

    print(
        f"Channels : {INPUT_CHANNELS}"
    )

    print(
        f"Height   : {FEATURE_HEIGHT}"
    )

    print(
        f"Width    : {FEATURE_WIDTH}"
    )

    print()
    print("Training")
    print("-" * 70)

    print(
        f"Batch size              : {BATCH_SIZE}"
    )

    print(
        f"Learning rate            : {LEARNING_RATE}"
    )

    print(
        f"Weight decay             : {WEIGHT_DECAY}"
    )

    print(
        f"Epochs                   : {EPOCHS}"
    )

    print(
        f"Optimizer                : {OPTIMIZER_NAME}"
    )

    print(
        f"Scheduler                : {SCHEDULER_NAME}"
    )

    print(
        f"Minimum learning rate    : "
        f"{MIN_LEARNING_RATE}"
    )

    print(
        f"Early stopping patience : "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Gradient clipping        : "
        f"{GRADIENT_CLIP_MAX_NORM}"
    )

    print(
        f"Mixed precision (AMP)    : {USE_AMP}"
    )

    print()
    print("Reproducibility")
    print("-" * 70)

    print(
        f"Random seed : {RANDOM_SEED}"
    )

    print()
    print("DataLoader")
    print("-" * 70)

    print(
        f"Workers     : {NUM_WORKERS}"
    )

    print(
        f"Pin memory  : {PIN_MEMORY}"
    )

    print()
    print("Device")
    print("-" * 70)

    print(
        f"Device      : {DEVICE}"
    )

    print()
    print(
        f"Primary metric: {PRIMARY_METRIC}"
    )

    print("=" * 70)


# ============================================================
# Main configuration test
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DeepShield V3 - Configuration Test")
    print("=" * 70)

    validate_config()

    create_directories()

    print_config()

    print()
    print(
        "V3 CONFIGURATION TEST: PASS"
    )


if __name__ == "__main__":

    main()