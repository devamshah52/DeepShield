"""
DeepShield V3.1 - Training Configuration

V3.1 keeps the same DeepShield V3 CNN architecture and training
philosophy while using the expanded V3.1 training dataset.

V3.1 classes:
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

V31_DIR = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3_1"
)


# ============================================================
# Dataset paths
# ============================================================

DATASET_DIR = (
    V31_DIR
    / "dataset"
)

FEATURE_DIR = (
    V31_DIR
    / "features"
)

V31_TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "v3_1_dataset_inspection"
    / "v3_1_train_manifest.csv"
)

# V3 dev is intentionally reused unchanged.
V3_DEV_MANIFEST = (
    V3_DIR
    / "dataset"
    / "manifests"
    / "v3_dev.csv"
)


# ============================================================
# Model
# ============================================================

MODEL_DIR = (
    V3_DIR
    / "model"
)

MODEL_FILE = (
    MODEL_DIR
    / "v3_model.py"
)


# ============================================================
# V3.1 checkpoints
# ============================================================

CHECKPOINT_DIR = (
    V31_DIR
    / "checkpoints"
)

BEST_CHECKPOINT_NAME = (
    "deepshield_v3_1_best.pth"
)

LAST_CHECKPOINT_NAME = (
    "deepshield_v3_1_last.pth"
)


# ============================================================
# V3.1 logging
# ============================================================

LOG_DIR = (
    V31_DIR
    / "logs"
)

TRAINING_HISTORY_NAME = (
    "v3_1_training_history.csv"
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
# Expected V3.1 training distribution
# ============================================================

EXPECTED_TRAIN_SAMPLES = 7224

EXPECTED_GENUINE_SAMPLES = 3608
EXPECTED_TTS_SAMPLES = 1808
EXPECTED_VOICE_CONVERSION_SAMPLES = 1808


# ============================================================
# Expected validation distribution
# ============================================================

EXPECTED_DEV_SAMPLES = 1158

EXPECTED_DEV_GENUINE = 386
EXPECTED_DEV_TTS = 386
EXPECTED_DEV_VOICE_CONVERSION = 386


# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42


# ============================================================
# Training configuration
# ============================================================

# Same batch size as V3.
BATCH_SIZE = 32

# Same initial learning rate as V3.
LEARNING_RATE = 0.0005

# Same AdamW regularization as V3.
WEIGHT_DECAY = 0.0001

# Same maximum number of epochs as V3.
EPOCHS = 30


# ============================================================
# Optimizer
# ============================================================

OPTIMIZER_NAME = "AdamW"


# ============================================================
# Learning-rate scheduler
# ============================================================

SCHEDULER_NAME = "CosineAnnealingLR"

MIN_LEARNING_RATE = 1e-6


# ============================================================
# Early stopping
# ============================================================

EARLY_STOPPING_PATIENCE = 7


# ============================================================
# Gradient handling
# ============================================================

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
# Validation / monitoring
# ============================================================

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
    Validate important V3.1 training settings.
    """

    if NUM_CLASSES != 3:

        raise ValueError(
            "V3.1 must use exactly 3 classes."
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
            "V3.1 expects exactly one input channel."
        )

    if EXPECTED_TRAIN_SAMPLES != 7224:

        raise ValueError(
            "Unexpected V3.1 training sample count."
        )

    if EXPECTED_DEV_SAMPLES != 1158:

        raise ValueError(
            "Unexpected V3 development sample count."
        )


# ============================================================
# Create required directories
# ============================================================

def create_directories():

    directories = [
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
    print("DeepShield V3.1 - Training Configuration")
    print("=" * 70)

    print()
    print("Dataset")
    print("-" * 70)

    print(
        f"V3.1 train manifest : "
        f"{V31_TRAIN_MANIFEST}"
    )

    print(
        f"V3 dev manifest    : "
        f"{V3_DEV_MANIFEST}"
    )

    print(
        f"Feature directory   : "
        f"{FEATURE_DIR}"
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
    print("Expected Training Data")
    print("-" * 70)

    print(
        f"Total samples       : "
        f"{EXPECTED_TRAIN_SAMPLES}"
    )

    print(
        f"Genuine             : "
        f"{EXPECTED_GENUINE_SAMPLES}"
    )

    print(
        f"TTS                 : "
        f"{EXPECTED_TTS_SAMPLES}"
    )

    print(
        f"Voice Conversion    : "
        f"{EXPECTED_VOICE_CONVERSION_SAMPLES}"
    )

    print()
    print("Expected Development Data")
    print("-" * 70)

    print(
        f"Total samples       : "
        f"{EXPECTED_DEV_SAMPLES}"
    )

    print(
        f"Genuine             : "
        f"{EXPECTED_DEV_GENUINE}"
    )

    print(
        f"TTS                 : "
        f"{EXPECTED_DEV_TTS}"
    )

    print(
        f"Voice Conversion    : "
        f"{EXPECTED_DEV_VOICE_CONVERSION}"
    )

    print()
    print("Model")
    print("-" * 70)

    print(
        f"Architecture : DeepShieldV3"
    )

    print(
        f"Initialization: RANDOM V3 WEIGHTS"
    )

    print(
        f"Model file   : {MODEL_FILE}"
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
    print("Checkpointing")
    print("-" * 70)

    print(
        f"Best : "
        f"{CHECKPOINT_DIR / BEST_CHECKPOINT_NAME}"
    )

    print(
        f"Last : "
        f"{CHECKPOINT_DIR / LAST_CHECKPOINT_NAME}"
    )

    print(
        f"History : "
        f"{LOG_DIR / TRAINING_HISTORY_NAME}"
    )

    print()
    print(
        f"Primary metric: {PRIMARY_METRIC}"
    )

    print()
    print(
        "Test set: NOT USED DURING TRAINING"
    )

    print("=" * 70)


# ============================================================
# Main configuration test
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DeepShield V3.1 - Configuration Test")
    print("=" * 70)

    validate_config()

    create_directories()

    print_config()

    print()
    print(
        "V3.1 CONFIGURATION TEST: PASS"
    )


if __name__ == "__main__":

    main()