from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

BACKEND_DIR = PROJECT_ROOT / "backend"

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

V32_DIR = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3_1"
)


# ============================================================
# V3.2 dataset
# ============================================================

V32_DATASET_DIR = (
    PROJECT_ROOT
    / "v3_2_dataset"
)

V32_TRAIN_MANIFEST = (
    V32_DATASET_DIR
    / "v3_2_train_manifest.csv"
)


# IMPORTANT:
# Development data remains the original V3 development set.
# V3 test data is NOT used during training.

V3_DEV_MANIFEST = (
    V3_DIR
    / "dataset"
    / "manifests"
    / "v3_dev.csv"
)

V3_TEST_MANIFEST = (
    V3_DIR
    / "dataset"
    / "manifests"
    / "v3_test.csv"
)


# ============================================================
# Feature directories
# ============================================================

V3_FEATURE_DIR = (
    V3_DIR
    / "features"
)

V32_GTTS_FEATURE_DIR = (
    V32_DATASET_DIR
    / "gtts_features"
)


# ============================================================
# Model
# ============================================================

MODEL_FILE = (
    V3_DIR
    / "model"
    / "v3_model.py"
)

MODEL_CLASS_NAME = "DeepShieldV3"


# ============================================================
# V3.2 output directories
# ============================================================

CHECKPOINT_DIR = (
    V31_DIR
    / "checkpoints"
)

LOG_DIR = (
    V31_DIR
    / "logs"
)


# ============================================================
# V3.2 checkpoint names
# ============================================================

BEST_CHECKPOINT_NAME = (
    "deepshield_v3_2_best.pth"
)

LAST_CHECKPOINT_NAME = (
    "deepshield_v3_2_last.pth"
)

TRAINING_HISTORY_NAME = (
    "v3_2_training_history.csv"
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

EXPECTED_TRAIN_SAMPLES = 9024

EXPECTED_TRAIN_CLASS_COUNTS = {
    0: 3608,
    1: 3608,
    2: 1808,
}

EXPECTED_DEV_SAMPLES = 1158

EXPECTED_DEV_CLASS_COUNTS = {
    0: 386,
    1: 386,
    2: 386,
}


# ============================================================
# Feature configuration
# ============================================================

FEATURE_HEIGHT = 128
FEATURE_WIDTH = 251
INPUT_CHANNELS = 1

EXPECTED_FEATURE_SHAPE = (
    FEATURE_HEIGHT,
    FEATURE_WIDTH,
)


# ============================================================
# Audio preprocessing
# ============================================================

SAMPLE_RATE = 16000

AUDIO_DURATION_SECONDS = 4.0

N_FFT = 1024

HOP_LENGTH = 256

N_MELS = 128

POWER = 2.0


# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42


# ============================================================
# Training configuration
# ============================================================

BATCH_SIZE = 32

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0001

EPOCHS = 30


# ============================================================
# Optimizer
# ============================================================

OPTIMIZER_NAME = "AdamW"


# ============================================================
# Scheduler
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
# DataLoader
# ============================================================

NUM_WORKERS = 0

PIN_MEMORY = True

DROP_LAST_TRAIN_BATCH = False


# ============================================================
# Mixed precision
# ============================================================

USE_AMP = True


# ============================================================
# Primary metric
# ============================================================

PRIMARY_METRIC = "balanced_accuracy"


# ============================================================
# Device
# ============================================================

DEVICE = "cuda"


# ============================================================
# Class weighting
# ============================================================

# We are NOT using class weights in this first V3.2 run.
#
# The purpose of this experiment is to measure the effect of
# adding gTTS data without introducing another training variable.

USE_CLASS_WEIGHTS = False


# ============================================================
# Validation
# ============================================================

def validate_config():

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    if not V32_TRAIN_MANIFEST.exists():
        raise FileNotFoundError(
            f"V3.2 training manifest not found:\n"
            f"{V32_TRAIN_MANIFEST}"
        )

    if not V3_DEV_MANIFEST.exists():
        raise FileNotFoundError(
            f"V3 development manifest not found:\n"
            f"{V3_DEV_MANIFEST}"
        )

    if not V3_TEST_MANIFEST.exists():
        raise FileNotFoundError(
            f"V3 test manifest not found:\n"
            f"{V3_TEST_MANIFEST}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"V3 model file not found:\n"
            f"{MODEL_FILE}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    if NUM_CLASSES != 3:
        raise ValueError(
            "V3.2 must use exactly 3 classes."
        )

    if EXPECTED_TRAIN_SAMPLES != 9024:
        raise ValueError(
            "Unexpected V3.2 training sample count."
        )

    if sum(EXPECTED_TRAIN_CLASS_COUNTS.values()) != (
        EXPECTED_TRAIN_SAMPLES
    ):
        raise ValueError(
            "V3.2 class counts do not sum to total samples."
        )

    if sum(EXPECTED_DEV_CLASS_COUNTS.values()) != (
        EXPECTED_DEV_SAMPLES
    ):
        raise ValueError(
            "V3 development class counts do not sum "
            "to total samples."
        )

    # --------------------------------------------------------
    # Feature shape
    # --------------------------------------------------------

    if EXPECTED_FEATURE_SHAPE != (
        128,
        251,
    ):
        raise ValueError(
            "Unexpected feature shape."
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    if BATCH_SIZE <= 0:
        raise ValueError(
            "BATCH_SIZE must be positive."
        )

    if LEARNING_RATE <= 0:
        raise ValueError(
            "LEARNING_RATE must be positive."
        )

    if EPOCHS <= 0:
        raise ValueError(
            "EPOCHS must be positive."
        )

    if EARLY_STOPPING_PATIENCE <= 0:
        raise ValueError(
            "EARLY_STOPPING_PATIENCE must be positive."
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if DEVICE not in {
        "cuda",
        "cpu",
    }:
        raise ValueError(
            f"Unsupported device: {DEVICE}"
        )

    # --------------------------------------------------------
    # Primary metric
    # --------------------------------------------------------

    if PRIMARY_METRIC != "balanced_accuracy":
        raise ValueError(
            "V3.2 primary metric must be "
            "'balanced_accuracy'."
        )

    return True


# ============================================================
# Directory creation
# ============================================================

def create_directories():

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Configuration display
# ============================================================

def print_config():

    print("=" * 70)
    print("DeepShield V3.2 Configuration")
    print("=" * 70)

    print()

    print("Project:")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  V3_DIR:       {V3_DIR}")
    print(f"  V31_DIR:      {V31_DIR}")

    print()

    print("Dataset:")
    print(f"  Train manifest: {V32_TRAIN_MANIFEST}")
    print(f"  Dev manifest:   {V3_DEV_MANIFEST}")
    print(f"  Test manifest:  {V3_TEST_MANIFEST}")

    print()

    print("Dataset sizes:")
    print(f"  Train: {EXPECTED_TRAIN_SAMPLES}")
    print(f"  Dev:   {EXPECTED_DEV_SAMPLES}")
    print("  Test:  NOT USED DURING TRAINING")

    print()

    print("Train class counts:")

    for label_id, name in CLASS_NAMES.items():

        print(
            f"  {label_id}: "
            f"{name:<18} "
            f"{EXPECTED_TRAIN_CLASS_COUNTS[label_id]}"
        )

    print()

    print("Dev class counts:")

    for label_id, name in CLASS_NAMES.items():

        print(
            f"  {label_id}: "
            f"{name:<18} "
            f"{EXPECTED_DEV_CLASS_COUNTS[label_id]}"
        )

    print()

    print("Features:")
    print(
        f"  Shape: "
        f"{EXPECTED_FEATURE_SHAPE}"
    )

    print(
        f"  Sample rate: "
        f"{SAMPLE_RATE}"
    )

    print(
        f"  Duration: "
        f"{AUDIO_DURATION_SECONDS}s"
    )

    print()

    print("Training:")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  Weight decay:    {WEIGHT_DECAY}")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  Optimizer:       {OPTIMIZER_NAME}")
    print(f"  Scheduler:       {SCHEDULER_NAME}")
    print(
        f"  Min LR:          {MIN_LEARNING_RATE}"
    )
    print(
        f"  Early stopping:  "
        f"{EARLY_STOPPING_PATIENCE}"
    )
    print(
        f"  Gradient clip:   "
        f"{GRADIENT_CLIP_MAX_NORM}"
    )

    print()

    print("Performance:")
    print(f"  Device:          {DEVICE}")
    print(f"  AMP:             {USE_AMP}")
    print(f"  Num workers:     {NUM_WORKERS}")
    print(f"  Pin memory:      {PIN_MEMORY}")

    print()

    print("Class weighting:")
    print(
        f"  Enabled:         {USE_CLASS_WEIGHTS}"
    )

    print()

    print("Output:")
    print(
        f"  Best checkpoint: "
        f"{CHECKPOINT_DIR / BEST_CHECKPOINT_NAME}"
    )

    print(
        f"  Last checkpoint: "
        f"{CHECKPOINT_DIR / LAST_CHECKPOINT_NAME}"
    )

    print(
        f"  History: "
        f"{LOG_DIR / TRAINING_HISTORY_NAME}"
    )

    print()

    print("=" * 70)


# ============================================================
# Direct execution
# ============================================================

if __name__ == "__main__":

    validate_config()
    create_directories()
    print_config()

    print()
    print(
        "V3.2 CONFIGURATION TEST: PASS"
    )