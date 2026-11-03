
"""
DeepShield V2-A Training
========================

Controlled architecture experiment for DeepShield.

V2-A uses:
    - SAME 29,380 training samples as V1
    - SAME development set as V1
    - NEW DeepShield V2 architecture
    - Training-time SpecAugment
    - Same class-weighted CrossEntropyLoss
    - AdamW optimizer
    - ReduceLROnPlateau scheduler
    - Early stopping
    - Reproducible seed
    - Detailed validation metrics

IMPORTANT:
    This script NEVER overwrites the V1 production model.

V1 model:
    C:\\DeepShield\\backend\\ml\\models\\
    deepshield_cnn_pool_augmented_best.pth

V2 checkpoints:
    C:\\DeepShield\\backend\\ml\\experiments\\v2\\checkpoints\\
"""

import sys
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
# ============================================================

# Current file:
#
# C:\DeepShield\backend\ml\experiments\v2\training\train_v2.py
#
# parents[0] = training
# parents[1] = v2
# parents[2] = experiments
# parents[3] = ml
# parents[4] = backend

BACKEND_DIR = Path(
    __file__
).resolve().parents[4]

PROJECT_DIR = BACKEND_DIR.parent

ML_DIR = BACKEND_DIR / "ml"


# ============================================================
# MANIFESTS
# ============================================================

TRAIN_MANIFEST = (
    ML_DIR
    / "dataset"
    / "train_augmented_manifest.csv"
)

DEV_MANIFEST = (
    ML_DIR
    / "dataset"
    / "dev_manifest.csv"
)


# ============================================================
# V2 DIRECTORIES
# ============================================================

V2_DIR = (
    ML_DIR
    / "experiments"
    / "v2"
)

CHECKPOINT_DIR = (
    V2_DIR
    / "checkpoints"
)

LOG_DIR = (
    V2_DIR
    / "logs"
)

V2_MODEL_DIR = (
    V2_DIR
    / "model"
)


# ============================================================
# OUTPUT FILES
# ============================================================

BEST_MODEL = (
    CHECKPOINT_DIR
    / "deepshield_v2_a_best.pth"
)

LAST_MODEL = (
    CHECKPOINT_DIR
    / "deepshield_v2_a_last.pth"
)

HISTORY_FILE = (
    LOG_DIR
    / "training_history.json"
)

CSV_LOG_FILE = (
    LOG_DIR
    / "training_log.csv"
)


# ============================================================
# PYTHON PATH
# ============================================================

if str(BACKEND_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(BACKEND_DIR)
    )


if str(V2_MODEL_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(V2_MODEL_DIR)
    )


# ============================================================
# IMPORT EXISTING DATASET
# ============================================================

# IMPORTANT:
#
# The actual DeepShield dataset class is:
#
#     DeepShieldAudioDataset
#
# It expects:
#
#     dataframe
#     split
#
# It does NOT expect:
#
#     manifest_path
#
# Therefore we use the exact interface already used
# by the existing V1 training pipeline.

from ml.training.audio_dataset import (
    DeepShieldAudioDataset,
)


# ============================================================
# IMPORT NEW V2 MODEL
# ============================================================

from model_v2 import DeepShieldV2


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_CLASSES = 3

CLASS_NAMES = [
    "Genuine",
    "TTS",
    "Voice Conversion",
]


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 30

BATCH_SIZE = 32

LEARNING_RATE = 0.0003

WEIGHT_DECAY = 0.0001

NUM_WORKERS = 0

PATIENCE = 7

GRADIENT_CLIP = 5.0


# ============================================================
# CLASS WEIGHTS
# ============================================================

# Same weights used by the existing V1 training.

CLASS_WEIGHTS = torch.tensor(
    [
        3.5228,
        0.6443,
        0.8591,
    ],
    dtype=torch.float32,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifests():

    print()
    print("Loading manifest files...")
    print("-" * 70)

    if not TRAIN_MANIFEST.exists():

        raise FileNotFoundError(
            "Training manifest not found:\n"
            f"{TRAIN_MANIFEST}"
        )

    if not DEV_MANIFEST.exists():

        raise FileNotFoundError(
            "Development manifest not found:\n"
            f"{DEV_MANIFEST}"
        )

    train_df = pd.read_csv(
        TRAIN_MANIFEST
    )

    dev_df = pd.read_csv(
        DEV_MANIFEST
    )

    print(
        f"Training samples: "
        f"{len(train_df):,}"
    )

    print(
        f"Development samples: "
        f"{len(dev_df):,}"
    )

    # --------------------------------------------------------
    # Expected V1 training size
    # --------------------------------------------------------

    if len(train_df) != 29380:

        print()
        print(
            "WARNING:"
        )

        print(
            "Training manifest does not contain "
            "the expected 29,380 samples."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = {
        "audio_id",
        "label_id",
        "label",
    }

    missing_train = (
        required_columns
        - set(train_df.columns)
    )

    missing_dev = (
        required_columns
        - set(dev_df.columns)
    )

    if missing_train:

        raise RuntimeError(
            "Training manifest is missing "
            f"required columns: {missing_train}"
        )

    if missing_dev:

        raise RuntimeError(
            "Development manifest is missing "
            f"required columns: {missing_dev}"
        )

    # --------------------------------------------------------
    # Training distribution
    # --------------------------------------------------------

    print()
    print(
        "Training class distribution:"
    )

    print(
        train_df.groupby(
            ["label_id", "label"]
        ).size()
    )

    # --------------------------------------------------------
    # Development distribution
    # --------------------------------------------------------

    print()
    print(
        "Development class distribution:"
    )

    print(
        dev_df.groupby(
            ["label_id", "label"]
        ).size()
    )

    return (
        train_df,
        dev_df,
    )


# ============================================================
# SPECAUGMENT
# ============================================================

def spec_augment(
    spectrogram,
    freq_mask_param=12,
    time_mask_param=20,
    num_freq_masks=2,
    num_time_masks=2,
):
    """
    Mild SpecAugment.

    Input:
        [B, 1, 128, 251]

    Frequency and time regions are masked only during
    training.

    Validation features remain untouched.
    """

    augmented = spectrogram.clone()

    batch_size = (
        augmented.size(0)
    )

    n_mels = (
        augmented.size(2)
    )

    time_steps = (
        augmented.size(3)
    )

    for b in range(batch_size):

        # ----------------------------------------------------
        # Frequency masking
        # ----------------------------------------------------

        for _ in range(
            num_freq_masks
        ):

            width = random.randint(
                0,
                min(
                    freq_mask_param,
                    n_mels
                )
            )

            if width > 0:

                start = random.randint(
                    0,
                    n_mels - width
                )

                augmented[
                    b,
                    :,
                    start:start + width,
                    :
                ] = 0.0

        # ----------------------------------------------------
        # Time masking
        # ----------------------------------------------------

        for _ in range(
            num_time_masks
        ):

            width = random.randint(
                0,
                min(
                    time_mask_param,
                    time_steps
                )
            )

            if width > 0:

                start = random.randint(
                    0,
                    time_steps - width
                )

                augmented[
                    b,
                    :,
                    :,
                    start:start + width
                ] = 0.0

    return augmented


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def create_dataloaders(
    train_df,
    dev_df,
):

    print()
    print(
        "Loading cached training features..."
    )

    # EXACT SAME DATASET INTERFACE AS V1
    train_dataset = (
        DeepShieldAudioDataset(
            train_df,
            split="train",
        )
    )

    print(
        "Loading cached development features..."
    )

    # EXACT SAME DATASET INTERFACE AS V1
    dev_dataset = (
        DeepShieldAudioDataset(
            dev_df,
            split="dev",
        )
    )

    print()
    print(
        "Cached feature datasets ready."
    )

    print(
        f"Train dataset: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Dev dataset: "
        f"{len(dev_dataset):,}"
    )

    # --------------------------------------------------------
    # Training loader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Development loader
    # --------------------------------------------------------

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_dataset,
        dev_dataset,
        train_loader,
        dev_loader,
    )


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_dataset(
    train_dataset,
):

    print()
    print(
        "Verifying one training feature..."
    )

    print("-" * 70)

    features, label = (
        train_dataset[0]
    )

    print(
        f"Feature shape : "
        f"{tuple(features.shape)}"
    )

    print(
        f"Label         : "
        f"{label}"
    )

    expected_shape = (
        1,
        128,
        251
    )

    actual_shape = tuple(
        features.shape
    )

    if actual_shape != expected_shape:

        raise RuntimeError(
            "Unexpected feature shape. "
            f"Expected {expected_shape}, "
            f"got {actual_shape}"
        )

    print(
        "Feature shape verification: OK"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_info(
    model
):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Model information:"
    )

    print("-" * 70)

    print(
        "Architecture       : "
        "DeepShield V2-A"
    )

    print(
        f"Total parameters   : "
        f"{total_params:,}"
    )

    print(
        f"Trainable params   : "
        f"{trainable_params:,}"
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_targets = []

    for features, targets in loader:

        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        features = features.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # SpecAugment
        # ----------------------------------------------------

        features = spec_augment(
            features
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(
            features
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            targets
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * features.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_targets.extend(
            targets.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    epoch_accuracy = (
        accuracy_score(
            all_targets,
            all_predictions
        )
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_targets = []

    for features, targets in loader:

        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        features = features.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # NO SpecAugment during validation
        # ----------------------------------------------------

        outputs = model(
            features
        )

        loss = criterion(
            outputs,
            targets
        )

        running_loss += (
            loss.item()
            * features.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
        )

        all_targets.extend(
            targets.cpu()
            .numpy()
        )

    validation_loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = (
        accuracy_score(
            all_targets,
            all_predictions
        )
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_targets,
            all_predictions
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    macro_f1 = float(
        np.mean(f1)
    )

    cm = confusion_matrix(
        all_targets,
        all_predictions,
        labels=[0, 1, 2],
    )

    metrics = {

        "loss":
            float(validation_loss),

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(balanced_accuracy),

        "macro_f1":
            macro_f1,

        "precision":
            [
                float(x)
                for x in precision
            ],

        "recall":
            [
                float(x)
                for x in recall
            ],

        "f1":
            [
                float(x)
                for x in f1
            ],

        "support":
            [
                int(x)
                for x in support
            ],

        "confusion_matrix":
            cm.tolist(),
    }

    return metrics


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
):

    checkpoint = {

        "version":
            "DeepShield V2-A",

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "metrics":
            metrics,

        "class_names":
            CLASS_NAMES,

        "config": {

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "seed":
                SEED,

            "specaugment":
                True,
        },
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(
    history,
    best_epoch,
    best_macro_f1,
):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "model":
                    "DeepShield V2-A",

                "best_epoch":
                    best_epoch,

                "best_macro_f1":
                    best_macro_f1,

                "history":
                    history,
            },
            f,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        SEED
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 70)

    print(
        "DeepShield V2-A Training"
    )

    print("=" * 70)

    print()

    print(
        "This is a NEW experimental model."
    )

    print(
        "The existing V1 model will NOT be modified."
    )

    print()

    print(
        f"Project directory : "
        f"{PROJECT_DIR}"
    )

    print(
        f"Backend directory : "
        f"{BACKEND_DIR}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()

    print(
        f"Device             : "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            "GPU                : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "CUDA version       : "
            f"{torch.version.cuda}"
        )

    else:

        print()
        print(
            "WARNING:"
        )

        print(
            "CUDA is not available."
        )

        print(
            "Training will use CPU."
        )

    # --------------------------------------------------------
    # Load manifests
    # --------------------------------------------------------

    (
        train_df,
        dev_df,
    ) = load_manifests()

    # --------------------------------------------------------
    # Create datasets/loaders
    # --------------------------------------------------------

    (
        train_dataset,
        dev_dataset,
        train_loader,
        dev_loader,
    ) = create_dataloaders(
        train_df,
        dev_df,
    )

    # --------------------------------------------------------
    # Verify dataset
    # --------------------------------------------------------

    verify_dataset(
        train_dataset
    )

    # --------------------------------------------------------
    # Create V2 model
    # --------------------------------------------------------

    model = DeepShieldV2(
        num_classes=NUM_CLASSES
    ).to(device)

    print_model_info(
        model
    )

    # --------------------------------------------------------
    # Test batch
    # --------------------------------------------------------

    print()

    print(
        "Testing one training batch..."
    )

    print("-" * 70)

    test_features, test_targets = next(
        iter(train_loader)
    )

    print(
        "Input batch shape  : "
        f"{tuple(test_features.shape)}"
    )

    print(
        "Target batch shape : "
        f"{tuple(test_targets.shape)}"
    )

    test_features_device = (
        test_features.to(
            device
        )
    )

    with torch.no_grad():

        test_output = model(
            test_features_device
        )

    print(
        "Output batch shape : "
        f"{tuple(test_output.shape)}"
    )

    # --------------------------------------------------------
    # Criterion
    # --------------------------------------------------------

    class_weights = (
        CLASS_WEIGHTS.to(
            device
        )
    )

    criterion = (
        nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    history = []

    best_macro_f1 = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    start_time = time.time()

    # --------------------------------------------------------
    # CSV header
    # --------------------------------------------------------

    with open(
        CSV_LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([

            "epoch",

            "learning_rate",

            "train_loss",

            "train_accuracy",

            "val_loss",

            "val_accuracy",

            "val_balanced_accuracy",

            "val_macro_f1",

            "genuine_precision",

            "genuine_recall",

            "genuine_f1",

            "tts_precision",

            "tts_recall",

            "tts_f1",

            "vc_precision",

            "vc_recall",

            "vc_f1",
        ])

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = (
            time.time()
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_metrics = validate(
            model,
            dev_loader,
            criterion,
            device,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            validation_metrics[
                "macro_f1"
            ]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print()

        print("=" * 70)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print("-" * 70)

        print(
            "Train Loss       : "
            f"{train_loss:.4f}"
        )

        print(
            "Train Accuracy   : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            "Val Loss         : "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            "Val Accuracy     : "
            f"{validation_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            "Balanced Accuracy: "
            f"{validation_metrics['balanced_accuracy'] * 100:.2f}%"
        )

        print(
            "Macro F1         : "
            f"{validation_metrics['macro_f1'] * 100:.2f}%"
        )

        print()

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            print(
                f"{class_name:<18} "
                f"Precision: "
                f"{validation_metrics['precision'][i] * 100:6.2f}%  "
                f"Recall: "
                f"{validation_metrics['recall'][i] * 100:6.2f}%  "
                f"F1: "
                f"{validation_metrics['f1'][i] * 100:6.2f}%"
            )

        print()

        print(
            "Learning Rate    : "
            f"{current_lr:.8f}"
        )

        print(
            "Epoch Time       : "
            f"{epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Record history
        # ----------------------------------------------------

        epoch_record = {

            "epoch":
                epoch,

            "learning_rate":
                float(current_lr),

            "train_loss":
                float(train_loss),

            "train_accuracy":
                float(train_accuracy),

            "validation":
                validation_metrics,

            "epoch_time_seconds":
                float(epoch_time),
        }

        history.append(
            epoch_record
        )

        # ----------------------------------------------------
        # Write CSV
        # ----------------------------------------------------

        with open(
            CSV_LOG_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([

                epoch,

                current_lr,

                train_loss,

                train_accuracy,

                validation_metrics[
                    "loss"
                ],

                validation_metrics[
                    "accuracy"
                ],

                validation_metrics[
                    "balanced_accuracy"
                ],

                validation_metrics[
                    "macro_f1"
                ],

                validation_metrics[
                    "precision"
                ][0],

                validation_metrics[
                    "recall"
                ][0],

                validation_metrics[
                    "f1"
                ][0],

                validation_metrics[
                    "precision"
                ][1],

                validation_metrics[
                    "recall"
                ][1],

                validation_metrics[
                    "f1"
                ][1],

                validation_metrics[
                    "precision"
                ][2],

                validation_metrics[
                    "recall"
                ][2],

                validation_metrics[
                    "f1"
                ][2],
            ])

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        current_macro_f1 = (
            validation_metrics[
                "macro_f1"
            ]
        )

        if (
            current_macro_f1
            > best_macro_f1
        ):

            best_macro_f1 = (
                current_macro_f1
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                BEST_MODEL,
                model,
                optimizer,
                scheduler,
                epoch,
                validation_metrics,
            )

            print()

            print(
                "*** NEW BEST MODEL SAVED "
                f"(Macro F1 = "
                f"{best_macro_f1 * 100:.2f}%) ***"
            )

        else:

            epochs_without_improvement += 1

            print()

            print(
                "No improvement for "
                f"{epochs_without_improvement} "
                "epoch(s)."
            )

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            LAST_MODEL,
            model,
            optimizer,
            scheduler,
            epoch,
            validation_metrics,
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        save_history(
            history,
            best_epoch,
            best_macro_f1,
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # COMPLETE
    # ========================================================

    total_time = (
        time.time()
        - start_time
    )

    print()

    print("=" * 70)

    print(
        "DeepShield V2-A Training Complete"
    )

    print("=" * 70)

    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )

    print(
        "Best Macro F1    : "
        f"{best_macro_f1 * 100:.2f}%"
    )

    print(
        "Training time    : "
        f"{total_time / 60:.1f} minutes"
    )

    print()

    print(
        "Best checkpoint:"
    )

    print(
        BEST_MODEL
    )

    print()

    print(
        "Last checkpoint:"
    )

    print(
        LAST_MODEL
    )

    print()

    print(
        "Training history:"
    )

    print(
        HISTORY_FILE
    )

    print()

    print(
        "CSV training log:"
    )

    print(
        CSV_LOG_FILE
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The existing DeepShield V1 production "
        "model was NOT modified."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
