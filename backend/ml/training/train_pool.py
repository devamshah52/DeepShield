import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)

from torch.utils.data import DataLoader


# ============================================================
# Backend path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# Imports
# ============================================================

from ml.training.model_pool import (
    DeepShieldCNNPool,
)

from ml.training.audio_dataset import (
    DeepShieldAudioDataset,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EPOCHS = 15

LEARNING_RATE = 0.0003

BATCH_SIZE = 32

NUM_CLASSES = 3

CLASS_NAMES = [
    "Genuine",
    "TTS",
    "Voice Conversion",
]


# ============================================================
# Paths
# ============================================================

# IMPORTANT:
# Experiment 3 uses the combined training manifest
# containing:
#
# Original training samples: 25,580
# VC augmented samples:       3,800
# Total:                     29,380

TRAIN_MANIFEST = (
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "train_augmented_manifest.csv"
)

# Development remains completely unchanged.
#
# This is important because A17/A18/A19 must remain
# completely unseen during training.

DEV_MANIFEST = (
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "dev_manifest.csv"
)


# New model filename so the previous dual-pooling model
# is NOT overwritten.

MODEL_OUTPUT = (
    BACKEND_DIR
    / "ml"
    / "models"
    / "deepshield_cnn_pool_augmented_best.pth"
)


# ============================================================
# Display
# ============================================================

print()
print("=" * 70)
print("DeepShield Dual-Pooling CNN + VC Augmentation Training")
print("=" * 70)

print()

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()

print(
    "Epochs:",
    EPOCHS
)

print(
    "Learning Rate:",
    LEARNING_RATE
)

print(
    "Batch Size:",
    BATCH_SIZE
)

print()


# ============================================================
# Load manifests
# ============================================================

train_df = pd.read_csv(
    TRAIN_MANIFEST
)

dev_df = pd.read_csv(
    DEV_MANIFEST
)


# ============================================================
# Dataset sizes
# ============================================================

print(
    "Training samples:",
    len(train_df)
)

print(
    "Development samples:",
    len(dev_df)
)


# ============================================================
# Training class distribution
# ============================================================

print()
print("Training class distribution:")

print(
    train_df.groupby(
        ["label_id", "label"]
    ).size()
)


# ============================================================
# Development class distribution
# ============================================================

print()
print("Development class distribution:")

print(
    dev_df.groupby(
        ["label_id", "label"]
    ).size()
)


# ============================================================
# Verify expected augmented training size
# ============================================================

EXPECTED_TRAINING_SAMPLES = 29380

if len(train_df) != EXPECTED_TRAINING_SAMPLES:

    raise RuntimeError(
        "Unexpected augmented training manifest size. "
        f"Expected {EXPECTED_TRAINING_SAMPLES}, "
        f"got {len(train_df)}.\n\n"
        "Make sure train_augmented_manifest.csv "
        "contains the original 25,580 samples plus "
        "the 3,800 VC augmented samples."
    )


# ============================================================
# Verify VC augmentation
# ============================================================

augmented_mask = (
    train_df["audio_id"]
    .astype(str)
    .str.startswith("VC_AUG_")
)

augmented_count = int(
    augmented_mask.sum()
)

EXPECTED_AUGMENTED_SAMPLES = 3800

if augmented_count != EXPECTED_AUGMENTED_SAMPLES:

    raise RuntimeError(
        "Unexpected VC augmentation count. "
        f"Expected {EXPECTED_AUGMENTED_SAMPLES}, "
        f"got {augmented_count}."
    )


print()

print(
    "VC augmented training samples:",
    augmented_count
)


# ============================================================
# Verify augmented samples are Voice Conversion
# ============================================================

augmented_labels = (
    train_df.loc[
        augmented_mask,
        "label_id"
    ]
    .unique()
)

if not (
    len(augmented_labels) == 1
    and int(augmented_labels[0]) == 2
):

    raise RuntimeError(
        "VC augmented samples do not all have "
        "label_id=2 (Voice Conversion)."
    )


# ============================================================
# Verify expected original training samples
# ============================================================

original_count = len(train_df) - augmented_count

if original_count != 25580:

    raise RuntimeError(
        "Unexpected original training sample count. "
        f"Expected 25580, got {original_count}."
    )


print(
    "Original training samples:",
    original_count
)

print(
    "Total training samples:",
    len(train_df)
)


# ============================================================
# Class weights
# ============================================================

class_counts = (
    train_df["label_id"]
    .value_counts()
    .sort_index()
)


# Make sure all 3 classes exist.

if len(class_counts) != NUM_CLASSES:

    raise RuntimeError(
        "Training manifest does not contain "
        "all three classes."
    )


total_samples = len(
    train_df
)


class_weights = (
    total_samples
    / (
        NUM_CLASSES
        * class_counts.values
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE,
)


print()
print("Class weights:")

for index, weight in enumerate(
    class_weights
):

    print(
        f"{CLASS_NAMES[index]}: "
        f"{weight.item():.4f}"
    )


# ============================================================
# Datasets
# ============================================================

print()
print(
    "Loading cached training features..."
)

train_dataset = DeepShieldAudioDataset(
    train_df,
    split="train",
)


print(
    "Loading cached development features..."
)

dev_dataset = DeepShieldAudioDataset(
    dev_df,
    split="dev",
)


print(
    "Cached feature datasets ready."
)


# ============================================================
# Explicit dataset verification
# ============================================================

print()
print(
    "Verifying original training feature..."
)

original_features, original_label = (
    train_dataset[0]
)

print(
    "Original feature shape:",
    original_features.shape
)

print(
    "Original label:",
    original_label.item()
)


# Find one augmented sample.

augmented_index = train_df.index[
    augmented_mask
][0]


print()
print(
    "Verifying VC augmented training feature..."
)

augmented_features, augmented_label = (
    train_dataset[
        augmented_index
    ]
)

print(
    "Augmented feature shape:",
    augmented_features.shape
)

print(
    "Augmented label:",
    augmented_label.item()
)

print(
    "Augmented audio ID:",
    train_df.loc[
        augmented_index,
        "audio_id"
    ]
)


# ============================================================
# Safety checks
# ============================================================

if tuple(
    original_features.shape
) != (
    1,
    128,
    251,
):

    raise RuntimeError(
        "Original feature has an unexpected shape."
    )


if tuple(
    augmented_features.shape
) != (
    1,
    128,
    251,
):

    raise RuntimeError(
        "VC augmented feature has an unexpected shape."
    )


if original_label.item() not in (
    0,
    1,
    2,
):

    raise RuntimeError(
        "Invalid original sample label."
    )


if augmented_label.item() != 2:

    raise RuntimeError(
        "VC augmented sample does not have "
        "Voice Conversion label_id=2."
    )


print()
print(
    "Dataset verification successful."
)


# ============================================================
# Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


print()

print(
    "Training batches:",
    len(train_loader)
)

print(
    "Development batches:",
    len(dev_loader)
)


# ============================================================
# Model
# ============================================================

model = DeepShieldCNNPool(
    num_classes=NUM_CLASSES
).to(
    DEVICE
)


print()
print(
    "Fresh dual-pooling model created."
)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Training state
# ============================================================

best_macro_f1 = -1.0

best_epoch = 0


# ============================================================
# Training loop
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1,
):

    print()
    print("=" * 70)

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print("=" * 70)

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    # ========================================================
    # Training batches
    # ========================================================

    for batch_index, (
        features,
        labels,
    ) in enumerate(
        train_loader,
        start=1,
    ):

        features = features.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            features
        )


        loss = criterion(
            outputs,
            labels,
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * labels.size(0)
        )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


        if batch_index % 100 == 0:

            batch_accuracy = (
                correct
                / total
                * 100
            )

            print(
                f"Batch {batch_index}/"
                f"{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: "
                f"{batch_accuracy:.2f}%"
            )


    # ========================================================
    # Training metrics
    # ========================================================

    training_loss = (
        running_loss
        / total
    )


    training_accuracy = (
        correct
        / total
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    validation_loss = 0.0

    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for features, labels in (
            dev_loader
        ):

            features = features.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )


            outputs = model(
                features
            )


            loss = criterion(
                outputs,
                labels,
            )


            validation_loss += (
                loss.item()
                * labels.size(0)
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )


            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )


    validation_loss /= len(
        dev_dataset
    )


    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )


    # ========================================================
    # Validation metrics
    # ========================================================

    validation_accuracy = (
        all_predictions
        == all_labels
    ).mean()


    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            all_predictions,
        )
    )


    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )


    # ========================================================
    # Display metrics
    # ========================================================

    print()
    print("-" * 70)


    print(
        f"Training Loss: "
        f"{training_loss:.4f}"
    )


    print(
        f"Training Accuracy: "
        f"{training_accuracy * 100:.2f}%"
    )


    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )


    print(
        f"Validation Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )


    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )


    print(
        f"Macro F1: "
        f"{macro_f1 * 100:.2f}%"
    )


    # ========================================================
    # Per-class metrics
    # ========================================================

    print()
    print(
        "Per-Class Metrics:"
    )


    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        mask_true = (
            all_labels == class_id
        )


        mask_pred = (
            all_predictions == class_id
        )


        tp = np.sum(
            mask_true
            & mask_pred
        )


        actual = np.sum(
            mask_true
        )


        predicted = np.sum(
            mask_pred
        )


        recall = (
            tp / actual
            if actual > 0
            else 0.0
        )


        precision = (
            tp / predicted
            if predicted > 0
            else 0.0
        )


        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )


        print(
            f"{class_name}: "
            f"Precision={precision * 100:.2f}% | "
            f"Recall={recall * 100:.2f}% | "
            f"F1={f1 * 100:.2f}%"
        )


    # ========================================================
    # Save best model
    # ========================================================

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        best_epoch = epoch


        torch.save(
            model.state_dict(),
            MODEL_OUTPUT,
        )


        print()
        print(
            "New best model saved!"
        )


        print(
            f"Best Macro F1: "
            f"{best_macro_f1 * 100:.2f}%"
        )


        print(
            f"Best Epoch: "
            f"{best_epoch}"
        )


# ============================================================
# Completion
# ============================================================

print()
print("=" * 70)

print(
    "Dual-Pooling + VC Augmentation "
    "Training Completed"
)

print("=" * 70)

print()


print(
    f"Best Macro F1: "
    f"{best_macro_f1 * 100:.2f}%"
)


print(
    "Best Epoch:",
    best_epoch,
)


print(
    "Model:",
    MODEL_OUTPUT,
)


print("=" * 70)