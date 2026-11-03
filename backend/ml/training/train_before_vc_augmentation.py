import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)


# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from ml.training.audio_dataset import (
    DeepShieldAudioDataset,
)

from ml.training.model import (
    DeepShieldCNN,
)


# ============================================================
# Configuration
# ============================================================

TRAIN_MANIFEST = (
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "train_manifest.csv"
)


DEV_MANIFEST = (
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "dev_manifest.csv"
)


MODEL_DIR = (
    BACKEND_DIR
    / "ml"
    / "models"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HISTORY_DIR = (
    BACKEND_DIR
    / "ml"
    / "training"
)


HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Training configuration
# ============================================================

BATCH_SIZE = 32

NUM_CLASSES = 3


# ------------------------------------------------------------
# Previous experiment
#
# EPOCHS = 5
# LEARNING_RATE = 0.001
#
# ------------------------------------------------------------
# Current experiment
# ------------------------------------------------------------

EPOCHS = 15

LEARNING_RATE = 0.0003


CLASS_NAMES = [
    "Genuine",
    "TTS",
    "Voice Conversion",
]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Start
# ============================================================

print()

print("=" * 70)

print(
    "DeepShield Cached-Feature Training"
)

print("=" * 70)

print()

print(
    "Device:",
    device
)

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

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Load manifests
# ============================================================

train_dataframe = pd.read_csv(
    TRAIN_MANIFEST
)


dev_dataframe = pd.read_csv(
    DEV_MANIFEST
)


print()

print(
    "Training samples:",
    len(train_dataframe)
)

print(
    "Development samples:",
    len(dev_dataframe)
)


# ============================================================
# Calculate class weights
# ============================================================

class_counts = (
    train_dataframe["label_id"]
    .value_counts()
    .sort_index()
)


print()

print(
    "Training class distribution:"
)

print(
    class_counts
)


total_samples = class_counts.sum()


class_weights = (
    total_samples
    / (
        NUM_CLASSES
        * class_counts
    )
)


class_weights = torch.tensor(
    class_weights.values,
    dtype=torch.float32,
    device=device,
)

# ============================================================
# Controlled VC weighting experiment
# ============================================================
# Give Voice Conversion errors more importance during training.
# Class IDs:
#   0 = Genuine
#   1 = TTS
#   2 = Voice Conversion

VC_WEIGHT_MULTIPLIER = 1.5

class_weights[2] *= VC_WEIGHT_MULTIPLIER


print()
print(
    "Class weights (after VC adjustment):"
)

for class_id, weight in enumerate(
    class_weights
):

    print(
        f"{CLASS_NAMES[class_id]}: "
        f"{weight.item():.4f}"
    )

print(
    f"Voice Conversion weight multiplier: "
    f"{VC_WEIGHT_MULTIPLIER:.1f}x"
)


for class_id, weight in enumerate(
    class_weights
):

    print(
        f"{CLASS_NAMES[class_id]}: "
        f"{weight.item():.4f}"
    )


# ============================================================
# Create datasets
# ============================================================

print()

print(
    "Loading cached training features..."
)


train_dataset = DeepShieldAudioDataset(
    train_dataframe,
    split="train"
)


print(
    "Loading cached development features..."
)


dev_dataset = DeepShieldAudioDataset(
    dev_dataframe,
    split="dev"
)


print(
    "Cached feature datasets ready."
)


# ============================================================
# Create DataLoaders
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
# Create fresh model
# ============================================================

model = DeepShieldCNN(
    num_classes=NUM_CLASSES
).to(device)


print()

print(
    "Fresh model created."
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training history
# ============================================================

history = []


best_macro_f1 = -1.0

best_epoch = 0


# ============================================================
# Epoch loop
# ============================================================

for epoch in range(EPOCHS):

    print()

    print("=" * 70)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("=" * 70)


    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_loss = 0.0

    training_correct = 0

    training_total = 0


    for batch_index, (
        features,
        labels
    ) in enumerate(train_loader):

        # ----------------------------------------------------
        # Move data to device
        # ----------------------------------------------------

        features = features.to(
            device,
            non_blocking=True
        )


        labels = labels.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            features
        )


        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        optimizer.step()


        # ----------------------------------------------------
        # Training statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item()
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        training_total += (
            labels.size(0)
        )


        training_correct += (
            predictions == labels
        ).sum().item()


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 100 == 0:

            current_accuracy = (
                100.0
                * training_correct
                / training_total
            )


            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} | "
                f"Loss: "
                f"{loss.item():.4f} | "
                f"Accuracy: "
                f"{current_accuracy:.2f}%"
            )


    # ========================================================
    # Training metrics
    # ========================================================

    train_loss = (
        running_loss
        / len(train_loader)
    )


    train_accuracy = (
        100.0
        * training_correct
        / training_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    validation_loss = 0.0

    validation_labels = []

    validation_predictions = []


    with torch.no_grad():

        for features, labels in dev_loader:

            # ------------------------------------------------
            # Move data to device
            # ------------------------------------------------

            features = features.to(
                device,
                non_blocking=True
            )


            labels = labels.to(
                device,
                non_blocking=True
            )


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                features
            )


            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                labels
            )


            validation_loss += (
                loss.item()
            )


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = torch.argmax(
                outputs,
                dim=1
            )


            validation_labels.extend(
                labels.cpu().numpy()
            )


            validation_predictions.extend(
                predictions.cpu().numpy()
            )


    # ========================================================
    # Validation metrics
    # ========================================================

    validation_loss = (
        validation_loss
        / len(dev_loader)
    )


    validation_accuracy = (
        accuracy_score(
            validation_labels,
            validation_predictions
        )
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            validation_labels,
            validation_predictions
        )
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            validation_labels,
            validation_predictions,
            labels=[
                0,
                1,
                2,
            ],
            zero_division=0,
        )
    )


    macro_f1 = (
        f1.mean()
    )


    # ========================================================
    # Print epoch results
    # ========================================================

    print()

    print("-" * 70)

    print(
        f"Training Loss: "
        f"{train_loss:.4f}"
    )


    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
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


    print()

    print(
        "Per-Class Metrics:"
    )


    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name}: "
            f"Precision="
            f"{precision[class_id] * 100:.2f}% | "
            f"Recall="
            f"{recall[class_id] * 100:.2f}% | "
            f"F1="
            f"{f1[class_id] * 100:.2f}%"
        )


    # ========================================================
    # Save history
    # ========================================================

    history_row = {

        "epoch":
            epoch + 1,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy / 100.0,

        "validation_loss":
            validation_loss,

        "validation_accuracy":
            validation_accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "genuine_precision":
            precision[0],

        "genuine_recall":
            recall[0],

        "genuine_f1":
            f1[0],

        "tts_precision":
            precision[1],

        "tts_recall":
            recall[1],

        "tts_f1":
            f1[1],

        "vc_precision":
            precision[2],

        "vc_recall":
            recall[2],

        "vc_f1":
            f1[2],
    }


    history.append(
        history_row
    )


    history_dataframe = pd.DataFrame(
        history
    )


    history_path = (
        HISTORY_DIR
        / "training_history.csv"
    )


    history_dataframe.to_csv(
        history_path,
        index=False
    )


    # ========================================================
    # Save latest model
    # ========================================================

    latest_model_path = (
        MODEL_DIR
        / "deepshield_cnn_latest.pth"
    )


    torch.save(
        model.state_dict(),
        latest_model_path
    )


    # ========================================================
    # Save best model based on Macro F1
    # ========================================================

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        best_epoch = epoch + 1


        best_model_path = (
            MODEL_DIR
            / "deepshield_cnn_best.pth"
        )


        torch.save(
            model.state_dict(),
            best_model_path
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


    print(
        "-" * 70
    )


# ============================================================
# Training completed
# ============================================================

print()

print("=" * 70)

print(
    "DeepShield Training Completed"
)

print("=" * 70)

print()

print(
    f"Best Epoch: "
    f"{best_epoch}"
)


print(
    f"Best Macro F1: "
    f"{best_macro_f1 * 100:.2f}%"
)


print()

print(
    "Best model:"
)


print(
    MODEL_DIR
    / "deepshield_cnn_best.pth"
)


print()

print(
    "Latest model:"
)


print(
    MODEL_DIR
    / "deepshield_cnn_latest.pth"
)


print()

print(
    "Training history:"
)


print(
    HISTORY_DIR
    / "training_history.csv"
)


print()

print("=" * 70)