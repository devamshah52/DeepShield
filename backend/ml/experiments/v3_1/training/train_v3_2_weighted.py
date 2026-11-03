"""
DeepShield V3.2 - Experiment 2
Class-Weighted Training

Purpose:
    Test whether class-weighted CrossEntropyLoss can recover
    Voice Conversion performance while preserving the gTTS
    generalization achieved by V3.2.

Training data:
    Same V3.2 9,024-sample training set.

Dev:
    Same original V3 dev set.

Test:
    NOT USED DURING TRAINING.

ONLY INTENTIONAL TRAINING CHANGE FROM V3.2:
    Class-weighted CrossEntropyLoss.

Class counts:
    Genuine          3608
    TTS              3608
    Voice_Conversion 1808

Weights:
    Genuine          0.8333
    TTS              0.8333
    Voice_Conversion 1.6667

The weights are normalized so the average class weight is 1.0.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(
    r"C:\DeepShield"
)

BACKEND_DIR = (
    PROJECT_ROOT
    / "backend"
)

V3_ROOT = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3"
)

V31_ROOT = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3_1"
)

CHECKPOINT_DIR = (
    V31_ROOT
    / "checkpoints"
)

LOG_DIR = (
    V31_ROOT
    / "logs"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "deepshield_v3_2_weighted_best.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "deepshield_v3_2_weighted_last.pth"
)

HISTORY_FILE = (
    LOG_DIR
    / "v3_2_weighted_training_history.csv"
)


# ============================================================
# Python paths
# ============================================================

if str(BACKEND_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(BACKEND_DIR)
    )

if str(V3_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(V3_ROOT)
    )


# ============================================================
# Imports
# ============================================================

from ml.experiments.v3.model.v3_model import (
    DeepShieldV3,
)

from ml.experiments.v3_1.dataset.v3_2_dataset import (
    DeepShieldV32Dataset,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

NUM_CLASSES = 3

BATCH_SIZE = 32

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 0.0001

EPOCHS = 30

MIN_LEARNING_RATE = 1e-6

EARLY_STOPPING_PATIENCE = 7

GRADIENT_CLIP_MAX_NORM = 5.0

NUM_WORKERS = 0

PIN_MEMORY = True

USE_AMP = True

PRIMARY_METRIC = "balanced_accuracy"


# ============================================================
# Class weights
# ============================================================

CLASS_WEIGHTS = torch.tensor(
    [
        0.8333333333,   # Genuine
        0.8333333333,   # TTS
        1.6666666667,   # Voice_Conversion
    ],
    dtype=torch.float32,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    targets,
    predictions,
    num_classes=3,
):

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    confusion = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    for target, prediction in zip(
        targets,
        predictions,
    ):

        confusion[
            target,
            prediction
        ] += 1

    total = confusion.sum()

    accuracy = (
        np.trace(confusion)
        / total
        if total > 0
        else 0.0
    )

    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for class_id in range(
        num_classes
    ):

        tp = confusion[
            class_id,
            class_id
        ]

        fp = (
            confusion[
                :,
                class_id
            ].sum()
            - tp
        )

        fn = (
            confusion[
                class_id,
                :
            ].sum()
            - tp
        )

        support = confusion[
            class_id,
            :
        ].sum()

        if tp + fp > 0:

            precision = (
                tp
                / (tp + fp)
            )

        else:

            precision = 0.0

        if tp + fn > 0:

            recall = (
                tp
                / (tp + fn)
            )

        else:

            recall = 0.0

        if precision + recall > 0:

            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        else:

            f1 = 0.0

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1_scores.append(
            f1
        )

        supports.append(
            support
        )

    balanced_accuracy = float(
        np.mean(
            recalls
        )
    )

    macro_precision = float(
        np.mean(
            precisions
        )
    )

    macro_recall = float(
        np.mean(
            recalls
        )
    )

    macro_f1 = float(
        np.mean(
            f1_scores
        )
    )

    supports = np.asarray(
        supports,
        dtype=np.float64,
    )

    weighted_f1 = float(
        np.sum(
            np.asarray(
                f1_scores
            )
            * supports
        )
        / supports.sum()
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion,
        "class_precision": precisions,
        "class_recall": recalls,
        "class_f1": f1_scores,
        "class_support": supports.astype(
            np.int64
        ),
    }


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
):

    model.train()

    running_loss = 0.0

    total_samples = 0

    all_targets = []

    all_predictions = []

    for features, labels in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            outputs = model(
                features
            )

            loss = criterion(
                outputs,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_MAX_NORM,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = (
            labels.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_targets.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
        NUM_CLASSES,
    )

    metrics["loss"] = (
        running_loss
        / total_samples
    )

    return metrics


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    total_samples = 0

    all_targets = []

    all_predictions = []

    for features, labels in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            outputs = model(
                features
            )

            loss = criterion(
                outputs,
                labels,
            )

        batch_size = (
            labels.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_targets.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
        NUM_CLASSES,
    )

    metrics["loss"] = (
        running_loss
        / total_samples
    )

    return metrics


# ============================================================
# Print metrics
# ============================================================

def print_metrics(
    name,
    metrics,
):

    print()
    print(
        f"{name} Metrics:"
    )

    print(
        f"  Loss:              "
        f"{metrics['loss']:.6f}"
    )

    print(
        f"  Accuracy:          "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Balanced Accuracy: "
        f"{metrics['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro Precision:   "
        f"{metrics['macro_precision'] * 100:.2f}%"
    )

    print(
        f"  Macro Recall:      "
        f"{metrics['macro_recall'] * 100:.2f}%"
    )

    print(
        f"  Macro F1:          "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"  Weighted F1:       "
        f"{metrics['weighted_f1'] * 100:.2f}%"
    )


# ============================================================
# Print confusion matrix
# ============================================================

def print_confusion_matrix(
    confusion,
):

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        "                  "
        "Genuine     TTS     "
        "Voice_Conversion"
    )

    names = [
        "Genuine",
        "TTS",
        "Voice_Conversion",
    ]

    for index, name in enumerate(
        names
    ):

        row = confusion[
            index
        ]

        print(
            f"{name:18s}"
            f"{row[0]:8d}"
            f"{row[1]:8d}"
            f"{row[2]:18d}"
        )


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_metric,
    train_metrics,
    dev_metrics,
):

    checkpoint = {
        "version": (
            "DeepShield V3.2 "
            "Class-Weighted"
        ),

        "architecture": (
            "DeepShieldV3"
        ),

        "epoch": epoch,

        "best_metric": best_metric,

        "primary_metric": (
            PRIMARY_METRIC
        ),

        "class_weights": (
            CLASS_WEIGHTS.tolist()
        ),

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
        ),

        "scaler_state_dict": (
            scaler.state_dict()
        ),

        "train_metrics": train_metrics,

        "dev_metrics": dev_metrics,
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# Save history
# ============================================================

def save_history(
    history,
):

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_balanced_accuracy",
        "train_macro_precision",
        "train_macro_recall",
        "train_macro_f1",
        "train_weighted_f1",
        "dev_loss",
        "dev_accuracy",
        "dev_balanced_accuracy",
        "dev_macro_precision",
        "dev_macro_recall",
        "dev_macro_f1",
        "dev_weighted_f1",
    ]

    with open(
        HISTORY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# Main
# ============================================================

def main():

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(
        SEED
    )

    print()
    print("=" * 70)

    print(
        "DeepShield V3.2 "
        "Class-Weighted Training"
    )

    print("=" * 70)

    print()
    print(
        "This experiment changes ONLY "
        "the loss weighting."
    )

    print()
    print(
        "Class weights:"
    )

    print(
        "  Genuine:           "
        f"{CLASS_WEIGHTS[0].item():.4f}"
    )

    print(
        "  TTS:               "
        f"{CLASS_WEIGHTS[1].item():.4f}"
    )

    print(
        "  Voice_Conversion:  "
        f"{CLASS_WEIGHTS[2].item():.4f}"
    )

    print()
    print(
        f"Train: 9,024"
    )

    print(
        f"Dev:   1,158"
    )

    print(
        "Test:  NOT USED"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda"
    )

    print()
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: "
        f"{torch.version.cuda}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print()
    print(
        "Loading datasets..."
    )

    train_dataset = (
        DeepShieldV32Dataset(
            "train"
        )
    )

    dev_dataset = (
        DeepShieldV32Dataset(
            "dev"
        )
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    print(
        f"Train batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Dev batches: "
        f"{len(dev_loader)}"
    )

    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    print()
    print(
        "Creating fresh DeepShieldV3 model..."
    )

    model = DeepShieldV3(
        num_classes=NUM_CLASSES
    ).to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Weighted loss
    # --------------------------------------------------------

    class_weights = (
        CLASS_WEIGHTS.to(
            device
        )
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print()
    print(
        "Weighted CrossEntropyLoss: ENABLED"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    best_metric = -float(
        "inf"
    )

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

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

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        print(
            f"Learning rate: "
            f"{current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
            )
        )

        # ----------------------------------------------------
        # Dev
        # ----------------------------------------------------

        dev_metrics = (
            evaluate(
                model,
                dev_loader,
                criterion,
                device,
            )
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print_metrics(
            "Train",
            train_metrics,
        )

        print_metrics(
            "Dev",
            dev_metrics,
        )

        print_confusion_matrix(
            dev_metrics[
                "confusion_matrix"
            ]
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,

                "train_loss": (
                    train_metrics["loss"]
                ),

                "train_accuracy": (
                    train_metrics["accuracy"]
                ),

                "train_balanced_accuracy": (
                    train_metrics[
                        "balanced_accuracy"
                    ]
                ),

                "train_macro_precision": (
                    train_metrics[
                        "macro_precision"
                    ]
                ),

                "train_macro_recall": (
                    train_metrics[
                        "macro_recall"
                    ]
                ),

                "train_macro_f1": (
                    train_metrics[
                        "macro_f1"
                    ]
                ),

                "train_weighted_f1": (
                    train_metrics[
                        "weighted_f1"
                    ]
                ),

                "dev_loss": (
                    dev_metrics["loss"]
                ),

                "dev_accuracy": (
                    dev_metrics["accuracy"]
                ),

                "dev_balanced_accuracy": (
                    dev_metrics[
                        "balanced_accuracy"
                    ]
                ),

                "dev_macro_precision": (
                    dev_metrics[
                        "macro_precision"
                    ]
                ),

                "dev_macro_recall": (
                    dev_metrics[
                        "macro_recall"
                    ]
                ),

                "dev_macro_f1": (
                    dev_metrics[
                        "macro_f1"
                    ]
                ),

                "dev_weighted_f1": (
                    dev_metrics[
                        "weighted_f1"
                    ]
                ),
            }
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        current_metric = (
            dev_metrics[
                PRIMARY_METRIC
            ]
        )

        if current_metric > best_metric:

            best_metric = (
                current_metric
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                train_metrics,
                dev_metrics,
            )

            print()
            print(
                "NEW BEST CHECKPOINT"
            )

            print(
                f"  Epoch: {epoch}"
            )

            print(
                f"  Dev {PRIMARY_METRIC}: "
                f"{best_metric * 100:.2f}%"
            )

            print(
                f"  Saved: "
                f"{BEST_CHECKPOINT}"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                "No improvement."
            )

            print(
                f"  Best epoch: "
                f"{best_epoch}"
            )

            print(
                f"  Best balanced_accuracy: "
                f"{best_metric * 100:.2f}%"
            )

            print(
                f"  Epochs without improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            train_metrics,
            dev_metrics,
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        save_history(
            history
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print("=" * 70)

            print(
                "EARLY STOPPING"
            )

            print(
                f"No improvement for "
                f"{EARLY_STOPPING_PATIENCE} "
                f"epochs."
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            print(
                f"Best dev balanced_accuracy: "
                f"{best_metric * 100:.2f}%"
            )

            print("=" * 70)

            break

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "V3.2 CLASS-WEIGHTED "
        "TRAINING COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best dev balanced_accuracy: "
        f"{best_metric * 100:.2f}%"
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print()
    print(
        "Last checkpoint:"
    )

    print(
        LAST_CHECKPOINT
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
        "The original V3 test set was NOT used."
    )

    print(
        "The 20-file gTTS blind test was NOT used."
    )

    print(
        "V3.2 unweighted checkpoints were NOT modified."
    )

    print(
        "V3.1 checkpoints were NOT modified."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()