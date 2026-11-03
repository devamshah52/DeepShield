"""
DeepShield V3.2 - Training

Training data:
    V3.1 training data + 1,800 gTTS TTS samples.

Development:
    Original V3 development set.

Test:
    NOT USED DURING TRAINING.

Model:
    Existing DeepShieldV3 architecture.

Outputs:
    V3.2-specific checkpoints and training history.

Important:
    - V3 architecture is NOT modified.
    - V3 checkpoints are NOT modified.
    - V3.1 checkpoints are NOT modified.
    - V3 test data is NOT used.
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

PROJECT_ROOT = Path(r"C:\DeepShield")

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

from ml.experiments.v3_1.training import (
    v3_2_config as config,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(
    seed: int,
):

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

    # Deterministic behavior where practical.
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    targets,
    predictions,
    num_classes: int = 3,
):
    """
    Calculate:
        accuracy
        balanced accuracy
        macro precision
        macro recall
        macro F1
        weighted F1
        confusion matrix
    """

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

    recalls = []
    precisions = []
    f1_scores = []
    supports = []

    for class_id in range(
        num_classes
    ):

        true_positive = (
            confusion[
                class_id,
                class_id
            ]
        )

        false_positive = (
            confusion[
                :,
                class_id
            ].sum()
            - true_positive
        )

        false_negative = (
            confusion[
                class_id,
                :
            ].sum()
            - true_positive
        )

        support = confusion[
            class_id,
            :
        ].sum()

        supports.append(
            support
        )

        if (
            true_positive
            + false_negative
        ) > 0:

            recall = (
                true_positive
                / (
                    true_positive
                    + false_negative
                )
            )

        else:

            recall = 0.0

        if (
            true_positive
            + false_positive
        ) > 0:

            precision = (
                true_positive
                / (
                    true_positive
                    + false_positive
                )
            )

        else:

            precision = 0.0

        if (
            precision
            + recall
        ) > 0:

            f1 = (
                2.0
                * precision
                * recall
                / (
                    precision
                    + recall
                )
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

    balanced_accuracy = (
        float(
            np.mean(
                recalls
            )
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

    total_support = (
        supports.sum()
    )

    if total_support > 0:

        weighted_f1 = float(
            np.sum(
                np.asarray(
                    f1_scores
                )
                * supports
            )
            / total_support
        )

    else:

        weighted_f1 = 0.0

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
            enabled=config.USE_AMP,
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
            max_norm=(
                config.GRADIENT_CLIP_MAX_NORM
            ),
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

        predictions = (
            torch.argmax(
                outputs,
                dim=1,
            )
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

    epoch_loss = (
        running_loss
        / total_samples
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
        config.NUM_CLASSES,
    )

    metrics["loss"] = epoch_loss

    return metrics


# ============================================================
# Evaluate
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
            enabled=config.USE_AMP,
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

        predictions = (
            torch.argmax(
                outputs,
                dim=1,
            )
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

    epoch_loss = (
        running_loss
        / total_samples
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
        config.NUM_CLASSES,
    )

    metrics["loss"] = epoch_loss

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
        f"  Macro F1:           "
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

    for index, name in enumerate(
        config.CLASS_NAMES.values()
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
        "version": "DeepShield V3.2",

        "architecture": "DeepShieldV3",

        "epoch": epoch,

        "best_metric": best_metric,

        "primary_metric": (
            config.PRIMARY_METRIC
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

        "config": {
            "batch_size": config.BATCH_SIZE,
            "learning_rate": (
                config.LEARNING_RATE
            ),
            "weight_decay": (
                config.WEIGHT_DECAY
            ),
            "epochs": config.EPOCHS,
            "optimizer": (
                config.OPTIMIZER_NAME
            ),
            "scheduler": (
                config.SCHEDULER_NAME
            ),
            "use_amp": config.USE_AMP,
            "use_class_weights": (
                config.USE_CLASS_WEIGHTS
            ),
        },
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# Save training history
# ============================================================

def save_history(
    history,
    path,
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
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in history:

            writer.writerow(
                row
            )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config.validate_config()

    config.create_directories()

    set_seed(
        config.RANDOM_SEED
    )

    print()
    print("=" * 70)

    print(
        "DeepShield V3.2 Training"
    )

    print("=" * 70)

    config.print_config()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "V3.2 training requires CUDA."
        )

    device = torch.device(
        "cuda"
    )

    print()
    print(
        f"Training device: "
        f"{device}"
    )

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

    print()
    print(
        "Creating DataLoaders..."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=(
            config.DROP_LAST_TRAIN_BATCH
        ),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    print(
        f"Train batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Dev batches: "
        f"{len(dev_loader):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Creating fresh DeepShieldV3 model..."
    )

    model = DeepShieldV3(
        num_classes=config.NUM_CLASSES
    )

    model = model.to(
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
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=config.MIN_LEARNING_RATE,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.USE_AMP,
    )

    # --------------------------------------------------------
    # Training state
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
        config.EPOCHS + 1,
    ):

        print()
        print("=" * 70)

        print(
            f"Epoch {epoch}/"
            f"{config.EPOCHS}"
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
                config.PRIMARY_METRIC
            ]
        )

        if current_metric > best_metric:

            best_metric = (
                current_metric
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            best_path = (
                config.CHECKPOINT_DIR
                / config.BEST_CHECKPOINT_NAME
            )

            save_checkpoint(
                best_path,
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
                f"  Epoch: "
                f"{epoch}"
            )

            print(
                f"  Dev "
                f"{config.PRIMARY_METRIC}: "
                f"{best_metric * 100:.2f}%"
            )

            print(
                f"  Saved: "
                f"{best_path}"
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
                f"  Best "
                f"{config.PRIMARY_METRIC}: "
                f"{best_metric * 100:.2f}%"
            )

            print(
                f"  Epochs without "
                f"improvement: "
                f"{epochs_without_improvement}/"
                f"{config.EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        last_path = (
            config.CHECKPOINT_DIR
            / config.LAST_CHECKPOINT_NAME
        )

        save_checkpoint(
            last_path,
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

        history_path = (
            config.LOG_DIR
            / config.TRAINING_HISTORY_NAME
        )

        save_history(
            history,
            history_path,
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
            >= config.EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "=" * 70
            )

            print(
                "EARLY STOPPING"
            )

            print(
                f"No improvement for "
                f"{config.EARLY_STOPPING_PATIENCE} "
                f"epochs."
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            print(
                f"Best dev "
                f"{config.PRIMARY_METRIC}: "
                f"{best_metric * 100:.2f}%"
            )

            print(
                "=" * 70
            )

            break

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "V3.2 TRAINING COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best dev "
        f"{config.PRIMARY_METRIC}: "
        f"{best_metric * 100:.2f}%"
    )

    print()

    print(
        "Best checkpoint:"
    )

    print(
        config.CHECKPOINT_DIR
        / config.BEST_CHECKPOINT_NAME
    )

    print()

    print(
        "Last checkpoint:"
    )

    print(
        config.CHECKPOINT_DIR
        / config.LAST_CHECKPOINT_NAME
    )

    print()

    print(
        "Training history:"
    )

    print(
        config.LOG_DIR
        / config.TRAINING_HISTORY_NAME
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The V3 test set was NOT used "
        "during training."
    )

    print(
        "The separate 20-sample gTTS "
        "blind test was NOT used."
    )

    print(
        "V3 and V3.1 checkpoints were "
        "not modified."
    )

    print()

    print("=" * 70)


if __name__ == "__main__":

    main()