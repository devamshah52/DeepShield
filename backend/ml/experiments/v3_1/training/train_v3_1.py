"""
DeepShield V3.1 - Training Pipeline

V3.1 classes:
    0 = Genuine
    1 = TTS
    2 = Voice_Conversion

Important:
    - Uses the V3.1 combined training dataset.
    - Reuses the existing DeepShield V3 architecture.
    - Starts from fresh/random V3 weights.
    - Does not load V1/V2 weights.
    - Test set is NOT used during training.
    - Uses the unchanged V3 development set.
    - Best checkpoint is selected using Dev Balanced Accuracy.
    - Supports CUDA + AMP on the RTX 5060 Laptop GPU.
    - V3 files and V3 checkpoints are NOT modified.
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# V3.1 paths
# ============================================================

V31_ROOT = Path(
    __file__
).resolve().parents[1]

if str(V31_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(V31_ROOT),
    )


# ============================================================
# V3 model path
# ============================================================

V3_ROOT = (
    V31_ROOT.parent
    / "v3"
)

if str(V3_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(V3_ROOT),
    )


# ============================================================
# V3.1 configuration
# ============================================================

from v3_1_config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_NAME,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    FEATURE_HEIGHT,
    FEATURE_WIDTH,
    GRADIENT_CLIP_MAX_NORM,
    INPUT_CHANNELS,
    LAST_CHECKPOINT_NAME,
    LEARNING_RATE,
    LOG_DIR,
    MIN_LEARNING_RATE,
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    PRIMARY_METRIC,
    RANDOM_SEED,
    SCHEDULER_NAME,
    TRAINING_HISTORY_NAME,
    USE_AMP,
    WEIGHT_DECAY,
)


# ============================================================
# V3.1 dataset
# ============================================================

from dataset.v3_1_dataset import (
    DeepShieldV31Dataset,
)


# ============================================================
# Existing V3 model
# ============================================================

from model.v3_model import (
    DeepShieldV3,
    count_parameters,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# Device
# ============================================================

def get_device():

    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 0
    ):

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# DataLoader
# ============================================================

def create_dataloaders():

    print()
    print("=" * 70)
    print("Loading V3.1 datasets")
    print("=" * 70)

    train_dataset = DeepShieldV31Dataset(
        "train"
    )

    dev_dataset = DeepShieldV31Dataset(
        "dev"
    )

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

    print()
    print(
        f"Train samples : "
        f"{len(train_dataset)}"
    )

    print(
        f"Dev samples   : "
        f"{len(dev_dataset)}"
    )

    print()
    print(
        f"Train batches : "
        f"{len(train_loader)}"
    )

    print(
        f"Dev batches   : "
        f"{len(dev_loader)}"
    )

    return (
        train_dataset,
        dev_dataset,
        train_loader,
        dev_loader,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    targets,
    predictions,
):

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            targets,
            predictions,
        )
    )

    precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(
        targets,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_precision": float(
            precision
        ),
        "macro_recall": float(
            recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "confusion_matrix": cm,
    }


# ============================================================
# Print metrics
# ============================================================

def print_metrics(
    metrics,
    prefix="",
):

    print(
        f"{prefix}Accuracy          : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"{prefix}Balanced Accuracy : "
        f"{metrics['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"{prefix}Macro Precision   : "
        f"{metrics['macro_precision'] * 100:.2f}%"
    )

    print(
        f"{prefix}Macro Recall      : "
        f"{metrics['macro_recall'] * 100:.2f}%"
    )

    print(
        f"{prefix}Macro F1          : "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"{prefix}Weighted F1       : "
        f"{metrics['weighted_f1'] * 100:.2f}%"
    )


# ============================================================
# Print confusion matrix
# ============================================================

def print_confusion_matrix(
    cm,
):

    print()
    print(
        "Confusion Matrix"
    )

    print(
        "Rows = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print()

    print(
        f"{'':22s}"
        f"{'Genuine':>12s}"
        f"{'TTS':>12s}"
        f"{'Voice_Conversion':>20s}"
    )

    for row_index in range(
        NUM_CLASSES
    ):

        print(
            f"{CLASS_NAMES[row_index]:22s}"
            f"{cm[row_index, 0]:12d}"
            f"{cm[row_index, 1]:12d}"
            f"{cm[row_index, 2]:20d}"
        )


# ============================================================
# Per-class metrics
# ============================================================

def print_per_class_metrics(
    targets,
    predictions,
):

    precision, recall, f1, support = (
        _per_class_metrics(
            targets,
            predictions,
        )
    )

    print()
    print(
        "Per-Class Metrics"
    )

    print()

    print(
        f"{'Class':22s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
        f"{'Support':>12s}"
    )

    for label_id in range(
        NUM_CLASSES
    ):

        print(
            f"{CLASS_NAMES[label_id]:22s}"
            f"{precision[label_id] * 100:11.2f}%"
            f"{recall[label_id] * 100:11.2f}%"
            f"{f1[label_id] * 100:11.2f}%"
            f"{support[label_id]:12d}"
        )


def _per_class_metrics(
    targets,
    predictions,
):

    from sklearn.metrics import (
        precision_recall_fscore_support,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=list(
                range(NUM_CLASSES)
            ),
            zero_division=0,
        )
    )

    return (
        precision,
        recall,
        f1,
        support,
    )


# ============================================================
# Training epoch
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

    for (
        features,
        labels,
    ) in loader:

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

        if (
            USE_AMP
            and device.type == "cuda"
        ):

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                outputs = model(
                    features
                )

                loss = criterion(
                    outputs,
                    labels,
                )

        else:

            outputs = model(
                features
            )

            loss = criterion(
                outputs,
                labels,
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        if (
            USE_AMP
            and device.type == "cuda"
        ):

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_MAX_NORM,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_MAX_NORM,
            )

            optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = (
            features.size(0)
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
    )

    metrics["loss"] = float(
        epoch_loss
    )

    return metrics


# ============================================================
# Validation epoch
# ============================================================

@torch.no_grad()
def validate_one_epoch(
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

    for (
        features,
        labels,
    ) in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if (
            USE_AMP
            and device.type == "cuda"
        ):

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                outputs = model(
                    features
                )

                loss = criterion(
                    outputs,
                    labels,
                )

        else:

            outputs = model(
                features
            )

            loss = criterion(
                outputs,
                labels,
            )

        batch_size = (
            features.size(0)
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
    )

    metrics["loss"] = float(
        epoch_loss
    )

    metrics["_targets"] = (
        all_targets
    )

    metrics["_predictions"] = (
        all_predictions
    )

    return metrics


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    train_metrics,
    dev_metrics,
    best_metric,
):

    checkpoint = {
        "version": "DeepShield V3.1",

        "architecture": "DeepShieldV3",

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict()
            if scaler is not None
            else None,

        "num_classes":
            NUM_CLASSES,

        "class_names":
            CLASS_NAMES,

        "input_channels":
            INPUT_CHANNELS,

        "feature_height":
            FEATURE_HEIGHT,

        "feature_width":
            FEATURE_WIDTH,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "batch_size":
            BATCH_SIZE,

        "random_seed":
            RANDOM_SEED,

        "primary_metric":
            PRIMARY_METRIC,

        "best_metric":
            best_metric,

        "train_metrics":
            _clean_metrics(
                train_metrics
            ),

        "dev_metrics":
            _clean_metrics(
                dev_metrics
            ),
    }

    torch.save(
        checkpoint,
        path,
    )


def _clean_metrics(
    metrics,
):

    cleaned = {}

    for key, value in metrics.items():

        if key.startswith("_"):

            continue

        if key == "confusion_matrix":

            cleaned[key] = (
                value.tolist()
            )

        else:

            cleaned[key] = value

    return cleaned


# ============================================================
# Save training history
# ============================================================

def save_history(
    history,
    path: Path,
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
        "epoch_time_seconds",
    ]

    with path.open(
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
# Main training
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DeepShield V3.1 - Training")
    print("=" * 70)

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()
    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

        print(
            f"GPU Memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    else:

        print(
            "WARNING: CUDA is not available."
        )

    # --------------------------------------------------------
    # Configuration summary
    # --------------------------------------------------------

    print()
    print(
        "Training configuration:"
    )

    print(
        f"  Epochs          : {EPOCHS}"
    )

    print(
        f"  Batch size      : {BATCH_SIZE}"
    )

    print(
        f"  Learning rate   : {LEARNING_RATE}"
    )

    print(
        f"  Weight decay    : {WEIGHT_DECAY}"
    )

    print(
        f"  Scheduler       : {SCHEDULER_NAME}"
    )

    print(
        f"  AMP             : {USE_AMP}"
    )

    print(
        f"  Early stopping  : "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"  Primary metric  : "
        f"{PRIMARY_METRIC}"
    )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    (
        train_dataset,
        dev_dataset,
        train_loader,
        dev_loader,
    ) = create_dataloaders()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Creating V3 model for V3.1 training")
    print("=" * 70)

    model = DeepShieldV3(
        num_classes=NUM_CLASSES
    )

    model = model.to(
        device
    )

    print()
    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    print(
        "Initialization: "
        "RANDOM V3 WEIGHTS"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=MIN_LEARNING_RATE,
        )
    )

    # --------------------------------------------------------
    # AMP scaler
    # --------------------------------------------------------

    scaler = None

    if (
        USE_AMP
        and device.type == "cuda"
    ):

        scaler = torch.amp.GradScaler(
            "cuda"
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

    best_checkpoint_path = (
        CHECKPOINT_DIR
        / BEST_CHECKPOINT_NAME
    )

    last_checkpoint_path = (
        CHECKPOINT_DIR
        / LAST_CHECKPOINT_NAME
    )

    history_path = (
        LOG_DIR
        / TRAINING_HISTORY_NAME
    )

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Starting V3.1 training")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Training dataset: V3.1 combined dataset"
    )

    print(
        "  Development dataset: "
        "unchanged V3 dev dataset"
    )

    print(
        "  Test dataset will NOT be used during training."
    )

    print(
        "  Best checkpoint is selected using DEV "
        "balanced accuracy."
    )

    print(
        "  V3 model architecture is reused unchanged."
    )

    print(
        "  Model starts from fresh/random weights."
    )

    print()

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = time.time()

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print()
        print("=" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Learning rate: "
            f"{current_lr:.8f}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
        )

        print()
        print(
            "TRAIN"
        )

        print(
            f"  Loss: "
            f"{train_metrics['loss']:.6f}"
        )

        print_metrics(
            train_metrics,
            prefix="  ",
        )

        # ----------------------------------------------------
        # Dev
        # ----------------------------------------------------

        dev_metrics = validate_one_epoch(
            model,
            dev_loader,
            criterion,
            device,
        )

        print()
        print(
            "DEV"
        )

        print(
            f"  Loss: "
            f"{dev_metrics['loss']:.6f}"
        )

        print_metrics(
            dev_metrics,
            prefix="  ",
        )

        # ----------------------------------------------------
        # Per-class dev metrics
        # ----------------------------------------------------

        print_per_class_metrics(
            dev_metrics["_targets"],
            dev_metrics["_predictions"],
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        print_confusion_matrix(
            dev_metrics["confusion_matrix"]
        )

        # ----------------------------------------------------
        # Epoch timing
        # ----------------------------------------------------

        epoch_time = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Epoch time: "
            f"{epoch_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history_row = {
            "epoch":
                epoch,

            "learning_rate":
                current_lr,

            "train_loss":
                train_metrics["loss"],

            "train_accuracy":
                train_metrics["accuracy"],

            "train_balanced_accuracy":
                train_metrics[
                    "balanced_accuracy"
                ],

            "train_macro_precision":
                train_metrics[
                    "macro_precision"
                ],

            "train_macro_recall":
                train_metrics[
                    "macro_recall"
                ],

            "train_macro_f1":
                train_metrics[
                    "macro_f1"
                ],

            "train_weighted_f1":
                train_metrics[
                    "weighted_f1"
                ],

            "dev_loss":
                dev_metrics["loss"],

            "dev_accuracy":
                dev_metrics["accuracy"],

            "dev_balanced_accuracy":
                dev_metrics[
                    "balanced_accuracy"
                ],

            "dev_macro_precision":
                dev_metrics[
                    "macro_precision"
                ],

            "dev_macro_recall":
                dev_metrics[
                    "macro_recall"
                ],

            "dev_macro_f1":
                dev_metrics[
                    "macro_f1"
                ],

            "dev_weighted_f1":
                dev_metrics[
                    "weighted_f1"
                ],

            "epoch_time_seconds":
                epoch_time,
        }

        history.append(
            history_row
        )

        save_history(
            history,
            history_path,
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

            improvement = (
                current_metric
                - best_metric
                if best_metric != -float("inf")
                else current_metric
            )

            best_metric = (
                current_metric
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                train_metrics,
                dev_metrics,
                best_metric,
            )

            print()
            print(
                "★ NEW BEST CHECKPOINT"
            )

            print(
                f"  Epoch: {epoch}"
            )

            print(
                f"  Dev {PRIMARY_METRIC}: "
                f"{current_metric * 100:.2f}%"
            )

            if epoch > 1:

                print(
                    f"  Improvement: "
                    f"{improvement * 100:.4f} percentage points"
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
                f"  Best dev {PRIMARY_METRIC}: "
                f"{best_metric * 100:.2f}%"
            )

            print(
                f"  Patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            train_metrics,
            dev_metrics,
            best_metric,
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
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            print(
                f"Best dev {PRIMARY_METRIC}: "
                f"{best_metric * 100:.2f}%"
            )

            print("=" * 70)

            break

    # ========================================================
    # Training finished
    # ========================================================

    print()
    print("=" * 70)
    print("V3.1 TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best dev {PRIMARY_METRIC}: "
        f"{best_metric * 100:.2f}%"
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        f"  {best_checkpoint_path}"
    )

    print()
    print(
        "Last checkpoint:"
    )

    print(
        f"  {last_checkpoint_path}"
    )

    print()
    print(
        "Training history:"
    )

    print(
        f"  {history_path}"
    )

    print()
    print(
        "The TEST dataset was not used."
    )

    print()
    print(
        "V3 files and V3 checkpoints were not modified."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()