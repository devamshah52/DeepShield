"""
DeepShield V3 - Held-Out Test Evaluation

Evaluates ONLY the best V3 checkpoint against v3_test.csv.

Important:
    - No training.
    - No optimizer.
    - No scheduler.
    - No model updates.
    - Test set is used only here.
    - V1/V2 models are not loaded.

Classes:
    0 = Genuine
    1 = TTS
    2 = Voice_Conversion
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)


# ============================================================
# Resolve V3 paths
# ============================================================

CURRENT_FILE = Path(
    __file__
).resolve()

V3_ROOT = (
    CURRENT_FILE
    .parents[1]
)

if str(V3_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(V3_ROOT),
    )


# ============================================================
# V3 imports
# ============================================================

from model.v3_model import (
    DeepShieldV3,
    count_parameters,
)

from training.v3_dataset import (
    DeepShieldV3Dataset,
)

from training.v3_config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CHECKPOINT_DIR,
    FEATURE_HEIGHT,
    FEATURE_WIDTH,
    INPUT_CHANNELS,
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    BEST_CHECKPOINT_NAME,
)


# ============================================================
# Evaluation paths
# ============================================================

EVALUATION_DIR = (
    V3_ROOT
    / "evaluation"
)

RESULTS_JSON = (
    EVALUATION_DIR
    / "v3_test_results.json"
)

CONFUSION_MATRIX_CSV = (
    EVALUATION_DIR
    / "v3_test_confusion_matrix.csv"
)

CLASS_METRICS_CSV = (
    EVALUATION_DIR
    / "v3_test_class_metrics.csv"
)


# ============================================================
# Configuration
# ============================================================

TEST_BATCH_SIZE = BATCH_SIZE


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# Load checkpoint
# ============================================================

def load_best_checkpoint(
    model,
    device,
):

    checkpoint_path = (
        CHECKPOINT_DIR
        / BEST_CHECKPOINT_NAME
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Best V3 checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print()
    print(
        "Loading best V3 checkpoint:"
    )

    print(
        f"  {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Validate checkpoint identity
    # --------------------------------------------------------

    version = checkpoint.get(
        "version"
    )

    if version != "DeepShield V3":

        raise RuntimeError(
            "Checkpoint is not a DeepShield V3 "
            f"checkpoint. Found: {version}"
        )

    # --------------------------------------------------------
    # Validate architecture configuration
    # --------------------------------------------------------

    checkpoint_num_classes = (
        checkpoint.get(
            "num_classes"
        )
    )

    if (
        checkpoint_num_classes
        != NUM_CLASSES
    ):

        raise RuntimeError(
            "Checkpoint class count mismatch: "
            f"{checkpoint_num_classes}"
        )

    checkpoint_channels = (
        checkpoint.get(
            "input_channels"
        )
    )

    if (
        checkpoint_channels
        != INPUT_CHANNELS
    ):

        raise RuntimeError(
            "Checkpoint input channel mismatch: "
            f"{checkpoint_channels}"
        )

    checkpoint_height = (
        checkpoint.get(
            "feature_height"
        )
    )

    checkpoint_width = (
        checkpoint.get(
            "feature_width"
        )
    )

    if (
        checkpoint_height
        != FEATURE_HEIGHT
        or checkpoint_width
        != FEATURE_WIDTH
    ):

        raise RuntimeError(
            "Checkpoint feature dimensions do not "
            "match the V3 configuration."
        )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    best_epoch = checkpoint.get(
        "epoch"
    )

    best_metric = checkpoint.get(
        "best_metric"
    )

    print()
    print(
        "Checkpoint information:"
    )

    print(
        f"  Version       : {version}"
    )

    print(
        f"  Best epoch    : {best_epoch}"
    )

    if best_metric is not None:

        print(
            f"  Dev balanced accuracy: "
            f"{best_metric * 100:.4f}%"
        )

    else:

        print(
            "  Dev balanced accuracy: N/A"
        )

    return (
        checkpoint,
        checkpoint_path,
    )


# ============================================================
# Evaluate model
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

    total_samples = 0

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

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                outputs = model(
                    features
                )

        else:

            outputs = model(
                features
            )

        # ----------------------------------------------------
        # Convert logits to probabilities
        #
        # Softmax is calculated normally from the model
        # outputs. The result is converted to float32 before
        # being stored.
        # ----------------------------------------------------

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        batch_size = (
            labels.size(0)
        )

        total_samples += (
            batch_size
        )

        all_targets.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

        all_probabilities.extend(
            probabilities.cpu()
            .float()
            .numpy()
            .tolist()
        )

    return (
        np.array(
            all_targets,
            dtype=np.int64,
        ),
        np.array(
            all_predictions,
            dtype=np.int64,
        ),
        np.array(
            all_probabilities,
            dtype=np.float32,
        ),
    )


# ============================================================
# Calculate metrics
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

    macro_precision = (
        precision_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = (
        recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    macro_f1 = (
        f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = (
        f1_score(
            targets,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        targets,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    (
        class_precision,
        class_recall,
        class_f1,
        support,
    ) = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "confusion_matrix": cm,
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_f1": class_f1,
        "class_support": support,
    }


# ============================================================
# Print overall metrics
# ============================================================

def print_overall_metrics(
    metrics,
):

    print()
    print("=" * 70)
    print("V3 TEST RESULTS")
    print("=" * 70)

    print()

    print(
        f"Accuracy          : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy : "
        f"{metrics['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro Precision   : "
        f"{metrics['macro_precision'] * 100:.2f}%"
    )

    print(
        f"Macro Recall      : "
        f"{metrics['macro_recall'] * 100:.2f}%"
    )

    print(
        f"Macro F1          : "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Weighted F1       : "
        f"{metrics['weighted_f1'] * 100:.2f}%"
    )


# ============================================================
# Print per-class metrics
# ============================================================

def print_class_metrics(
    metrics,
):

    print()
    print(
        "Per-Class Test Metrics"
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
            f"{metrics['class_precision'][label_id] * 100:11.2f}%"
            f"{metrics['class_recall'][label_id] * 100:11.2f}%"
            f"{metrics['class_f1'][label_id] * 100:11.2f}%"
            f"{metrics['class_support'][label_id]:12d}"
        )


# ============================================================
# Print confusion matrix
# ============================================================

def print_confusion_matrix(
    cm,
):

    print()
    print(
        "Test Confusion Matrix"
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
# Save confusion matrix
# ============================================================

def save_confusion_matrix(
    cm,
):

    with CONFUSION_MATRIX_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Actual \\ Predicted",
                "Genuine",
                "TTS",
                "Voice_Conversion",
            ]
        )

        for label_id in range(
            NUM_CLASSES
        ):

            writer.writerow(
                [
                    CLASS_NAMES[label_id],
                    int(
                        cm[
                            label_id,
                            0
                        ]
                    ),
                    int(
                        cm[
                            label_id,
                            1
                        ]
                    ),
                    int(
                        cm[
                            label_id,
                            2
                        ]
                    ),
                ]
            )


# ============================================================
# Save class metrics
# ============================================================

def save_class_metrics(
    metrics,
):

    with CLASS_METRICS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "label_id",
                "class_name",
                "precision",
                "recall",
                "f1",
                "support",
            ]
        )

        for label_id in range(
            NUM_CLASSES
        ):

            writer.writerow(
                [
                    label_id,
                    CLASS_NAMES[label_id],
                    float(
                        metrics[
                            "class_precision"
                        ][label_id]
                    ),
                    float(
                        metrics[
                            "class_recall"
                        ][label_id]
                    ),
                    float(
                        metrics[
                            "class_f1"
                        ][label_id]
                    ),
                    int(
                        metrics[
                            "class_support"
                        ][label_id]
                    ),
                ]
            )


# ============================================================
# Save JSON results
# ============================================================

def save_json_results(
    metrics,
    checkpoint,
    checkpoint_path,
    total_test_samples,
):

    result = {
        "project": "DeepShield",
        "version": "V3",
        "evaluation_type": "held_out_test",

        "checkpoint": {
            "path": str(
                checkpoint_path
            ),
            "epoch": checkpoint.get(
                "epoch"
            ),
            "dev_balanced_accuracy":
                checkpoint.get(
                    "best_metric"
                ),
        },

        "dataset": {
            "split": "test",
            "samples":
                total_test_samples,
            "num_classes":
                NUM_CLASSES,
            "class_names":
                CLASS_NAMES,
        },

        "overall_metrics": {
            "accuracy":
                metrics["accuracy"],
            "balanced_accuracy":
                metrics["balanced_accuracy"],
            "macro_precision":
                metrics["macro_precision"],
            "macro_recall":
                metrics["macro_recall"],
            "macro_f1":
                metrics["macro_f1"],
            "weighted_f1":
                metrics["weighted_f1"],
        },

        "per_class_metrics": {
            CLASS_NAMES[label_id]: {
                "precision":
                    float(
                        metrics[
                            "class_precision"
                        ][label_id]
                    ),
                "recall":
                    float(
                        metrics[
                            "class_recall"
                        ][label_id]
                    ),
                "f1":
                    float(
                        metrics[
                            "class_f1"
                        ][label_id]
                    ),
                "support":
                    int(
                        metrics[
                            "class_support"
                        ][label_id]
                    ),
            }

            for label_id in range(
                NUM_CLASSES
            )
        },

        "confusion_matrix":
            metrics[
                "confusion_matrix"
            ].tolist(),
    }

    with RESULTS_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DeepShield V3 - Held-Out Test Evaluation")
    print("=" * 70)

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

    # --------------------------------------------------------
    # Create evaluation directory
    # --------------------------------------------------------

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load TEST dataset only
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Loading V3 TEST dataset")
    print("=" * 70)

    test_dataset = DeepShieldV3Dataset(
        "test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    print()
    print(
        f"Test samples : "
        f"{len(test_dataset)}"
    )

    print(
        f"Test batches : "
        f"{len(test_loader)}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Creating V3 model for evaluation")
    print("=" * 70)

    model = DeepShieldV3(
        num_classes=NUM_CLASSES
    )

    print()
    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    (
        checkpoint,
        checkpoint_path,
    ) = load_best_checkpoint(
        model,
        device,
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Evaluating V3 on HELD-OUT TEST SET")
    print("=" * 70)

    print()
    print(
        "No training or weight updates will occur."
    )

    (
        targets,
        predictions,
        probabilities,
    ) = evaluate(
        model,
        test_loader,
        device,
    )

    # --------------------------------------------------------
    # Validate sample count
    # --------------------------------------------------------

    if len(targets) != len(
        test_dataset
    ):

        raise RuntimeError(
            "Evaluation sample count mismatch."
        )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        targets,
        predictions,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_overall_metrics(
        metrics
    )

    print_class_metrics(
        metrics
    )

    print_confusion_matrix(
        metrics[
            "confusion_matrix"
        ]
    )

    # --------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------

    # First make sure no NaN or infinity exists.
    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            "Evaluation probabilities "
            "contain NaN or infinity."
        )

    # Softmax probabilities should sum to 1.
    #
    # Because CUDA/AMP may use reduced precision during
    # inference, allow a small floating-point tolerance.
    probability_sums = (
        probabilities.sum(
            axis=1
        )
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-3,
    ):

        max_deviation = float(
            np.max(
                np.abs(
                    probability_sums - 1.0
                )
            )
        )

        raise RuntimeError(
            "Softmax probabilities have an "
            "unexpected deviation from 1. "
            f"Maximum deviation: "
            f"{max_deviation:.8f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_confusion_matrix(
        metrics[
            "confusion_matrix"
        ]
    )

    save_class_metrics(
        metrics
    )

    save_json_results(
        metrics,
        checkpoint,
        checkpoint_path,
        len(test_dataset),
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V3 TEST EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Results saved:"
    )

    print(
        f"  {RESULTS_JSON}"
    )

    print(
        f"  {CONFUSION_MATRIX_CSV}"
    )

    print(
        f"  {CLASS_METRICS_CSV}"
    )

    print()
    print(
        "Test set was used only for evaluation."
    )

    print(
        "No training or model updates were performed."
    )

    print()
    print(
        "V3 HELD-OUT TEST EVALUATION: PASS"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()