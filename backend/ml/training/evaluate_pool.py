import sys
from pathlib import Path
import json

import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt


# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from ml.training.audio_dataset import DeepShieldAudioDataset
from ml.training.model_pool import DeepShieldCNNPool


# ============================================================
# Configuration
# ============================================================

EVAL_MANIFEST = (
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "eval_manifest.csv"
)

MODEL_PATH = (
    BACKEND_DIR
    / "ml"
    / "models"
    / "deepshield_cnn_pool_augmented_best.pth"
)
OUTPUT_DIR = (
    BACKEND_DIR
    / "ml"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_SIZE = 8

NUM_CLASSES = 3

CLASS_NAMES = [
    "Genuine",
    "TTS",
    "Voice Conversion",
]

CLASS_IDS = [
    0,
    1,
    2,
]


# ============================================================
# Output files
# ============================================================

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "eval_pool_confusion_matrix.png"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "eval_pool_classification_metrics.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "eval_pool_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "eval_pool_summary.json"
)

ATTACK_METRICS_PATH = (
    OUTPUT_DIR
    / "eval_pool_attack_metrics.csv"
)


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
print("DeepShield Dual-Pooling Final Evaluation")
print("=" * 70)
print()

print(
    "Device:",
    device,
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# Verify files
# ============================================================

if not EVAL_MANIFEST.exists():

    raise FileNotFoundError(
        f"Evaluation manifest not found:\n"
        f"{EVAL_MANIFEST}"
    )

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Dual-pooling model not found:\n"
        f"{MODEL_PATH}\n\n"
        f"Run train_pool.py first."
    )


# ============================================================
# Load evaluation manifest
# ============================================================

print()
print("Loading evaluation manifest...")

dataframe = pd.read_csv(
    EVAL_MANIFEST
)

print(
    "Evaluation samples:",
    len(dataframe),
)


# ============================================================
# Class distribution
# ============================================================

print()
print("Evaluation class distribution:")

class_distribution = (
    dataframe["label_id"]
    .value_counts()
    .sort_index()
)

for class_id in CLASS_IDS:

    count = int(
        class_distribution.get(
            class_id,
            0,
        )
    )

    print(
        f"{CLASS_NAMES[class_id]}: {count}"
    )


# ============================================================
# Attack distribution
# ============================================================

if "attack_id" in dataframe.columns:

    print()
    print("Evaluation attack distribution:")

    attack_distribution = (
        dataframe["attack_id"]
        .value_counts()
        .sort_index()
    )

    for attack_id, count in (
        attack_distribution.items()
    ):

        print(
            f"{attack_id}: {count}"
        )


# ============================================================
# Dataset
# ============================================================

print()
print("Creating evaluation dataset...")

dataset = DeepShieldAudioDataset(
    dataframe,
    split="eval",
)


# ============================================================
# DataLoader
# ============================================================

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(
    "Evaluation batches:",
    len(dataloader),
)


# ============================================================
# Model
# ============================================================

model = DeepShieldCNNPool(
    num_classes=NUM_CLASSES
).to(device)


# ============================================================
# Load checkpoint
# ============================================================

print()
print("Loading dual-pooling model:")
print(MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# Inference
# ============================================================

all_labels = []
all_predictions = []


print()
print("Running DUAL-POOLING FINAL evaluation...")
print()


with torch.no_grad():

    for batch_index, (
        features,
        labels,
    ) in enumerate(dataloader):

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(
            features
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"Processed "
                f"{batch_index + 1}/"
                f"{len(dataloader)} batches"
            )


# ============================================================
# Safety check
# ============================================================

if len(all_labels) != len(dataframe):

    raise RuntimeError(
        "Number of predictions does not "
        "match number of evaluation samples."
    )


# ============================================================
# Overall metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

balanced_accuracy = (
    balanced_accuracy_score(
        all_labels,
        all_predictions,
    )
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=CLASS_IDS,
        zero_division=0,
    )
)

macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()

weighted_precision = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=CLASS_IDS,
        average="weighted",
        zero_division=0,
    )[0]
)

weighted_recall = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=CLASS_IDS,
        average="weighted",
        zero_division=0,
    )[1]
)

weighted_f1 = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=CLASS_IDS,
        average="weighted",
        zero_division=0,
    )[2]
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 70)
print("DUAL-POOLING FINAL EVALUATION RESULTS")
print("=" * 70)
print()

print(
    f"Total Samples: "
    f"{len(all_labels)}"
)

print()

print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision: "
    f"{macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall: "
    f"{macro_recall * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{macro_f1 * 100:.2f}%"
)

print()

print(
    f"Weighted Precision: "
    f"{weighted_precision * 100:.2f}%"
)

print(
    f"Weighted Recall: "
    f"{weighted_recall * 100:.2f}%"
)

print(
    f"Weighted F1: "
    f"{weighted_f1 * 100:.2f}%"
)


# ============================================================
# Per-class
# ============================================================

print()
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

for index, class_name in enumerate(
    CLASS_NAMES
):

    print()

    print(
        class_name
    )

    print(
        f"  Support:   "
        f"{support[index]}"
    )

    print(
        f"  Precision: "
        f"{precision[index] * 100:.2f}%"
    )

    print(
        f"  Recall:    "
        f"{recall[index] * 100:.2f}%"
    )

    print(
        f"  F1 Score:  "
        f"{f1[index] * 100:.2f}%"
    )


# ============================================================
# Classification report
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=CLASS_IDS,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print()
print(report)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=CLASS_IDS,
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print()
print(cm)


# ============================================================
# Confusion matrix plot
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest",
)

plt.title(
    "DeepShield Dual-Pooling Evaluation Confusion Matrix"
)

plt.colorbar()

tick_marks = range(
    len(CLASS_NAMES)
)

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=30,
)

plt.yticks(
    tick_marks,
    CLASS_NAMES,
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

for i in range(
    len(CLASS_NAMES)
):

    for j in range(
        len(CLASS_NAMES)
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
        )

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=150,
)

plt.close()


# ============================================================
# Classification metrics CSV
# ============================================================

metrics_data = []

for index, class_name in enumerate(
    CLASS_NAMES
):

    metrics_data.append(
        {
            "class": class_name,
            "support": int(
                support[index]
            ),
            "precision": float(
                precision[index]
            ),
            "recall": float(
                recall[index]
            ),
            "f1_score": float(
                f1[index]
            ),
        }
    )


pd.DataFrame(
    metrics_data
).to_csv(
    METRICS_PATH,
    index=False,
)


# ============================================================
# Predictions
# ============================================================

prediction_dataframe = (
    dataframe.copy()
)

prediction_dataframe[
    "predicted_label_id"
] = all_predictions

prediction_dataframe[
    "predicted_label"
] = [
    CLASS_NAMES[prediction]
    for prediction in all_predictions
]

prediction_dataframe[
    "correct"
] = (
    prediction_dataframe[
        "label_id"
    ].values
    ==
    prediction_dataframe[
        "predicted_label_id"
    ].values
)

prediction_dataframe.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# Per-attack evaluation
# ============================================================

attack_metrics = []

if "attack_id" in prediction_dataframe.columns:

    for attack_id in sorted(
        prediction_dataframe[
            "attack_id"
        ].unique()
    ):

        attack_dataframe = (
            prediction_dataframe[
                prediction_dataframe[
                    "attack_id"
                ] == attack_id
            ]
        )

        attack_labels = (
            attack_dataframe[
                "label_id"
            ]
            .astype(int)
            .tolist()
        )

        attack_predictions = (
            attack_dataframe[
                "predicted_label_id"
            ]
            .astype(int)
            .tolist()
        )

        attack_accuracy = accuracy_score(
            attack_labels,
            attack_predictions,
        )

        attack_balanced_accuracy = (
            balanced_accuracy_score(
                attack_labels,
                attack_predictions,
            )
        )

        attack_precision, \
        attack_recall, \
        attack_f1, \
        attack_support = (
            precision_recall_fscore_support(
                attack_labels,
                attack_predictions,
                labels=CLASS_IDS,
                zero_division=0,
            )
        )

        attack_metrics.append(
            {
                "attack_id": attack_id,
                "samples": len(
                    attack_dataframe
                ),
                "accuracy":
                    attack_accuracy,
                "balanced_accuracy":
                    attack_balanced_accuracy,
                "macro_precision":
                    attack_precision.mean(),
                "macro_recall":
                    attack_recall.mean(),
                "macro_f1":
                    attack_f1.mean(),
            }
        )


pd.DataFrame(
    attack_metrics
).to_csv(
    ATTACK_METRICS_PATH,
    index=False,
)


# ============================================================
# Summary JSON
# ============================================================

summary = {

    "evaluation_set": {
        "manifest": str(
            EVAL_MANIFEST
        ),
        "samples": len(
            all_labels
        ),
    },

    "model": {
        "path": str(
            MODEL_PATH
        ),
        "architecture": "DeepShieldCNNPool",
        "device": str(
            device
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    },

    "overall": {

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(balanced_accuracy),

        "macro_precision":
            float(macro_precision),

        "macro_recall":
            float(macro_recall),

        "macro_f1":
            float(macro_f1),

        "weighted_precision":
            float(weighted_precision),

        "weighted_recall":
            float(weighted_recall),

        "weighted_f1":
            float(weighted_f1),
    },

    "classes": {},
}


for index, class_name in enumerate(
    CLASS_NAMES
):

    summary["classes"][
        class_name
    ] = {

        "support":
            int(support[index]),

        "precision":
            float(precision[index]),

        "recall":
            float(recall[index]),

        "f1_score":
            float(f1[index]),
    }


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=4,
    )


# ============================================================
# Final output
# ============================================================

print()
print("=" * 70)
print("DUAL-POOLING EVALUATION COMPLETED")
print("=" * 70)
print()

print(
    "Confusion matrix:"
)

print(
    CONFUSION_MATRIX_PATH
)

print()

print(
    "Classification metrics:"
)

print(
    METRICS_PATH
)

print()

print(
    "Predictions:"
)

print(
    PREDICTIONS_PATH
)

print()

print(
    "Attack metrics:"
)

print(
    ATTACK_METRICS_PATH
)

print()

print(
    "Summary:"
)

print(
    SUMMARY_PATH
)

print()
print("=" * 70)
