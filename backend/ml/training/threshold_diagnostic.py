import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from ml.training.model import DeepShieldCNN
from ml.training.feature_extractor import (
    extract_mel_spectrogram,
)


# ============================================================
# Configuration
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "eval_manifest.csv"
)

MODEL_PATH = (
    BACKEND_DIR
    / "ml"
    / "models"
    / "deepshield_cnn_best.pth"
)

CLASS_NAMES = [
    "Genuine",
    "TTS",
    "Voice Conversion",
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Configuration for diagnostic size
# ============================================================

SAMPLES_PER_GROUP = 300

RANDOM_STATE = 42


# ============================================================
# Load model
# ============================================================

print()
print("=" * 80)
print("DeepShield Threshold Diagnostic")
print("=" * 80)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Model:",
    MODEL_PATH
)


model = DeepShieldCNN(
    num_classes=3
).to(DEVICE)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
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
# Load evaluation manifest
# ============================================================

df = pd.read_csv(
    MANIFEST_PATH
)


# ------------------------------------------------------------
# Select representative groups
#
# Genuine: attack_id "-"
# TTS:     A07-A16
# VC:      A17-A19
# ------------------------------------------------------------

genuine = (
    df[df["attack_id"] == "-"]
    .sample(
        n=min(
            SAMPLES_PER_GROUP,
            len(df[df["attack_id"] == "-"])
        ),
        random_state=RANDOM_STATE,
    )
)


tts = (
    df[
        df["attack_id"].isin(
            [
                "A07",
                "A08",
                "A09",
                "A10",
                "A11",
                "A12",
                "A13",
                "A14",
                "A15",
                "A16",
            ]
        )
    ]
    .sample(
        n=min(
            SAMPLES_PER_GROUP,
            len(
                df[
                    df["attack_id"].isin(
                        [
                            "A07",
                            "A08",
                            "A09",
                            "A10",
                            "A11",
                            "A12",
                            "A13",
                            "A14",
                            "A15",
                            "A16",
                        ]
                    )
                ]
            ),
        ),
        random_state=RANDOM_STATE,
    )
)


vc = (
    df[
        df["attack_id"].isin(
            [
                "A17",
                "A18",
                "A19",
            ]
        )
    ]
    .sample(
        n=min(
            SAMPLES_PER_GROUP,
            len(
                df[
                    df["attack_id"].isin(
                        [
                            "A17",
                            "A18",
                            "A19",
                        ]
                    )
                ]
            ),
        ),
        random_state=RANDOM_STATE,
    )
)


selected = pd.concat(
    [
        genuine,
        tts,
        vc,
    ],
    ignore_index=True,
)


print()
print(
    "Diagnostic samples:",
    len(selected)
)

print()
print(
    selected.groupby(
        ["attack_id", "label"]
    ).size().to_string()
)


# ============================================================
# Run inference
# ============================================================

results = []


print()
print("=" * 80)
print("Running inference")
print("=" * 80)


with torch.no_grad():

    for counter, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        try:

            mel = extract_mel_spectrogram(
                row["audio_path"]
            )

            features = torch.from_numpy(
                mel
            ).float()

            features = (
                features
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            outputs = model(
                features
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[0]

            probs = (
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

            results.append(
                {
                    "audio_id": row["audio_id"],
                    "attack_id": row["attack_id"],
                    "true_label": row["label"],
                    "true_label_id": int(
                        row["label_id"]
                    ),
                    "genuine_prob": float(
                        probs[0]
                    ),
                    "tts_prob": float(
                        probs[1]
                    ),
                    "vc_prob": float(
                        probs[2]
                    ),
                }
            )

        except Exception as error:

            print()
            print(
                "ERROR:",
                row["audio_id"]
            )

            print(
                "Reason:",
                error
            )

        if counter % 100 == 0:

            print(
                f"Processed {counter}/{len(selected)}"
            )


results_df = pd.DataFrame(
    results
)


# ============================================================
# Save raw diagnostic results
# ============================================================

output_path = (
    BACKEND_DIR
    / "ml"
    / "evaluation"
    / "threshold_diagnostic.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


# ============================================================
# Baseline 3-class prediction
# ============================================================

results_df["baseline_prediction"] = (
    results_df[
        [
            "genuine_prob",
            "tts_prob",
            "vc_prob",
        ]
    ]
    .idxmax(axis=1)
    .map(
        {
            "genuine_prob": "Genuine",
            "tts_prob": "TTS",
            "vc_prob": "Voice Conversion",
        }
    )
)


print()
print("=" * 80)
print("BASELINE 3-CLASS RESULTS")
print("=" * 80)

print()

print(
    pd.crosstab(
        results_df["attack_id"],
        results_df["baseline_prediction"],
    ).to_string()
)


# ============================================================
# Binary Deepfake threshold experiment
#
# Deepfake probability:
#
# TTS + Voice Conversion
#
# If:
#
# synthetic_prob >= threshold
#
# => Deepfake
#
# otherwise:
#
# => Genuine
# ============================================================

results_df["deepfake_prob"] = (
    results_df["tts_prob"]
    + results_df["vc_prob"]
)


true_binary = (
    results_df["true_label_id"] != 0
).astype(int)


print()
print("=" * 80)
print("BINARY GENUINE vs DEEPFAKE THRESHOLD TEST")
print("=" * 80)

print()
print(
    "Threshold | Accuracy | Balanced Acc | Precision | Recall | F1 | Genuine FPR"
)
print(
    "-" * 78
)


best_row = None


for threshold in [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]:

    predicted_binary = (
        results_df["deepfake_prob"]
        >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        true_binary,
        predicted_binary,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            true_binary,
            predicted_binary,
        )
    )

    precision = precision_score(
        true_binary,
        predicted_binary,
        zero_division=0,
    )

    recall = recall_score(
        true_binary,
        predicted_binary,
        zero_division=0,
    )

    f1 = f1_score(
        true_binary,
        predicted_binary,
        zero_division=0,
    )

    genuine_mask = (
        results_df["true_label_id"] == 0
    )

    genuine_false_positive_rate = (
        predicted_binary[genuine_mask].mean()
    )

    print(
        f"{threshold:9.2f} | "
        f"{accuracy * 100:8.2f}% | "
        f"{balanced_accuracy * 100:12.2f}% | "
        f"{precision * 100:9.2f}% | "
        f"{recall * 100:6.2f}% | "
        f"{f1 * 100:5.2f}% | "
        f"{genuine_false_positive_rate * 100:10.2f}%"
    )

    candidate = {
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "genuine_fpr": genuine_false_positive_rate,
    }

    if (
        best_row is None
        or candidate["balanced_accuracy"]
        > best_row["balanced_accuracy"]
    ):

        best_row = candidate


# ============================================================
# Best threshold
# ============================================================

print()
print("=" * 80)
print("BEST BINARY THRESHOLD")
print("=" * 80)

print()

print(
    f"Threshold: "
    f"{best_row['threshold']:.2f}"
)

print(
    f"Accuracy: "
    f"{best_row['accuracy'] * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{best_row['balanced_accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{best_row['precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{best_row['recall'] * 100:.2f}%"
)

print(
    f"F1: "
    f"{best_row['f1'] * 100:.2f}%"
)

print(
    f"Genuine False Positive Rate: "
    f"{best_row['genuine_fpr'] * 100:.2f}%"
)


# ============================================================
# Attack-wise binary results
# ============================================================

threshold = best_row["threshold"]

results_df["binary_prediction"] = (
    results_df["deepfake_prob"]
    >= threshold
).astype(int)


print()
print("=" * 80)
print("ATTACK-WISE DEEPFAKE DETECTION")
print("=" * 80)

print()

attack_results = []

for attack_id, group in results_df.groupby(
    "attack_id"
):

    actual = (
        group["true_label_id"] != 0
    ).astype(int)

    predicted = group[
        "binary_prediction"
    ]

    attack_results.append(
        {
            "attack_id": attack_id,
            "samples": len(group),
            "deepfake_detection": (
                predicted.mean() * 100
            ),
        }
    )


attack_results_df = pd.DataFrame(
    attack_results
)

print(
    attack_results_df.to_string(
        index=False
    )
)


# ============================================================
# Save updated diagnostic
# ============================================================

results_df.to_csv(
    output_path,
    index=False,
)


print()
print("=" * 80)
print("DIAGNOSTIC COMPLETED")
print("=" * 80)

print()
print(
    "Results:",
    output_path
)