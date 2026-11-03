"""
DeepShield V3.2 Evaluation

Evaluates the V3.2 BEST checkpoint on:

1. Original untouched V3 test set
2. Original 20-sample gTTS blind test

IMPORTANT:
- No training is performed.
- V3 test data is used ONLY for evaluation.
- gTTS blind-test files are used ONLY for evaluation.
- No production files are modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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

CHECKPOINT_PATH = (
    V31_ROOT
    / "checkpoints"
    / "deepshield_v3_2_best.pth"
)

GTTS_BLIND_DIR = (
    PROJECT_ROOT
    / "v3_1_dataset_inspection"
    / "gtts_test"
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
        (num_classes, num_classes),
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
        np.trace(confusion) / total
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
                tp / (tp + fp)
            )
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = (
                tp / (tp + fn)
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
        np.mean(recalls)
    )

    macro_precision = float(
        np.mean(precisions)
    )

    macro_recall = float(
        np.mean(recalls)
    )

    macro_f1 = float(
        np.mean(f1_scores)
    )

    supports = np.asarray(
        supports,
        dtype=np.float64,
    )

    total_support = supports.sum()

    weighted_f1 = float(
        np.sum(
            np.asarray(f1_scores)
            * supports
        )
        / total_support
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "macro_precision": (
            macro_precision
        ),
        "macro_recall": (
            macro_recall
        ),
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
# Print metrics
# ============================================================

def print_metrics(
    title,
    metrics,
):

    print()
    print("=" * 70)

    print(title)

    print("=" * 70)

    print(
        f"Accuracy:          "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro Precision:   "
        f"{metrics['macro_precision'] * 100:.2f}%"
    )

    print(
        f"Macro Recall:      "
        f"{metrics['macro_recall'] * 100:.2f}%"
    )

    print(
        f"Macro F1:          "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Weighted F1:       "
        f"{metrics['weighted_f1'] * 100:.2f}%"
    )


# ============================================================
# Print confusion matrix
# ============================================================

def print_confusion_matrix(
    confusion,
):

    print()
    print("Confusion Matrix:")

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
# Load model
# ============================================================

def load_model(
    device,
):

    print()
    print(
        "Loading V3.2 BEST checkpoint..."
    )

    print(
        f"Checkpoint:"
        f"\n{CHECKPOINT_PATH}"
    )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "V3.2 best checkpoint was not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    model = DeepShieldV3(
        num_classes=config.NUM_CLASSES
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Checkpoint best "
        f"{config.PRIMARY_METRIC}: "
        f"{checkpoint.get('best_metric', 0) * 100:.2f}%"
    )

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# Evaluate dataset
# ============================================================

@torch.no_grad()
def evaluate_dataset(
    model,
    loader,
    device,
):

    all_targets = []
    all_predictions = []

    for features, labels in loader:

        features = features.to(
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

        all_targets.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

    return calculate_metrics(
        all_targets,
        all_predictions,
        3,
    )


# ============================================================
# Evaluate original V3 test
# ============================================================

def evaluate_v3_test(
    model,
    device,
):
    """
    Evaluate V3.2 on the original untouched V3 test set.

    The original V3 test manifest uses:
        label_id = 0 -> Genuine
        label_id = 1 -> TTS
        label_id = 2 -> Voice_Conversion

    The original V3 test features are loaded from:
        v3/features/test/
    """

    print()
    print(
        "Loading ORIGINAL V3 TEST SET..."
    )

    # --------------------------------------------------------
    # Import pandas
    # --------------------------------------------------------

    import pandas as pd

    # --------------------------------------------------------
    # Original V3 test manifest
    # --------------------------------------------------------

    test_manifest = (
        V3_ROOT
        / "dataset"
        / "manifests"
        / "v3_test.csv"
    )

    if not test_manifest.exists():

        raise FileNotFoundError(
            "Original V3 test manifest not found:\n"
            f"{test_manifest}"
        )

    df = pd.read_csv(
        test_manifest
    )

    # --------------------------------------------------------
    # Validate manifest columns
    # --------------------------------------------------------

    required_columns = {
        "audio_path",
        "label",
        "label_name",
        "label_id",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Original V3 test manifest "
            f"is missing columns: {missing}"
        )

    print(
        f"Test samples: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Verify original V3 test distribution
    # --------------------------------------------------------

    class_counts = (
        df["label_id"]
        .astype(int)
        .value_counts()
        .sort_index()
    )

    print(
        "Test class distribution:"
    )

    for label, count in class_counts.items():

        name = {
            0: "Genuine",
            1: "TTS",
            2: "Voice_Conversion",
        }.get(
            int(label),
            f"Unknown({label})",
        )

        print(
            f"  {int(label)} "
            f"{name:18s}: "
            f"{int(count):,}"
        )

    # --------------------------------------------------------
    # Original V3 test dataset
    # --------------------------------------------------------

    class OriginalV3TestDataset(
        torch.utils.data.Dataset
    ):

        def __init__(
            self,
            dataframe,
        ):

            self.df = (
                dataframe.reset_index(
                    drop=True
                )
            )

        def __len__(
            self,
        ):

            return len(
                self.df
            )

        def __getitem__(
            self,
            index,
        ):

            row = self.df.iloc[
                index
            ]

            audio_path = str(
                row["audio_path"]
            )

            audio_stem = (
                Path(audio_path)
                .stem
            )

            feature_path = (
                V3_ROOT
                / "features"
                / "test"
                / f"{audio_stem}.npy"
            )

            if not feature_path.exists():

                raise FileNotFoundError(
                    "V3 test feature not found:\n"
                    f"{feature_path}"
                )

            feature = np.load(
                feature_path
            )

            if feature.shape != (
                128,
                251,
            ):

                raise ValueError(
                    "Invalid V3 test feature "
                    f"shape {feature.shape}:\n"
                    f"{feature_path}"
                )

            feature = feature.astype(
                np.float32,
                copy=False,
            )

            if not np.isfinite(
                feature
            ).all():

                raise ValueError(
                    "Non-finite V3 test feature:\n"
                    f"{feature_path}"
                )

            tensor = torch.from_numpy(
                feature
            ).unsqueeze(0)

            # IMPORTANT:
            # Use label_id, not the ASVspoof
            # "bonafide"/"spoof" label.
            label = torch.tensor(
                int(row["label_id"]),
                dtype=torch.long,
            )

            return (
                tensor,
                label,
            )

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    test_dataset = (
        OriginalV3TestDataset(
            df
        )
    )

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_dataset(
        model,
        test_loader,
        device,
    )

    # --------------------------------------------------------
    # Print overall metrics
    # --------------------------------------------------------

    print_metrics(
        "V3.2 on ORIGINAL V3 TEST SET",
        metrics,
    )

    # --------------------------------------------------------
    # Print confusion matrix
    # --------------------------------------------------------

    print_confusion_matrix(
        metrics[
            "confusion_matrix"
        ]
    )

    # --------------------------------------------------------
    # Print per-class metrics
    # --------------------------------------------------------

    print()
    print(
        "Per-class results:"
    )

    names = [
        "Genuine",
        "TTS",
        "Voice_Conversion",
    ]

    for index, name in enumerate(
        names
    ):

        print(
            f"{name:18s}"
            f"Precision: "
            f"{metrics['class_precision'][index] * 100:.2f}%   "
            f"Recall: "
            f"{metrics['class_recall'][index] * 100:.2f}%   "
            f"F1: "
            f"{metrics['class_f1'][index] * 100:.2f}%   "
            f"Support: "
            f"{metrics['class_support'][index]}"
        )

    return metrics


# ============================================================
# Load one feature
# ============================================================

def load_feature(
    path,
):

    feature = np.load(
        path
    )

    if feature.shape != (
        128,
        251,
    ):

        raise ValueError(
            f"Invalid feature shape "
            f"{feature.shape}: {path}"
        )

    feature = feature.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        feature
    ).all():

        raise ValueError(
            f"Non-finite feature: "
            f"{path}"
        )

    tensor = torch.from_numpy(
        feature
    ).unsqueeze(0)

    return tensor


# ============================================================
# Evaluate gTTS blind test
# ============================================================

@torch.no_grad()
def evaluate_gtts_blind(
    model,
    device,
):
    """
    Evaluate the untouched 20-file gTTS blind test.

    Input:
        C:\\DeepShield\\v3_1_dataset_inspection\\gtts_test\\*.mp3

    Preprocessing:
        Same feature extraction used by DeepShield V3/V3.2:
            - 16 kHz
            - mono
            - 4 seconds
            - n_fft=1024
            - hop_length=256
            - n_mels=128
            - power spectrogram
            - dB conversion
            - per-file normalization

    IMPORTANT:
        These files are evaluation-only.
        They are NOT part of the training dataset.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "gTTS BLIND TEST"
    )

    print(
        "=" * 70
    )

    if not GTTS_BLIND_DIR.exists():

        raise FileNotFoundError(
            "gTTS blind-test directory not found:\n"
            f"{GTTS_BLIND_DIR}"
        )

    # --------------------------------------------------------
    # Import audio / feature dependencies
    # --------------------------------------------------------

    import librosa

    # --------------------------------------------------------
    # Find original MP3 files
    # --------------------------------------------------------

    files = sorted(
        GTTS_BLIND_DIR.glob(
            "*.mp3"
        )
    )

    if len(files) == 0:

        raise RuntimeError(
            "No .mp3 gTTS blind-test files found in:\n"
            f"{GTTS_BLIND_DIR}"
        )

    print(
        f"Blind-test samples: "
        f"{len(files)}"
    )

    # --------------------------------------------------------
    # Expected blind-test count
    # --------------------------------------------------------

    if len(files) != 20:

        print(
            "WARNING:"
        )

        print(
            "Expected 20 blind-test files, "
            f"but found {len(files)}."
        )

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    names = [
        "Genuine",
        "TTS",
        "Voice_Conversion",
    ]

    counts = {
        "Genuine": 0,
        "TTS": 0,
        "Voice_Conversion": 0,
    }

    results = []

    # --------------------------------------------------------
    # Process every blind-test file
    # --------------------------------------------------------

    for index, path in enumerate(
        files,
        start=1,
    ):

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio, _ = librosa.load(
            path,
            sr=16000,
            mono=True,
        )

        # ----------------------------------------------------
        # Match DeepShield's 4-second input
        # ----------------------------------------------------

        target_length = (
            16000 * 4
        )

        if len(audio) < target_length:

            audio = np.pad(
                audio,
                (
                    0,
                    target_length
                    - len(audio),
                ),
                mode="constant",
            )

        elif len(audio) > target_length:

            audio = audio[
                :target_length
            ]

        # ----------------------------------------------------
        # Mel spectrogram
        # ----------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            power=2.0,
        )

        # ----------------------------------------------------
        # Convert to dB
        # ----------------------------------------------------

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max,
        )

        # ----------------------------------------------------
        # Per-file normalization
        # ----------------------------------------------------

        mean = mel_db.mean()

        std = mel_db.std()

        if std > 1e-8:

            mel_db = (
                mel_db - mean
            ) / std

        else:

            mel_db = (
                mel_db - mean
            )

        # ----------------------------------------------------
        # Verify expected feature size
        # ----------------------------------------------------

        if mel_db.shape != (
            128,
            251,
        ):

            raise ValueError(
                "Unexpected feature shape "
                f"{mel_db.shape} for:\n"
                f"{path}"
            )

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        feature = torch.tensor(
            mel_db,
            dtype=torch.float32,
        )

        feature = feature.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        feature = feature.to(
            device
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        outputs = model(
            feature
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

        prediction = int(
            torch.argmax(
                probabilities
            ).item()
        )

        predicted_name = names[
            prediction
        ]

        confidence = float(
            probabilities[
                prediction
            ].item()
        )

        counts[
            predicted_name
        ] += 1

        results.append(
            (
                path.name,
                predicted_name,
                confidence,
            )
        )

        print(
            f"{index:02d}. "
            f"{path.name:20s} "
            f"-> "
            f"{predicted_name:17s} "
            f"{confidence * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(files)

    tts_count = counts[
        "TTS"
    ]

    detection_rate = (
        tts_count
        / total
        * 100
    )

    print()
    print(
        "gTTS Blind-Test Summary:"
    )

    print(
        f"  Total:              "
        f"{total}"
    )

    print(
        f"  Genuine:            "
        f"{counts['Genuine']}"
    )

    print(
        f"  TTS:                "
        f"{counts['TTS']}"
    )

    print(
        f"  Voice_Conversion:   "
        f"{counts['Voice_Conversion']}"
    )

    print(
        f"  TTS detection rate: "
        f"{detection_rate:.2f}%"
    )

    return {
        "total": total,
        "counts": counts,
        "detection_rate": detection_rate,
        "results": results,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "DeepShield V3.2 Evaluation"
    )

    print("=" * 70)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda"
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
    # Load V3.2 best model
    # --------------------------------------------------------

    model = load_model(
        device
    )

    # --------------------------------------------------------
    # Evaluation 1:
    # Original V3 test
    # --------------------------------------------------------

    test_metrics = (
        evaluate_v3_test(
            model,
            device,
        )
    )

    # --------------------------------------------------------
    # Evaluation 2:
    # gTTS blind test
    # --------------------------------------------------------

    gtts_results = (
        evaluate_gtts_blind(
            model,
            device,
        )
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "V3.2 EVALUATION COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        "Original V3 test:"
    )

    print(
        f"  Accuracy: "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "gTTS blind test:"
    )

    print(
        f"  Total: "
        f"{gtts_results['total']}"
    )

    print(
        f"  TTS detected: "
        f"{gtts_results['counts']['TTS']}"
    )

    print(
        f"  Genuine: "
        f"{gtts_results['counts']['Genuine']}"
    )

    print(
        f"  Voice Conversion: "
        f"{gtts_results['counts']['Voice_Conversion']}"
    )

    print(
        f"  Detection rate: "
        f"{gtts_results['detection_rate']:.2f}%"
    )

    print()
    print(
        "No training was performed."
    )

    print(
        "No production files were modified."
    )

    print(
        "V3/V3.1 checkpoints were not modified."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()