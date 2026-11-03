"""
DeepShield V3.1 - Final Test Evaluation

Purpose:
    Evaluate the V3.1 BEST checkpoint on the ORIGINAL V3 TEST SET.

Important:
    - No training is performed.
    - Original V3 test manifest is used.
    - Original V3 test features are used.
    - V3 test set is NOT modified.
    - V3 model architecture is reused unchanged.
    - V3.1 best checkpoint is loaded.
    - No V3/V3.1 checkpoint is modified.
"""

import sys
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

V3_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "experiments"
    / "v3"
)

V31_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "experiments"
    / "v3_1"
)

# Original V3 test manifest
V3_TEST_MANIFEST = (
    V3_ROOT
    / "dataset"
    / "manifests"
    / "v3_test.csv"
)

# Original V3 test feature cache
V3_TEST_FEATURE_DIR = (
    V3_ROOT
    / "features"
    / "test"
)

# V3.1 best checkpoint
V31_BEST_CHECKPOINT = (
    V31_ROOT
    / "checkpoints"
    / "deepshield_v3_1_best.pth"
)


# ============================================================
# IMPORT EXISTING V3 MODEL
# ============================================================

sys.path.insert(
    0,
    str(V3_ROOT)
)

from model.v3_model import DeepShieldV3


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_FEATURE_SHAPE = (
    128,
    251,
)

NUM_CLASSES = 3

CLASS_NAMES = {
    0: "Genuine",
    1: "TTS",
    2: "Voice_Conversion",
}

EXPECTED_TEST_SAMPLES = 1158
EXPECTED_CLASS_COUNT = 386


# ============================================================
# ORIGINAL V3 TEST DATASET
# ============================================================

class V3OriginalTestDataset(Dataset):
    """
    Loads the ORIGINAL V3 test manifest and ORIGINAL V3
    precomputed test features.

    This follows the same feature-path logic as the existing
    DeepShieldV3Dataset:

        feature_path =
            V3/features/test/{audio_id}.npy
    """

    def __init__(self):

        self.manifest_path = (
            V3_TEST_MANIFEST
        )

        self.feature_dir = (
            V3_TEST_FEATURE_DIR
        )

        # ----------------------------------------------------
        # Validate paths
        # ----------------------------------------------------

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                "V3 test manifest not found:\n"
                f"{self.manifest_path}"
            )

        if not self.feature_dir.exists():

            raise FileNotFoundError(
                "V3 test feature directory not found:\n"
                f"{self.feature_dir}"
            )

        # ----------------------------------------------------
        # Load manifest
        # ----------------------------------------------------

        self.records = []

        with self.manifest_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(
                file
            )

            required_columns = {
                "audio_id",
                "audio_path",
                "speaker_id",
                "attack_id",
                "label",
                "label_name",
                "label_id",
                "split",
            }

            missing = (
                required_columns
                - set(reader.fieldnames or [])
            )

            if missing:

                raise RuntimeError(
                    "V3 test manifest is missing "
                    f"required columns: {sorted(missing)}"
                )

            for row in reader:

                self.records.append(
                    row
                )

        if not self.records:

            raise RuntimeError(
                "V3 test manifest is empty."
            )

        # ----------------------------------------------------
        # Validate records
        # ----------------------------------------------------

        self._validate_records()

    # ========================================================
    # Validate records
    # ========================================================

    def _validate_records(self):

        seen_audio_ids = set()

        class_counts = Counter()

        for index, record in enumerate(
            self.records
        ):

            audio_id = (
                record["audio_id"]
            )

            # ------------------------------------------------
            # Audio ID uniqueness
            # ------------------------------------------------

            if audio_id in seen_audio_ids:

                raise RuntimeError(
                    "Duplicate audio_id detected "
                    f"at index {index}: {audio_id}"
                )

            seen_audio_ids.add(
                audio_id
            )

            # ------------------------------------------------
            # Split validation
            # ------------------------------------------------

            if record["split"] != "test":

                raise RuntimeError(
                    f"Split mismatch at index {index}: "
                    f"expected 'test', "
                    f"found '{record['split']}'"
                )

            # ------------------------------------------------
            # Label validation
            # ------------------------------------------------

            try:

                label_id = int(
                    record["label_id"]
                )

            except ValueError:

                raise RuntimeError(
                    f"Invalid label_id at index "
                    f"{index}: "
                    f"{record['label_id']}"
                )

            if label_id not in CLASS_NAMES:

                raise RuntimeError(
                    f"Invalid label_id at index "
                    f"{index}: {label_id}"
                )

            expected_name = (
                CLASS_NAMES[label_id]
            )

            if (
                record["label_name"]
                != expected_name
            ):

                raise RuntimeError(
                    f"Label mismatch for "
                    f"{audio_id}: "
                    f"label_id={label_id}, "
                    f"label_name={record['label_name']}"
                )

            class_counts[
                label_id
            ] += 1

            # ------------------------------------------------
            # Feature validation
            # ------------------------------------------------

            feature_path = (
                self.feature_dir
                / f"{audio_id}.npy"
            )

            if not feature_path.exists():

                raise FileNotFoundError(
                    "V3 test feature missing:\n"
                    f"{feature_path}"
                )

        # ----------------------------------------------------
        # Dataset size
        # ----------------------------------------------------

        if len(self.records) != EXPECTED_TEST_SAMPLES:

            raise RuntimeError(
                "Unexpected V3 test size.\n"
                f"Expected: {EXPECTED_TEST_SAMPLES}\n"
                f"Found: {len(self.records)}"
            )

        # ----------------------------------------------------
        # Class distribution
        # ----------------------------------------------------

        for label_id in range(
            NUM_CLASSES
        ):

            if (
                class_counts[label_id]
                != EXPECTED_CLASS_COUNT
            ):

                raise RuntimeError(
                    f"Unexpected count for "
                    f"{CLASS_NAMES[label_id]}.\n"
                    f"Expected: {EXPECTED_CLASS_COUNT}\n"
                    f"Found: {class_counts[label_id]}"
                )

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(
            self.records
        )

    # ========================================================
    # Load feature
    # ========================================================

    def _load_feature(
        self,
        audio_id,
    ):

        feature_path = (
            self.feature_dir
            / f"{audio_id}.npy"
        )

        feature = np.load(
            feature_path,
            allow_pickle=False,
        )

        # ----------------------------------------------------
        # dtype
        # ----------------------------------------------------

        if feature.dtype != np.float32:

            raise RuntimeError(
                f"Unexpected dtype for "
                f"{audio_id}: "
                f"{feature.dtype}. "
                f"Expected float32."
            )

        # ----------------------------------------------------
        # shape
        # ----------------------------------------------------

        if (
            feature.shape
            != EXPECTED_FEATURE_SHAPE
        ):

            raise RuntimeError(
                f"Unexpected feature shape "
                f"for {audio_id}: "
                f"{feature.shape}. "
                f"Expected "
                f"{EXPECTED_FEATURE_SHAPE}."
            )

        # ----------------------------------------------------
        # numerical validity
        # ----------------------------------------------------

        if not np.isfinite(
            feature
        ).all():

            raise RuntimeError(
                f"Feature contains NaN or "
                f"infinite values: {audio_id}"
            )

        return feature

    # ========================================================
    # Get item
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        record = (
            self.records[index]
        )

        audio_id = (
            record["audio_id"]
        )

        label_id = int(
            record["label_id"]
        )

        feature = self._load_feature(
            audio_id
        )

        feature_tensor = torch.from_numpy(
            feature
        ).unsqueeze(
            0
        )

        label_tensor = torch.tensor(
            label_id,
            dtype=torch.long,
        )

        return (
            feature_tensor,
            label_tensor,
        )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):

    print()
    print("=" * 70)
    print("Loading V3.1 BEST checkpoint")
    print("=" * 70)

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "V3.1 best checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "best_metric" in checkpoint:

        print(
            "Checkpoint best balanced "
            f"accuracy: "
            f"{checkpoint['best_metric'] * 100:.2f}%"
        )

    print(
        "Checkpoint loading: PASS"
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DeepShield V3.1 - FINAL TEST EVALUATION")
    print("=" * 70)

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    # ========================================================
    # Paths
    # ========================================================

    print()
    print("Paths:")

    print(
        "  Original V3 test manifest:"
    )

    print(
        f"    {V3_TEST_MANIFEST}"
    )

    print(
        "  Original V3 test features:"
    )

    print(
        f"    {V3_TEST_FEATURE_DIR}"
    )

    print(
        "  V3.1 best checkpoint:"
    )

    print(
        f"    {V31_BEST_CHECKPOINT}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print("=" * 70)
    print("Loading ORIGINAL V3 test dataset")
    print("=" * 70)

    dataset = (
        V3OriginalTestDataset()
    )

    print()
    print(
        f"Test samples: "
        f"{len(dataset)}"
    )

    counts = Counter(
        int(record["label_id"])
        for record in dataset.records
    )

    print()
    print(
        "Test class distribution:"
    )

    for label_id in range(
        NUM_CLASSES
    ):

        print(
            f"  {label_id} "
            f"{CLASS_NAMES[label_id]:18s}: "
            f"{counts[label_id]}"
        )

    print()
    print(
        "Test dataset validation: PASS"
    )

    # ========================================================
    # DataLoader
    # ========================================================

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print()
    print(
        f"Test batches: "
        f"{len(loader)}"
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print("=" * 70)
    print("Creating DeepShieldV3")
    print("=" * 70)

    model = DeepShieldV3(
        num_classes=3
    )

    model = model.to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # ========================================================
    # Load BEST checkpoint
    # ========================================================

    model = load_checkpoint(
        model,
        V31_BEST_CHECKPOINT,
        device,
    )

    model.eval()

    # ========================================================
    # Evaluation
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Evaluating V3.1 BEST checkpoint "
        "on ORIGINAL V3 TEST SET"
    )
    print("=" * 70)

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    criterion = (
        torch.nn.CrossEntropyLoss()
    )

    with torch.no_grad():

        for (
            features,
            labels_batch,
        ) in loader:

            features = features.to(
                device,
                non_blocking=True,
            )

            labels_batch = (
                labels_batch.to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    device.type == "cuda"
                ),
            ):

                outputs = model(
                    features
                )

                loss = criterion(
                    outputs,
                    labels_batch,
                )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1,
                )
            )

            batch_size = (
                labels_batch.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

            all_labels.extend(
                labels_batch
                .cpu()
                .numpy()
                .tolist()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

    # ========================================================
    # Metrics
    # ========================================================

    test_loss = (
        total_loss
        / total_samples
    )

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

    macro_precision = (
        precision_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = (
        recall_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    macro_f1 = (
        f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = (
        f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[
            0,
            1,
            2,
        ],
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 70)
    print("V3.1 FINAL TEST RESULTS")
    print("=" * 70)

    print()
    print(
        f"Test Loss          : "
        f"{test_loss:.6f}"
    )

    print(
        f"Accuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy  : "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision    : "
        f"{macro_precision * 100:.2f}%"
    )

    print(
        f"Macro Recall       : "
        f"{macro_recall * 100:.2f}%"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1 * 100:.2f}%"
    )

    # ========================================================
    # Per-class metrics
    # ========================================================

    print()
    print("=" * 70)
    print("Per-Class Metrics")
    print("=" * 70)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=[
            0,
            1,
            2,
        ],
        target_names=[
            "Genuine",
            "TTS",
            "Voice_Conversion",
        ],
        digits=4,
        zero_division=0,
    )

    print(
        report
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    print("=" * 70)
    print("Confusion Matrix")
    print("=" * 70)

    print()
    print(
        "Rows = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print()

    print(
        f"{'':25}"
        f"{'Genuine':>12}"
        f"{'TTS':>12}"
        f"{'Voice_Conversion':>20}"
    )

    for i, class_name in enumerate(
        [
            "Genuine",
            "TTS",
            "Voice_Conversion",
        ]
    ):

        print(
            f"{class_name:<25}"
            f"{cm[i, 0]:>12}"
            f"{cm[i, 1]:>12}"
            f"{cm[i, 2]:>20}"
        )

    # ========================================================
    # Integrity
    # ========================================================

    print()
    print("=" * 70)
    print("TEST EVALUATION INTEGRITY")
    print("=" * 70)

    print(
        "✓ Original V3 test manifest used"
    )

    print(
        "✓ Original V3 test features used"
    )

    print(
        "✓ Test set was not used during "
        "V3.1 training"
    )

    print(
        "✓ V3.1 BEST checkpoint used"
    )

    print(
        "✓ No training performed"
    )

    print(
        "✓ No V3 checkpoint modified"
    )

    print(
        "✓ No V3.1 checkpoint modified"
    )

    print(
        "✓ Existing detection system "
        "not modified"
    )

    print()
    print("=" * 70)
    print(
        "V3.1 TEST EVALUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()