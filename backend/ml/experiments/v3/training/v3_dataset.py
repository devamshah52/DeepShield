"""
DeepShield V3 - PyTorch Dataset Loader

Loads precomputed V3 Mel-spectrogram features and labels.

V3 classes:
    0 = Genuine
    1 = TTS
    2 = Voice Conversion

Feature shape:
    (128, 251)

Tensor shape returned by __getitem__:
    (1, 128, 251)

This file is completely independent of the existing V1/V2
audio_dataset.py.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

BACKEND_DIR = (
    PROJECT_ROOT
    / "backend"
)

V3_DIR = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3"
)

MANIFEST_DIR = (
    V3_DIR
    / "dataset"
    / "manifests"
)

FEATURE_DIR = (
    V3_DIR
    / "features"
)


# ============================================================
# Configuration
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

SPLITS = (
    "train",
    "dev",
    "test",
)

MANIFEST_NAMES = {
    "train": "v3_train.csv",
    "dev": "v3_dev.csv",
    "test": "v3_test.csv",
}


# ============================================================
# V3 Dataset
# ============================================================

class DeepShieldV3Dataset(Dataset):
    """
    PyTorch Dataset for the DeepShield V3 feature cache.

    Each sample contains:
        - audio_id
        - speaker_id
        - attack_id
        - label_name
        - label_id
        - path to cached Mel feature
    """

    def __init__(
        self,
        split: str,
    ):
        if split not in SPLITS:

            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of: {SPLITS}"
            )

        self.split = split

        self.manifest_path = (
            MANIFEST_DIR
            / MANIFEST_NAMES[split]
        )

        self.feature_dir = (
            FEATURE_DIR
            / split
        )

        # ----------------------------------------------------
        # Validate paths
        # ----------------------------------------------------

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                "V3 manifest not found:\n"
                f"{self.manifest_path}"
            )

        if not self.feature_dir.exists():

            raise FileNotFoundError(
                "V3 feature directory not found:\n"
                f"{self.feature_dir}"
            )

        # ----------------------------------------------------
        # Load manifest
        # ----------------------------------------------------

        self.records = self._read_manifest()

        if not self.records:

            raise RuntimeError(
                f"V3 {split} manifest is empty."
            )

        # ----------------------------------------------------
        # Validate manifest and cache
        # ----------------------------------------------------

        self._validate_records()

        print()
        print(
            f"Loaded V3 {split.upper()} dataset:"
        )

        print(
            f"  Samples : {len(self.records)}"
        )

        counts = Counter(
            record["label_id"]
            for record in self.records
        )

        for label_id in range(NUM_CLASSES):

            print(
                f"  {label_id} "
                f"{CLASS_NAMES[label_id]:18s}: "
                f"{counts[label_id]}"
            )

    # ========================================================
    # Read manifest
    # ========================================================

    def _read_manifest(self):

        records = []

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
                    "Manifest is missing required "
                    f"columns: {sorted(missing)}"
                )

            for row in reader:

                records.append(row)

        return records

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
                    f"in {self.split}: {audio_id}"
                )

            seen_audio_ids.add(
                audio_id
            )

            # ------------------------------------------------
            # Split validation
            # ------------------------------------------------

            if record["split"] != self.split:

                raise RuntimeError(
                    f"Manifest split mismatch at "
                    f"index {index}: "
                    f"expected '{self.split}', "
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
                    f"Label name mismatch for "
                    f"{audio_id}: "
                    f"label_id={label_id}, "
                    f"label_name="
                    f"{record['label_name']}"
                )

            class_counts[
                label_id
            ] += 1

            # ------------------------------------------------
            # Feature file validation
            # ------------------------------------------------

            feature_path = (
                self.feature_dir
                / f"{audio_id}.npy"
            )

            if not feature_path.exists():

                raise FileNotFoundError(
                    "Feature file missing:\n"
                    f"{feature_path}"
                )

        # ----------------------------------------------------
        # Verify all three classes exist
        # ----------------------------------------------------

        for label_id in range(NUM_CLASSES):

            if class_counts[label_id] == 0:

                raise RuntimeError(
                    f"Class {label_id} "
                    f"({CLASS_NAMES[label_id]}) "
                    f"contains zero samples."
                )

    # ========================================================
    # Dataset length
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
        audio_id: str,
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
        # Validate dtype
        # ----------------------------------------------------

        if feature.dtype != np.float32:

            raise RuntimeError(
                f"Unexpected dtype for "
                f"{audio_id}: "
                f"{feature.dtype}. "
                f"Expected float32."
            )

        # ----------------------------------------------------
        # Validate shape
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
        # Validate numerical values
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
        index: int,
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

        # ----------------------------------------------------
        # Load Mel feature
        # ----------------------------------------------------

        feature = self._load_feature(
            audio_id
        )

        # ----------------------------------------------------
        # Convert:
        #
        # NumPy:
        #     (128, 251)
        #
        # PyTorch:
        #     (1, 128, 251)
        # ----------------------------------------------------

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
# Dataset information helper
# ============================================================

def print_dataset_info(
    split: str,
):

    dataset = DeepShieldV3Dataset(
        split
    )

    print()
    print("=" * 70)
    print(
        f"DeepShield V3 {split.upper()} Dataset"
    )
    print("=" * 70)

    print(
        f"Samples: {len(dataset)}"
    )

    counts = Counter(
        int(record["label_id"])
        for record in dataset.records
    )

    print()
    print(
        "Class distribution:"
    )

    for label_id in range(
        NUM_CLASSES
    ):

        print(
            f"  {label_id} "
            f"{CLASS_NAMES[label_id]:18s}: "
            f"{counts[label_id]}"
        )

    # --------------------------------------------------------
    # Inspect first sample
    # --------------------------------------------------------

    features, label = dataset[0]

    print()
    print(
        "First sample:"
    )

    print(
        f"  Tensor shape : "
        f"{tuple(features.shape)}"
    )

    print(
        f"  Tensor dtype : "
        f"{features.dtype}"
    )

    print(
        f"  Label        : "
        f"{label.item()}"
    )

    print(
        f"  Class        : "
        f"{CLASS_NAMES[label.item()]}"
    )

    print(
        f"  Min          : "
        f"{features.min().item():.6f}"
    )

    print(
        f"  Max          : "
        f"{features.max().item():.6f}"
    )

    print(
        f"  Mean         : "
        f"{features.mean().item():.6f}"
    )

    print(
        f"  Std          : "
        f"{features.std().item():.6f}"
    )

    print("=" * 70)


# ============================================================
# Main test
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DeepShield V3 - Dataset Loader Test")
    print("=" * 70)

    for split in SPLITS:

        print_dataset_info(
            split
        )

    print()
    print(
        "V3 DATASET LOADER TEST: PASS"
    )


if __name__ == "__main__":

    main()