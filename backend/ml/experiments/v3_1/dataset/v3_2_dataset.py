"""
DeepShield V3.2 - Dataset Loader

Training:
    V3.1 training data + 1,800 gTTS TTS samples.

Development:
    Original V3 development manifest/features only.

Classes:
    0 = Genuine
    1 = TTS
    2 = Voice_Conversion

Important:
    - Existing V3 dataset files are NOT modified.
    - Existing V3 features are NOT modified.
    - Existing V3.1 manifest is NOT modified.
    - gTTS features are stored separately.
    - Test data is NOT loaded by this training dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

V3_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "experiments"
    / "v3"
)

V3_1_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "experiments"
    / "v3_1"
)

V3_2_DATASET_ROOT = (
    PROJECT_ROOT
    / "v3_2_dataset"
)

V3_FEATURE_ROOT = (
    V3_ROOT
    / "features"
)

V3_MANIFEST_ROOT = (
    V3_ROOT
    / "dataset"
    / "manifests"
)

V3_2_MANIFEST = (
    V3_2_DATASET_ROOT
    / "v3_2_train_manifest.csv"
)

V3_1_COMMON_VOICE_FEATURE_ROOT = (
    V3_1_ROOT
    / "features"
    / "train"
    / "common_voice"
)


# ============================================================
# Feature configuration
# ============================================================

FEATURE_HEIGHT = 128
FEATURE_WIDTH = 251

EXPECTED_FEATURE_SHAPE = (
    FEATURE_HEIGHT,
    FEATURE_WIDTH,
)

EXPECTED_DTYPE = np.float32


# ============================================================
# Classes
# ============================================================

CLASS_NAMES = (
    "Genuine",
    "TTS",
    "Voice_Conversion",
)

NUM_CLASSES = 3


# ============================================================
# Expected dataset counts
# ============================================================

EXPECTED_TRAIN_SAMPLES = 9024

EXPECTED_TRAIN_SOURCE_COUNTS = {
    "asvspoof_v3": 5424,
    "common_voice": 1800,
    "gtts": 1800,
}

EXPECTED_TRAIN_LABEL_COUNTS = {
    0: 3608,
    1: 3608,
    2: 1808,
}

EXPECTED_DEV_SAMPLES = 1158

EXPECTED_DEV_LABEL_COUNTS = {
    0: 386,
    1: 386,
    2: 386,
}


# ============================================================
# Manifest helper
# ============================================================

def load_manifest(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
    "audio_path",
    "label",
    "label_name",
}

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"Manifest {path.name} is missing "
            f"columns: {sorted(missing)}"
        )

    return df


# ============================================================
# Feature path resolver
# ============================================================

def get_feature_path(
    row,
    split: str,
) -> Path:
    """
    Resolve the feature corresponding to one manifest row.

    Original V3 / ASVspoof:
        v3/features/{split}/{audio_stem}.npy

    Common Voice:
        v3_1/features/train/common_voice/{audio_stem}.npy

    gTTS:
        v3_2_dataset/gtts_features/{feature_file}.npy

    For gTTS, audio_path in the V3.2 manifest already points
    directly to the feature file, so it is returned directly.
    """

    source = str(
        row["source"]
    )

    audio_path = Path(
        str(row["audio_path"])
    )

    audio_stem = (
        audio_path.stem
    )

    # --------------------------------------------------------
    # Original V3 / ASVspoof
    # --------------------------------------------------------

    if source == "asvspoof_v3":

        return (
            V3_FEATURE_ROOT
            / split
            / f"{audio_stem}.npy"
        )

    # --------------------------------------------------------
    # Common Voice
    # --------------------------------------------------------

    if source == "common_voice":

        return (
            V3_1_COMMON_VOICE_FEATURE_ROOT
            / f"{audio_stem}.npy"
        )

    # --------------------------------------------------------
    # gTTS
    # --------------------------------------------------------

    if source == "gtts":

        return audio_path

    # --------------------------------------------------------
    # Unknown source
    # --------------------------------------------------------

    raise RuntimeError(
        f"Unknown dataset source: {source}"
    )


# ============================================================
# Label validation
# ============================================================

def validate_label(
    row,
) -> int:

    raw_label = row["label"]

    label_name = str(
        row["label_name"]
    ).strip()

    label_name_to_id = {
        "Genuine": 0,
        "TTS": 1,
        "Voice_Conversion": 2,
    }

    if label_name not in label_name_to_id:

        raise ValueError(
            "Unknown label_name: "
            f"{label_name}"
        )

    expected_label = (
        label_name_to_id[label_name]
    )

    # --------------------------------------------------------
    # Numeric label validation
    # --------------------------------------------------------

    try:

        numeric_label = int(
            raw_label
        )

    except (
        ValueError,
        TypeError,
    ):

        numeric_label = expected_label

    if numeric_label != expected_label:

        raise ValueError(
            "Label mismatch: "
            f"label={raw_label}, "
            f"label_name={label_name}, "
            f"expected_numeric_label="
            f"{expected_label}"
        )

    return expected_label


# ============================================================
# Feature validation
# ============================================================

def load_feature(
    feature_path: Path,
) -> np.ndarray:

    if not feature_path.exists():

        raise FileNotFoundError(
            f"Feature not found:\n{feature_path}"
        )

    feature = np.load(
        feature_path,
        allow_pickle=False,
    )

    if feature.shape != EXPECTED_FEATURE_SHAPE:

        raise RuntimeError(
            "Unexpected feature shape: "
            f"{feature.shape}. "
            f"Expected {EXPECTED_FEATURE_SHAPE}.\n"
            f"File: {feature_path}"
        )

    if feature.dtype != EXPECTED_DTYPE:

        raise RuntimeError(
            "Unexpected feature dtype: "
            f"{feature.dtype}. "
            f"Expected {EXPECTED_DTYPE}.\n"
            f"File: {feature_path}"
        )

    if not np.isfinite(
        feature
    ).all():

        raise RuntimeError(
            "Feature contains non-finite values:\n"
            f"{feature_path}"
        )

    return feature


# ============================================================
# Dataset
# ============================================================

class DeepShieldV32Dataset(
    Dataset
):

    def __init__(
        self,
        split: str,
    ):

        if split not in (
            "train",
            "dev",
        ):

            raise ValueError(
                "V3.2 training dataset supports "
                "only 'train' and 'dev'."
            )

        self.split = split

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        if split == "train":

            self.manifest = (
                load_manifest(
                    V3_2_MANIFEST
                )
            )

            # ------------------------------------------------
            # Validate source counts
            # ------------------------------------------------

            source_counts = (
                self.manifest[
                    "source"
                ]
                .value_counts()
                .to_dict()
            )

            expected_sources = (
                EXPECTED_TRAIN_SOURCE_COUNTS
            )

            if source_counts != expected_sources:

                raise RuntimeError(
                    "Unexpected V3.2 training "
                    "source distribution:\n"
                    f"Expected: {expected_sources}\n"
                    f"Found:    {source_counts}"
                )

            # ------------------------------------------------
            # Validate labels
            # ------------------------------------------------

            label_counts = (
                self.manifest[
                    "label"
                ]
                .value_counts()
                .to_dict()
            )

            expected_labels = (
                EXPECTED_TRAIN_LABEL_COUNTS
            )

            if label_counts != expected_labels:

                raise RuntimeError(
                    "Unexpected V3.2 training "
                    "label distribution:\n"
                    f"Expected: {expected_labels}\n"
                    f"Found:    {label_counts}"
                )

        # ----------------------------------------------------
        # Development
        # ----------------------------------------------------

        else:

            v3_dev = (
                V3_MANIFEST_ROOT
                / "v3_dev.csv"
            )

            self.manifest = (
                load_manifest(
                    v3_dev
                )
            )

            # Original V3 dev is ASVspoof.
            self.manifest = (
                self.manifest.copy()
            )

            self.manifest[
                "source"
            ] = "asvspoof_v3"

            # ------------------------------------------------
            # Validate development labels
            # ------------------------------------------------

            dev_labels = []

            for _, row in (
                self.manifest.iterrows()
            ):

                dev_labels.append(
                    validate_label(
                        row
                    )
                )

            dev_label_counts = (
                pd.Series(
                    dev_labels
                )
                .value_counts()
                .to_dict()
            )

            expected_dev_labels = (
                EXPECTED_DEV_LABEL_COUNTS
            )

            if (
                dev_label_counts
                != expected_dev_labels
            ):

                raise RuntimeError(
                    "Unexpected V3 development "
                    "label distribution:\n"
                    f"Expected: "
                    f"{expected_dev_labels}\n"
                    f"Found: "
                    f"{dev_label_counts}"
                )

        # ----------------------------------------------------
        # Validate every row's label
        # ----------------------------------------------------

        self.labels = []

        for _, row in (
            self.manifest.iterrows()
        ):

            self.labels.append(
                validate_label(
                    row
                )
            )

        self.labels = np.asarray(
            self.labels,
            dtype=np.int64,
        )

        # ----------------------------------------------------
        # Validate feature availability
        # ----------------------------------------------------

        print()
        print(
            f"Validating V3.2 {split} dataset..."
        )

        missing_features = []

        for _, row in (
            self.manifest.iterrows()
        ):

            feature_path = (
                get_feature_path(
                    row,
                    split,
                )
            )

            if not feature_path.exists():

                missing_features.append(
                    str(feature_path)
                )

        if missing_features:

            preview = (
                missing_features[:10]
            )

            raise RuntimeError(
                f"{len(missing_features):,} "
                f"feature files are missing.\n"
                "First missing files:\n"
                + "\n".join(preview)
            )

        print(
            f"  Samples : "
            f"{len(self.manifest):,}"
        )

        print(
            "  Feature validation: PASS"
        )

        # ----------------------------------------------------
        # Class distribution
        # ----------------------------------------------------

        print(
            "  Class distribution:"
        )

        for label_id, class_name in enumerate(
            CLASS_NAMES
        ):

            count = int(
                np.sum(
                    self.labels
                    == label_id
                )
            )

            print(
                f"    {label_id} "
                f"{class_name:18s}: "
                f"{count:,}"
            )

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(
            self.manifest
        )

    # ========================================================
    # Get item
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ):

        row = self.manifest.iloc[
            index
        ]

        feature_path = (
            get_feature_path(
                row,
                self.split,
            )
        )

        feature = load_feature(
            feature_path
        )

        # ----------------------------------------------------
        # Add channel dimension
        #
        # (128, 251)
        #      ↓
        # (1, 128, 251)
        # ----------------------------------------------------

        feature = np.expand_dims(
            feature,
            axis=0,
        )

        feature_tensor = (
            torch.from_numpy(
                feature
            ).float()
        )

        label_tensor = torch.tensor(
            int(
                self.labels[index]
            ),
            dtype=torch.long,
        )

        return (
            feature_tensor,
            label_tensor,
        )


# ============================================================
# Dataset summary
# ============================================================

def print_dataset_summary(
    dataset: DeepShieldV32Dataset,
):

    print()
    print("=" * 70)

    print(
        f"V3.2 "
        f"{dataset.split.upper()} "
        f"DATASET SUMMARY"
    )

    print("=" * 70)

    print(
        f"Samples: "
        f"{len(dataset):,}"
    )

    print()

    print(
        "Classes:"
    )

    for label_id, class_name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            np.sum(
                dataset.labels
                == label_id
            )
        )

        print(
            f"  {label_id} "
            f"{class_name:18s}: "
            f"{count:,}"
        )


# ============================================================
# Standalone validation
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "DeepShield V3.2 - Dataset Validation"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

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
    # Print summaries
    # --------------------------------------------------------

    print_dataset_summary(
        train_dataset
    )

    print_dataset_summary(
        dev_dataset
    )

    # --------------------------------------------------------
    # Test one sample from each dataset
    # --------------------------------------------------------

    print()
    print(
        "Testing dataset samples..."
    )

    train_features, train_label = (
        train_dataset[0]
    )

    dev_features, dev_label = (
        dev_dataset[0]
    )

    print()
    print(
        "Train sample:"
    )

    print(
        f"  Feature shape: "
        f"{tuple(train_features.shape)}"
    )

    print(
        f"  Feature dtype: "
        f"{train_features.dtype}"
    )

    print(
        f"  Label: "
        f"{train_label.item()} "
        f"({CLASS_NAMES[train_label.item()]})"
    )

    print()
    print(
        "Dev sample:"
    )

    print(
        f"  Feature shape: "
        f"{tuple(dev_features.shape)}"
    )

    print(
        f"  Feature dtype: "
        f"{dev_features.dtype}"
    )

    print(
        f"  Label: "
        f"{dev_label.item()} "
        f"({CLASS_NAMES[dev_label.item()]})"
    )

    # --------------------------------------------------------
    # Explicitly test a gTTS sample
    # --------------------------------------------------------

    print()
    print(
        "Testing gTTS sample..."
    )

    gtts_indices = np.where(
        train_dataset.manifest[
            "source"
        ].values
        == "gtts"
    )[0]

    if len(gtts_indices) == 0:

        raise RuntimeError(
            "No gTTS samples found "
            "in the V3.2 training dataset."
        )

    gtts_index = int(
        gtts_indices[0]
    )

    gtts_features, gtts_label = (
        train_dataset[
            gtts_index
        ]
    )

    print(
        f"  Index: "
        f"{gtts_index}"
    )

    print(
        f"  Feature shape: "
        f"{tuple(gtts_features.shape)}"
    )

    print(
        f"  Feature dtype: "
        f"{gtts_features.dtype}"
    )

    print(
        f"  Label: "
        f"{gtts_label.item()} "
        f"({CLASS_NAMES[gtts_label.item()]})"
    )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    if len(train_dataset) != (
        EXPECTED_TRAIN_SAMPLES
    ):

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_TRAIN_SAMPLES:,} "
            "training samples."
        )

    if len(dev_dataset) != (
        EXPECTED_DEV_SAMPLES
    ):

        raise RuntimeError(
            "Expected "
            f"{EXPECTED_DEV_SAMPLES:,} "
            "development samples."
        )

    if tuple(train_features.shape) != (
        1,
        128,
        251,
    ):

        raise RuntimeError(
            "Unexpected gTTS training "
            "feature tensor shape."
        )

    if tuple(dev_features.shape) != (
        1,
        128,
        251,
    ):

        raise RuntimeError(
            "Unexpected development "
            "feature tensor shape."
        )

    if gtts_label.item() != 1:

        raise RuntimeError(
            "gTTS sample does not have "
            "TTS label 1."
        )

    print()
    print("=" * 70)

    print(
        "V3.2 DATASET VALIDATION: PASS"
    )

    print("=" * 70)

    print()
    print(
        "Training: 9,024 samples"
    )

    print(
        "Development: 1,158 samples"
    )

    print(
        "Feature tensor: (1, 128, 251)"
    )

    print(
        "gTTS samples: 1,800"
    )

    print(
        "Test set: NOT loaded"
    )


if __name__ == "__main__":

    main()