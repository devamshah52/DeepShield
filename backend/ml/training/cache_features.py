import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from ml.training.feature_extractor import (
    extract_mel_spectrogram,
)


# ============================================================
# Paths
# ============================================================

DATASET_DIR = (
    BACKEND_DIR
    / "ml"
    / "dataset"
)


FEATURE_CACHE_DIR = (
    BACKEND_DIR
    / "ml"
    / "features"
    / "cache"
)


FEATURE_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Manifests
# ============================================================

MANIFESTS = {

    "train": (
        DATASET_DIR
        / "train_manifest.csv"
    ),

    "dev": (
        DATASET_DIR
        / "dev_manifest.csv"
    ),

    "eval": (
        DATASET_DIR
        / "eval_manifest.csv"
    ),
}


# ============================================================
# Expected feature shape
# ============================================================

EXPECTED_FEATURE_SHAPE = (
    128,
    251,
)


# ============================================================
# Cache one manifest
# ============================================================

def cache_manifest(
    manifest_path,
    split_name,
):

    print()
    print("=" * 70)
    print(
        f"Caching {split_name.upper()} features"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Verify manifest
    # --------------------------------------------------------

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{manifest_path}"
        )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        manifest_path
    )

    print(
        "Samples:",
        len(dataframe)
    )

    # --------------------------------------------------------
    # Split-specific cache directory
    # --------------------------------------------------------

    split_cache_dir = (
        FEATURE_CACHE_DIR
        / split_name
    )

    split_cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    created = 0
    skipped = 0
    failed = 0

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    for _, row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
        desc=f"{split_name} features",
    ):

        audio_path = Path(
            row["audio_path"]
        )

        audio_id = str(
            row["audio_id"]
        )

        cache_path = (
            split_cache_dir
            / f"{audio_id}.npy"
        )

        # ----------------------------------------------------
        # Skip existing feature
        # ----------------------------------------------------

        if cache_path.exists():

            skipped += 1

            continue

        # ----------------------------------------------------
        # Extract Mel spectrogram
        # ----------------------------------------------------

        try:

            mel = (
                extract_mel_spectrogram(
                    audio_path
                )
            )

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if mel.shape != (
                EXPECTED_FEATURE_SHAPE
            ):

                raise ValueError(
                    "Unexpected feature shape: "
                    f"{mel.shape}"
                )

            # ------------------------------------------------
            # Save feature
            # ------------------------------------------------

            np.save(
                cache_path,
                mel,
            )

            created += 1

        except Exception as error:

            failed += 1

            print()
            print(
                "ERROR:"
            )

            print(
                "Audio:",
                audio_path,
            )

            print(
                "Reason:",
                error,
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        "Created:",
        created,
    )

    print(
        "Skipped:",
        skipped,
    )

    print(
        "Failed:",
        failed,
    )

    print(
        "Cache:",
        split_cache_dir,
    )

    print("-" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "DeepShield Feature Cache Generator"
    )
    print("=" * 70)

    print()
    print(
        "Cache directory:"
    )

    print(
        FEATURE_CACHE_DIR
    )

    print()

    # ========================================================
    # TRAIN
    # ========================================================

    cache_manifest(
        MANIFESTS["train"],
        "train",
    )

    # ========================================================
    # DEVELOPMENT
    # ========================================================

    cache_manifest(
        MANIFESTS["dev"],
        "dev",
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    cache_manifest(
        MANIFESTS["eval"],
        "eval",
    )

    # ========================================================
    # Completed
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Feature caching completed."
    )
    print("=" * 70)