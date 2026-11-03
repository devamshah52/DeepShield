"""
DeepShield V3.1 - Common Voice Feature Extraction

Extracts Mel-spectrogram features from the selected 1,800
Common Voice Genuine recordings.

This uses the SAME audio/Mel preprocessing as DeepShield V3:

    - 16 kHz
    - Mono
    - Exactly 4 seconds
    - n_fft = 1024
    - hop_length = 256
    - n_mels = 128
    - power = 2.0
    - librosa.power_to_db(ref=np.max)
    - Per-file mean/std normalization
    - float32
    - Expected shape: (128, 251)

IMPORTANT:
    - Existing V3 files are NOT modified.
    - Existing V3 features are NOT modified.
    - No model is trained.
    - FastAPI / React should remain OFF.

Input:
    C:\\DeepShield\\v3_1_dataset_inspection\\v3_1_train_manifest.csv

Common Voice audio:
    C:\\DeepShield\\v3_1_dataset\\clips

Output:
    C:\\DeepShield\\backend\\ml\\experiments\\v3_1\\features\\train\\common_voice
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

MANIFEST_PATH = (
    PROJECT_ROOT
    / "v3_1_dataset_inspection"
    / "v3_1_train_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "experiments"
    / "v3_1"
    / "features"
    / "train"
    / "common_voice"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "experiments"
    / "v3_1"
    / "dataset"
    / "metadata"
)


# ============================================================
# Audio / Mel configuration
# ============================================================

TARGET_SAMPLE_RATE = 16000

TARGET_DURATION = 4

TARGET_SAMPLES = (
    TARGET_SAMPLE_RATE
    * TARGET_DURATION
)

N_FFT = 1024

HOP_LENGTH = 256

N_MELS = 128

EXPECTED_FEATURE_SHAPE = (
    128,
    251,
)


# ============================================================
# Utility
# ============================================================

def ensure_directories():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Load and standardize audio
# ============================================================

def load_audio(
    audio_path: Path,
):
    """
    Load one Common Voice audio file and standardize it to:

        - mono
        - 16 kHz
        - exactly 4 seconds
    """

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    audio, sample_rate = (
        librosa.load(
            str(audio_path),
            sr=TARGET_SAMPLE_RATE,
            mono=True,
        )
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:

        raise ValueError(
            "Decoded audio contains no samples."
        )

    # --------------------------------------------------------
    # Standardize to exactly 4 seconds
    # --------------------------------------------------------

    if len(audio) > TARGET_SAMPLES:

        audio = audio[
            :TARGET_SAMPLES
        ]

    elif len(audio) < TARGET_SAMPLES:

        padding = (
            TARGET_SAMPLES
            - len(audio)
        )

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant",
        )

    return (
        audio.astype(np.float32),
        TARGET_SAMPLE_RATE,
    )


# ============================================================
# Mel extraction
# ============================================================

def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
):
    """
    Convert standardized audio into the exact V3-style
    normalized Mel-spectrogram.

    Expected output:
        (128, 251)
    """

    mel = (
        librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
    )

    mel_db = (
        librosa.power_to_db(
            mel,
            ref=np.max,
        )
    )

    # --------------------------------------------------------
    # Per-file normalization
    # --------------------------------------------------------

    mean = np.mean(
        mel_db
    )

    std = np.std(
        mel_db
    )

    if std > 0:

        mel_db = (
            mel_db - mean
        ) / std

    else:

        mel_db = (
            mel_db - mean
        )

    mel_db = mel_db.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if mel_db.shape != EXPECTED_FEATURE_SHAPE:

        raise RuntimeError(
            "Unexpected Mel feature shape: "
            f"{mel_db.shape}. "
            f"Expected {EXPECTED_FEATURE_SHAPE}."
        )

    return mel_db


# ============================================================
# Feature filename
# ============================================================

def feature_filename(
    audio_path: str,
):
    """
    Use the Common Voice audio filename stem.

    Example:

        common_voice_en_18277778.mp3

    becomes:

        common_voice_en_18277778.npy
    """

    return (
        f"{Path(audio_path).stem}.npy"
    )


# ============================================================
# Main extraction
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "DeepShield V3.1 - Common Voice Feature Extraction"
    )
    print("=" * 75)

    print()
    print(
        f"Manifest:\n{MANIFEST_PATH}"
    )

    print()
    print(
        f"Output:\n{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Validate manifest
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    ensure_directories()

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    print()
    print("Loading V3.1 training manifest...")

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Total manifest rows: {len(df):,}"
    )

    required_columns = {
        "source",
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
            "Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Select Common Voice only
    # --------------------------------------------------------

    cv_df = df[
        df["source"]
        == "common_voice"
    ].copy()

    print(
        f"Common Voice rows: {len(cv_df):,}"
    )

    if len(cv_df) != 1800:

        raise RuntimeError(
            "Expected exactly 1,800 Common Voice "
            f"rows, found {len(cv_df):,}."
        )

    # --------------------------------------------------------
    # Validate Common Voice labels
    # --------------------------------------------------------

    if not (
        cv_df["label"]
        == 0
    ).all():

        raise RuntimeError(
            "Common Voice rows must all have label 0."
        )

    if not (
        cv_df["label_name"]
        == "Genuine"
    ).all():

        raise RuntimeError(
            "Common Voice rows must all have "
            "label_name='Genuine'."
        )

    # --------------------------------------------------------
    # Validate unique audio paths
    # --------------------------------------------------------

    unique_paths = (
        cv_df["audio_path"]
        .nunique()
    )

    print(
        f"Unique Common Voice paths: {unique_paths:,}"
    )

    if unique_paths != 1800:

        raise RuntimeError(
            "Common Voice audio paths are not unique."
        )

    # --------------------------------------------------------
    # Extraction counters
    # --------------------------------------------------------

    successful = 0

    skipped = 0

    failed = 0

    failures = []

    feature_shapes = {}

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("EXTRACTING COMMON VOICE FEATURES")
    print("=" * 75)

    for index, row in enumerate(
        cv_df.itertuples(index=False),
        start=1,
    ):

        audio_path_text = str(
            row.audio_path
        )

        audio_path = Path(
            audio_path_text
        )

        output_path = (
            OUTPUT_DIR
            / feature_filename(
                audio_path_text
            )
        )

        # ----------------------------------------------------
        # Skip existing valid feature
        # ----------------------------------------------------

        if output_path.exists():

            try:

                existing = np.load(
                    output_path,
                    allow_pickle=False,
                )

                if (
                    existing.shape
                    == EXPECTED_FEATURE_SHAPE
                    and existing.dtype
                    == np.float32
                ):

                    skipped += 1

                    shape_key = str(
                        existing.shape
                    )

                    feature_shapes[
                        shape_key
                    ] = (
                        feature_shapes.get(
                            shape_key,
                            0,
                        )
                        + 1
                    )

                    continue

            except Exception:
                pass

        try:

            # ------------------------------------------------
            # Load audio
            # ------------------------------------------------

            audio, sample_rate = (
                load_audio(
                    audio_path
                )
            )

            # ------------------------------------------------
            # Extract Mel feature
            # ------------------------------------------------

            feature = (
                extract_mel_spectrogram(
                    audio,
                    sample_rate,
                )
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            np.save(
                output_path,
                feature,
                allow_pickle=False,
            )

            successful += 1

            shape_key = str(
                feature.shape
            )

            feature_shapes[
                shape_key
            ] = (
                feature_shapes.get(
                    shape_key,
                    0,
                )
                + 1
            )

        except Exception as error:

            failed += 1

            failures.append(
                {
                    "audio_path": audio_path_text,
                    "error": str(error),
                }
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index == 1
            or index % 100 == 0
            or index == len(cv_df)
        ):

            print(
                f"[{index:4d}/{len(cv_df)}] "
                f"successful={successful} "
                f"skipped={skipped} "
                f"failed={failed}"
            )

    # ========================================================
    # Validate all features
    # ========================================================

    print()
    print("=" * 75)
    print("VALIDATING COMMON VOICE FEATURES")
    print("=" * 75)

    valid_features = 0

    invalid_features = []

    for row in cv_df.itertuples(
        index=False
    ):

        audio_path_text = str(
            row.audio_path
        )

        feature_path = (
            OUTPUT_DIR
            / feature_filename(
                audio_path_text
            )
        )

        if not feature_path.exists():

            invalid_features.append(
                {
                    "audio_path": audio_path_text,
                    "reason": "feature_missing",
                }
            )

            continue

        try:

            feature = np.load(
                feature_path,
                allow_pickle=False,
            )

            if (
                feature.shape
                != EXPECTED_FEATURE_SHAPE
            ):

                invalid_features.append(
                    {
                        "audio_path": audio_path_text,
                        "reason": (
                            "invalid_shape"
                        ),
                        "shape": str(
                            feature.shape
                        ),
                    }
                )

                continue

            if (
                feature.dtype
                != np.float32
            ):

                invalid_features.append(
                    {
                        "audio_path": audio_path_text,
                        "reason": (
                            "invalid_dtype"
                        ),
                        "dtype": str(
                            feature.dtype
                        ),
                    }
                )

                continue

            if not np.isfinite(
                feature
            ).all():

                invalid_features.append(
                    {
                        "audio_path": audio_path_text,
                        "reason": (
                            "non_finite_values"
                        ),
                    }
                )

                continue

            valid_features += 1

        except Exception as error:

            invalid_features.append(
                {
                    "audio_path": audio_path_text,
                    "reason": str(error),
                }
            )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "dataset": "Common Voice",
        "purpose": (
            "V3.1 Genuine class improvement"
        ),
        "manifest_samples": len(cv_df),
        "successful_new": successful,
        "skipped_existing": skipped,
        "failed_during_extraction": failed,
        "valid_features": valid_features,
        "invalid_features": len(
            invalid_features
        ),
        "feature_shape": list(
            EXPECTED_FEATURE_SHAPE
        ),
        "feature_dtype": "float32",
        "sample_rate": TARGET_SAMPLE_RATE,
        "duration_seconds": TARGET_DURATION,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "normalization": (
            "per-file mean/std after dB conversion"
        ),
        "feature_shapes": feature_shapes,
    }

    summary_path = (
        METADATA_DIR
        / "v3_1_common_voice_feature_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Failure report
    # --------------------------------------------------------

    if failures:

        failure_path = (
            METADATA_DIR
            / "v3_1_common_voice_feature_failures.json"
        )

        with failure_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                failures,
                file,
                indent=2,
            )

        print()
        print(
            f"Extraction failure report:\n"
            f"{failure_path}"
        )

    if invalid_features:

        invalid_path = (
            METADATA_DIR
            / "v3_1_common_voice_invalid_features.json"
        )

        with invalid_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                invalid_features,
                file,
                indent=2,
            )

        print()
        print(
            f"Invalid feature report:\n"
            f"{invalid_path}"
        )

    # ========================================================
    # Final result
    # ========================================================

    print()
    print("=" * 75)
    print("COMMON VOICE FEATURE EXTRACTION COMPLETE")
    print("=" * 75)

    print()
    print(
        f"Manifest samples : {len(cv_df):,}"
    )

    print(
        f"New features     : {successful:,}"
    )

    print(
        f"Skipped existing : {skipped:,}"
    )

    print(
        f"Extraction failed: {failed:,}"
    )

    print(
        f"Valid features   : {valid_features:,}"
    )

    print(
        f"Invalid features : {len(invalid_features):,}"
    )

    print(
        f"Expected shape   : {EXPECTED_FEATURE_SHAPE}"
    )

    print(
        "Expected dtype   : float32"
    )

    print()

    if (
        valid_features == 1800
        and failed == 0
        and len(invalid_features) == 0
    ):

        print(
            "STATUS: PASS"
        )

    else:

        print(
            "STATUS: FAIL"
        )

        raise RuntimeError(
            "Common Voice feature validation failed."
        )

    print()
    print(
        f"Features saved to:\n{OUTPUT_DIR}"
    )

    print()
    print(
        "Existing V3 features were NOT modified."
    )

    print(
        "No model was trained."
    )

    print("=" * 75)


if __name__ == "__main__":

    main()