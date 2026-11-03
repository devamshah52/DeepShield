"""
DeepShield V3 - Feature Extraction

Reads the V3 manifests and extracts Mel-spectrogram features
directly from ASVspoof 2019 LA audio stored inside LA.zip.

V3 preprocessing:
    - 16 kHz
    - Mono
    - Exactly 4 seconds
    - n_fft = 1024
    - hop_length = 256
    - n_mels = 128
    - power = 2.0
    - dB conversion
    - Per-file mean/std normalization

Output:
    ml/experiments/v3/features/
        train/
        dev/
        test/

Existing V1/V2 datasets and preprocessing files are NOT modified.
LA.zip is opened read-only.
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import librosa
import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

BACKEND_DIR = PROJECT_ROOT / "backend"

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

METADATA_DIR = (
    V3_DIR
    / "dataset"
    / "metadata"
)

LA_ZIP = (
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "LA.zip"
)


# ============================================================
# ASVspoof archive configuration
# ============================================================

AUDIO_PREFIX = (
    "LA/"
    "ASVspoof2019_LA_train/"
    "flac/"
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
# V3 configuration
# ============================================================

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
# Utility
# ============================================================

def ensure_directories():

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in SPLITS:

        (
            FEATURE_DIR / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# Load audio from ZIP
# ============================================================

def load_audio_from_zip(
    archive: zipfile.ZipFile,
    member_path: str,
):
    """
    Read one FLAC file directly from LA.zip.

    The audio is decoded using librosa/soundfile and then
    standardized to:
        - mono
        - 16 kHz
        - exactly 4 seconds
    """

    with archive.open(
        member_path,
        "r",
    ) as audio_file:

        audio_bytes = (
            audio_file.read()
        )

    # --------------------------------------------------------
    # Decode FLAC bytes
    # --------------------------------------------------------

    import io

    audio, sample_rate = (
        librosa.load(
            io.BytesIO(audio_bytes),
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
    Convert standardized audio into a normalized
    Mel-spectrogram.

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
# Read manifest
# ============================================================

def read_manifest(
    path: Path,
):
    """
    Read one V3 manifest.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{path}"
        )

    records = []

    with path.open(
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


# ============================================================
# Feature filename
# ============================================================

def feature_filename(
    audio_id: str,
):
    """
    Generate a safe feature filename.
    """

    return (
        f"{audio_id}.npy"
    )


# ============================================================
# Extract one split
# ============================================================

def extract_split(
    archive: zipfile.ZipFile,
    split: str,
):
    """
    Extract all features for one split.
    """

    manifest_path = (
        MANIFEST_DIR
        / MANIFEST_NAMES[split]
    )

    records = read_manifest(
        manifest_path
    )

    output_dir = (
        FEATURE_DIR
        / split
    )

    print()
    print("=" * 70)
    print(
        f"Extracting V3 {split.upper()} features"
    )
    print("=" * 70)

    print(
        f"Manifest samples: {len(records)}"
    )

    successful = 0

    failed = 0

    skipped = 0

    failure_records = []

    label_counts = Counter()

    feature_shapes = Counter()

    for index, record in enumerate(
        records,
        start=1,
    ):

        audio_id = (
            record["audio_id"]
        )

        audio_path = (
            record["audio_path"]
        )

        label_name = (
            record["label_name"]
        )

        label_id = int(
            record["label_id"]
        )

        output_path = (
            output_dir
            / feature_filename(
                audio_id
            )
        )

        # ----------------------------------------------------
        # Skip existing valid features
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

                    label_counts[
                        label_name
                    ] += 1

                    feature_shapes[
                        str(existing.shape)
                    ] += 1

                    continue

            except Exception:
                pass

        # ----------------------------------------------------
        # Construct archive member
        # ----------------------------------------------------

        archive_member = (
            audio_path
        )

        if not archive_member.startswith(
            AUDIO_PREFIX
        ):

            archive_member = (
                AUDIO_PREFIX
                + Path(
                    audio_path
                ).name
            )

        try:

            # ------------------------------------------------
            # Read audio from ZIP
            # ------------------------------------------------

            audio, sample_rate = (
                load_audio_from_zip(
                    archive,
                    archive_member,
                )
            )

            # ------------------------------------------------
            # Extract Mel
            # ------------------------------------------------

            features = (
                extract_mel_spectrogram(
                    audio,
                    sample_rate,
                )
            )

            # ------------------------------------------------
            # Save feature
            # ------------------------------------------------

            np.save(
                output_path,
                features,
                allow_pickle=False,
            )

            successful += 1

            label_counts[
                label_name
            ] += 1

            feature_shapes[
                str(features.shape)
            ] += 1

        except Exception as error:

            failed += 1

            failure_records.append(
                {
                    "audio_id": audio_id,
                    "audio_path": audio_path,
                    "label_name": label_name,
                    "label_id": label_id,
                    "error": str(error),
                }
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index == 1
            or index % 250 == 0
            or index == len(records)
        ):

            print(
                f"[{index:5d}/{len(records)}] "
                f"success={successful} "
                f"skipped={skipped} "
                f"failed={failed}"
            )

    # ========================================================
    # Split summary
    # ========================================================

    summary = {
        "split": split,
        "manifest_samples": len(records),
        "successful": successful,
        "skipped_existing": skipped,
        "failed": failed,
        "total_available": (
            successful
            + skipped
            + failed
        ),
        "feature_shape": (
            list(EXPECTED_FEATURE_SHAPE)
        ),
        "feature_dtype": "float32",
        "class_counts": dict(
            label_counts
        ),
        "feature_shapes": dict(
            feature_shapes
        ),
    }

    summary_path = (
        METADATA_DIR
        / f"v3_{split}_feature_summary.json"
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

    print()
    print(
        f"{split.upper()} extraction complete."
    )

    print(
        f"  Successful : {successful}"
    )

    print(
        f"  Skipped    : {skipped}"
    )

    print(
        f"  Failed     : {failed}"
    )

    print(
        f"  Shape      : {EXPECTED_FEATURE_SHAPE}"
    )

    print(
        "  Classes:"
    )

    for label in (
        "Genuine",
        "TTS",
        "Voice_Conversion",
    ):

        print(
            f"    {label:18s}: "
            f"{label_counts[label]}"
        )

    # --------------------------------------------------------
    # Failure report
    # --------------------------------------------------------

    if failure_records:

        failure_path = (
            METADATA_DIR
            / f"v3_{split}_feature_failures.json"
        )

        with failure_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                failure_records,
                file,
                indent=2,
            )

        print()
        print(
            "Failure report:"
        )

        print(
            failure_path
        )

    return summary


# ============================================================
# Final validation
# ============================================================

def validate_feature_cache(
    summaries,
):
    """
    Validate that every manifest sample has a valid feature.
    """

    print()
    print("=" * 70)
    print("Validating V3 feature cache")
    print("=" * 70)

    total_manifest = 0

    total_available = 0

    total_failed = 0

    all_passed = True

    for split in SPLITS:

        summary = summaries[split]

        manifest_samples = (
            summary[
                "manifest_samples"
            ]
        )

        available = (
            summary[
                "successful"
            ]
            + summary[
                "skipped_existing"
            ]
        )

        failed = (
            summary[
                "failed"
            ]
        )

        total_manifest += (
            manifest_samples
        )

        total_available += (
            available
        )

        total_failed += (
            failed
        )

        expected_classes = {
            "Genuine": None,
            "TTS": None,
            "Voice_Conversion": None,
        }

        manifest_records = read_manifest(
            MANIFEST_DIR
            / MANIFEST_NAMES[split]
        )

        manifest_counts = Counter(
            record["label_name"]
            for record in manifest_records
        )

        feature_counts = Counter()

        output_dir = (
            FEATURE_DIR
            / split
        )

        for record in manifest_records:

            feature_path = (
                output_dir
                / feature_filename(
                    record["audio_id"]
                )
            )

            if not feature_path.exists():

                all_passed = False

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

                    all_passed = False

                    continue

                if (
                    feature.dtype
                    != np.float32
                ):

                    all_passed = False

                    continue

                feature_counts[
                    record["label_name"]
                ] += 1

            except Exception:

                all_passed = False

        print()
        print(
            split.upper()
        )

        print(
            f"  Manifest samples : "
            f"{manifest_samples}"
        )

        print(
            f"  Valid features   : "
            f"{sum(feature_counts.values())}"
        )

        print(
            f"  Failed extraction: "
            f"{failed}"
        )

        print(
            "  Class validation:"
        )

        for label in (
            "Genuine",
            "TTS",
            "Voice_Conversion",
        ):

            expected = (
                manifest_counts[label]
            )

            actual = (
                feature_counts[label]
            )

            status = (
                "PASS"
                if expected == actual
                else "FAIL"
            )

            print(
                f"    {label:18s}: "
                f"{actual}/{expected} "
                f"{status}"
            )

            if expected != actual:

                all_passed = False

    print()
    print("=" * 70)

    print(
        f"Total manifest samples : "
        f"{total_manifest}"
    )

    print(
        f"Total valid features   : "
        f"{total_available}"
    )

    print(
        f"Total failed           : "
        f"{total_failed}"
    )

    if (
        total_manifest
        != total_available
    ):

        all_passed = False

    if total_failed != 0:

        all_passed = False

    print()

    if all_passed:

        print(
            "V3 FEATURE CACHE VALIDATION: PASS"
        )

    else:

        print(
            "V3 FEATURE CACHE VALIDATION: FAIL"
        )

        raise RuntimeError(
            "V3 feature cache validation failed."
        )

    # --------------------------------------------------------
    # Save global summary
    # --------------------------------------------------------

    final_summary = {
        "total_manifest_samples": (
            total_manifest
        ),
        "total_valid_features": (
            total_available
        ),
        "total_failed": (
            total_failed
        ),
        "expected_feature_shape": (
            list(
                EXPECTED_FEATURE_SHAPE
            )
        ),
        "dtype": "float32",
        "splits": summaries,
    }

    final_path = (
        METADATA_DIR
        / "v3_feature_extraction_summary.json"
    )

    with final_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_summary,
            file,
            indent=2,
        )

    print()
    print(
        "Created:"
    )

    print(
        final_path
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DeepShield V3 - Feature Extraction")
    print("=" * 70)

    print()
    print(
        f"LA archive:\n{LA_ZIP}"
    )

    print()
    print(
        f"V3 feature directory:\n"
        f"{FEATURE_DIR}"
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not LA_ZIP.exists():

        raise FileNotFoundError(
            f"LA.zip not found:\n{LA_ZIP}"
        )

    for split in SPLITS:

        manifest_path = (
            MANIFEST_DIR
            / MANIFEST_NAMES[split]
        )

        if not manifest_path.exists():

            raise FileNotFoundError(
                f"Manifest not found:\n"
                f"{manifest_path}"
            )

    ensure_directories()

    # --------------------------------------------------------
    # Open archive read-only
    # --------------------------------------------------------

    print()
    print(
        "Opening LA.zip in read-only mode..."
    )

    with zipfile.ZipFile(
        LA_ZIP,
        "r",
    ) as archive:

        archive_members = set(
            archive.namelist()
        )

        print(
            f"Archive members: "
            f"{len(archive_members)}"
        )

        summaries = {}

        # ----------------------------------------------------
        # Extract each split
        # ----------------------------------------------------

        for split in SPLITS:

            summaries[split] = (
                extract_split(
                    archive,
                    split,
                )
            )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    validate_feature_cache(
        summaries
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "V3 FEATURE EXTRACTION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "LA.zip was NOT modified."
    )

    print(
        "Existing V1/V2 datasets were NOT modified."
    )

    print(
        "Existing V1/V2 preprocessing files were NOT modified."
    )

    print()
    print(
        f"Features saved under:\n"
        f"{FEATURE_DIR}"
    )


if __name__ == "__main__":

    main()