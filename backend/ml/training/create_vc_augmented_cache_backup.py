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


from ml.training.audio_processor import load_audio
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

TRAIN_MANIFEST = (
    DATASET_DIR
    / "train_manifest.csv"
)

AUG_CACHE_DIR = (
    BACKEND_DIR
    / "ml"
    / "features"
    / "cache"
    / "train_vc_augmented"
)

AUG_MANIFEST = (
    DATASET_DIR
    / "train_vc_augmented_manifest.csv"
)

AUG_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

# Add one augmented sample for approximately half
# of the existing Voice Conversion training samples.
AUGMENTATION_FRACTION = 0.50


# ============================================================
# Waveform augmentation
# ============================================================

def augment_waveform(
    audio,
    sample_rate,
    rng,
):
    """
    Apply mild waveform-level augmentation.

    The original label remains unchanged.

    Augmentations:
        1. Random circular time shift
        2. Mild additive Gaussian noise
        3. Mild amplitude scaling

    The transformations are intentionally conservative.
    """

    augmented = audio.copy().astype(
        np.float32
    )

    # --------------------------------------------------------
    # 1. Random time shift
    # --------------------------------------------------------

    max_shift = int(
        0.08 * sample_rate
    )

    shift = int(
        rng.integers(
            -max_shift,
            max_shift + 1,
        )
    )

    if shift != 0:

        augmented = np.roll(
            augmented,
            shift,
        )

    # --------------------------------------------------------
    # 2. Mild Gaussian noise
    # --------------------------------------------------------

    noise_level = float(
        rng.uniform(
            0.001,
            0.004,
        )
    )

    noise = rng.normal(
        0.0,
        noise_level,
        size=augmented.shape,
    ).astype(
        np.float32
    )

    augmented = (
        augmented + noise
    )

    # --------------------------------------------------------
    # 3. Mild amplitude variation
    # --------------------------------------------------------

    gain = float(
        rng.uniform(
            0.85,
            1.15,
        )
    )

    augmented = (
        augmented * gain
    )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    augmented = np.nan_to_num(
        augmented,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    return augmented.astype(
        np.float32
    )


# ============================================================
# Extract Mel from already-loaded waveform
# ============================================================

def waveform_to_mel(
    audio,
    sample_rate,
):
    """
    Reproduce the existing DeepShield feature pipeline
    without decoding the original file again.
    """

    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

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

    if mel_db.shape != (
        128,
        251,
    ):

        raise ValueError(
            f"Unexpected feature shape: "
            f"{mel_db.shape}"
        )

    return mel_db


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "DeepShield VC Augmented Feature Generator"
    )
    print("=" * 70)

    print()
    print(
        "Training manifest:"
    )

    print(
        TRAIN_MANIFEST
    )

    print()
    print(
        "Augmented cache:"
    )

    print(
        AUG_CACHE_DIR
    )

    # --------------------------------------------------------
    # Load training manifest
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        TRAIN_MANIFEST
    )

    vc_dataframe = dataframe[
        dataframe["label_id"] == 2
    ].copy()

    print()
    print(
        "Existing Voice Conversion samples:",
        len(vc_dataframe),
    )

    # --------------------------------------------------------
    # Select subset
    # --------------------------------------------------------

    sample_count = int(
        len(vc_dataframe)
        * AUGMENTATION_FRACTION
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    selected = vc_dataframe.sample(
        n=sample_count,
        random_state=RANDOM_SEED,
    ).reset_index(
        drop=True
    )

    print(
        "Samples selected for augmentation:",
        len(selected),
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    created = 0
    skipped = 0
    failed = 0

    augmented_rows = []

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for _, row in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc="VC augmentation",
    ):

        original_id = str(
            row["audio_id"]
        )

        augmented_id = (
            f"VC_AUG_{original_id}"
        )

        cache_path = (
            AUG_CACHE_DIR
            / f"{augmented_id}.npy"
        )

        # ----------------------------------------------------
        # Skip existing
        # ----------------------------------------------------

        if cache_path.exists():

            skipped += 1

            augmented_rows.append(
                {
                    **row.to_dict(),
                    "audio_id": augmented_id,
                    "split": "train",
                    "augmentation": "waveform",
                    "source_audio_id": original_id,
                }
            )

            continue

        try:

            # ------------------------------------------------
            # Load using existing DeepShield loader
            # ------------------------------------------------

            audio, sample_rate = load_audio(
                row["audio_path"]
            )

            # ------------------------------------------------
            # Augment
            # ------------------------------------------------

            augmented_audio = augment_waveform(
                audio,
                sample_rate,
                rng,
            )

            # ------------------------------------------------
            # Extract Mel
            # ------------------------------------------------

            mel = waveform_to_mel(
                augmented_audio,
                sample_rate,
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            np.save(
                cache_path,
                mel,
            )

            created += 1

            # ------------------------------------------------
            # Manifest row
            # ------------------------------------------------

            augmented_rows.append(
                {
                    **row.to_dict(),
                    "audio_id": augmented_id,
                    "split": "train",
                    "augmentation": "waveform",
                    "source_audio_id": original_id,
                }
            )

        except Exception as error:

            failed += 1

            print()
            print(
                "ERROR:"
            )

            print(
                "Audio:",
                row["audio_path"],
            )

            print(
                "Reason:",
                error,
            )

    # --------------------------------------------------------
    # Save augmented manifest
    # --------------------------------------------------------

    augmented_dataframe = pd.DataFrame(
        augmented_rows
    )

    augmented_dataframe.to_csv(
        AUG_MANIFEST,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "Augmentation completed."
    )

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
        "Augmented samples:",
        len(augmented_dataframe),
    )

    print(
        "Cache:",
        AUG_CACHE_DIR,
    )

    print(
        "Manifest:",
        AUG_MANIFEST,
    )

    print("=" * 70)