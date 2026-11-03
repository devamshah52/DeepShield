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

# Original training VC samples:
#
# A05 = 3800
# A06 = 3800
#
# Create approximately one augmented sample for 50%
# of the available VC training samples.
#
# Total:
# approximately 3800 augmented samples.
#
AUGMENTATION_FRACTION = 0.50


TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLES = (
    TARGET_SAMPLE_RATE * 4
)


# ============================================================
# Utility
# ============================================================

def safe_normalize(
    audio,
):
    """
    Safely normalize waveform amplitude.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    if peak > 1e-8:

        audio = (
            audio / peak
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# Random circular time shift
# ============================================================

def random_time_shift(
    audio,
    sample_rate,
    rng,
):
    """
    Apply a random circular time shift.
    """

    max_shift = int(
        0.08 * sample_rate
    )

    shift = int(
        rng.integers(
            -max_shift,
            max_shift + 1,
        )
    )

    if shift == 0:
        return audio

    return np.roll(
        audio,
        shift,
    )


# ============================================================
# Amplitude variation
# ============================================================

def random_gain(
    audio,
    rng,
):
    """
    Apply moderate amplitude variation.
    """

    gain = float(
        rng.uniform(
            0.75,
            1.25,
        )
    )

    return (
        audio * gain
    ).astype(
        np.float32
    )


# ============================================================
# Additive noise
# ============================================================

def add_noise(
    audio,
    rng,
):
    """
    Add low-level Gaussian noise.
    """

    noise_level = float(
        rng.uniform(
            0.001,
            0.006,
        )
    )

    noise = rng.normal(
        0.0,
        noise_level,
        size=audio.shape,
    ).astype(
        np.float32
    )

    return (
        audio + noise
    ).astype(
        np.float32
    )


# ============================================================
# Random polarity
# ============================================================

def random_polarity(
    audio,
    rng,
):
    """
    Randomly invert waveform polarity.
    """

    if rng.random() < 0.50:

        return -audio

    return audio


# ============================================================
# Random temporal crop
# ============================================================

def random_temporal_crop(
    audio,
    rng,
):
    """
    Randomly crop a small amount from the beginning
    and end, then restore the original duration.

    This changes temporal alignment without changing
    the final feature size.
    """

    length = len(audio)

    if length <= 1000:

        return audio

    max_crop = int(
        0.05 * length
    )

    if max_crop < 1:

        return audio

    left = int(
        rng.integers(
            0,
            max_crop + 1,
        )
    )

    right = int(
        rng.integers(
            0,
            max_crop + 1,
        )
    )

    start = left

    end = (
        length - right
    )

    if end <= start + 100:

        return audio

    cropped = audio[
        start:end
    ]

    if len(cropped) == length:

        return cropped

    # Resize back to exactly target length
    old_positions = np.linspace(
        0,
        1,
        num=len(cropped),
    )

    new_positions = np.linspace(
        0,
        1,
        num=length,
    )

    resized = np.interp(
        new_positions,
        old_positions,
        cropped,
    )

    return resized.astype(
        np.float32
    )


# ============================================================
# Simple resampling / speed variation
# ============================================================

def random_time_scale(
    audio,
    rng,
):
    """
    Apply a small time-scale variation.

    The resulting waveform is restored to exactly
    the original number of samples.

    This does NOT use external audio libraries.
    """

    length = len(audio)

    scale = float(
        rng.uniform(
            0.92,
            1.08,
        )
    )

    new_length = int(
        length / scale
    )

    new_length = max(
        1000,
        new_length,
    )

    old_positions = np.linspace(
        0,
        1,
        num=length,
    )

    new_positions = np.linspace(
        0,
        1,
        num=new_length,
    )

    stretched = np.interp(
        new_positions,
        old_positions,
        audio,
    )

    # Restore exactly 4 seconds
    restore_positions = np.linspace(
        0,
        1,
        num=length,
    )

    source_positions = np.linspace(
        0,
        1,
        num=len(stretched),
    )

    restored = np.interp(
        restore_positions,
        source_positions,
        stretched,
    )

    return restored.astype(
        np.float32
    )


# ============================================================
# Spectral-style waveform filtering
# ============================================================

def random_high_frequency_filter(
    audio,
    sample_rate,
    rng,
):
    """
    Apply a simple frequency-domain attenuation.

    This is intentionally mild and is designed to create
    spectral diversity rather than destroy the original
    VC characteristics.
    """

    n = len(audio)

    spectrum = np.fft.rfft(
        audio
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sample_rate,
    )

    cutoff = float(
        rng.uniform(
            4500,
            7500,
        )
    )

    attenuation = np.ones_like(
        frequencies,
        dtype=np.float32,
    )

    high_frequency = (
        frequencies > cutoff
    )

    if np.any(
        high_frequency
    ):

        ratio = (
            frequencies[
                high_frequency
            ]
            - cutoff
        ) / (
            sample_rate / 2
            - cutoff
            + 1e-8
        )

        attenuation[
            high_frequency
        ] = (
            1.0
            - 0.35 * ratio
        )

    spectrum = (
        spectrum
        * attenuation
    )

    filtered = np.fft.irfft(
        spectrum,
        n=n,
    )

    return filtered.astype(
        np.float32
    )


# ============================================================
# Spectral-style low-frequency filtering
# ============================================================

def random_low_frequency_filter(
    audio,
    sample_rate,
    rng,
):
    """
    Apply mild low-frequency attenuation.
    """

    n = len(audio)

    spectrum = np.fft.rfft(
        audio
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sample_rate,
    )

    cutoff = float(
        rng.uniform(
            60,
            180,
        )
    )

    attenuation = np.ones_like(
        frequencies,
        dtype=np.float32,
    )

    low_frequency = (
        frequencies < cutoff
    )

    if np.any(
        low_frequency
    ):

        ratio = (
            frequencies[
                low_frequency
            ]
            / (
                cutoff
                + 1e-8
            )
        )

        attenuation[
            low_frequency
        ] = (
            0.70
            + 0.30 * ratio
        )

    spectrum = (
        spectrum
        * attenuation
    )

    filtered = np.fft.irfft(
        spectrum,
        n=n,
    )

    return filtered.astype(
        np.float32
    )


# ============================================================
# Main VC augmentation pipeline
# ============================================================

def augment_vc_waveform(
    audio,
    sample_rate,
    rng,
    augmentation_type,
):
    """
    Create a diverse VC training augmentation.

    All transformations preserve the VC label.
    """

    augmented = (
        audio.copy()
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Type 0
    # Temporal + amplitude
    # --------------------------------------------------------

    if augmentation_type == 0:

        augmented = random_time_shift(
            augmented,
            sample_rate,
            rng,
        )

        augmented = random_temporal_crop(
            augmented,
            rng,
        )

        augmented = random_gain(
            augmented,
            rng,
        )

    # --------------------------------------------------------
    # Type 1
    # Time-scale + gain
    # --------------------------------------------------------

    elif augmentation_type == 1:

        augmented = random_time_scale(
            augmented,
            rng,
        )

        augmented = random_gain(
            augmented,
            rng,
        )

    # --------------------------------------------------------
    # Type 2
    # Noise + temporal shift
    # --------------------------------------------------------

    elif augmentation_type == 2:

        augmented = random_time_shift(
            augmented,
            sample_rate,
            rng,
        )

        augmented = add_noise(
            augmented,
            rng,
        )

        augmented = random_gain(
            augmented,
            rng,
        )

    # --------------------------------------------------------
    # Type 3
    # High-frequency filtering
    # --------------------------------------------------------

    elif augmentation_type == 3:

        augmented = random_high_frequency_filter(
            augmented,
            sample_rate,
            rng,
        )

        augmented = random_gain(
            augmented,
            rng,
        )

        if rng.random() < 0.50:

            augmented = add_noise(
                augmented,
                rng,
            )

    # --------------------------------------------------------
    # Type 4
    # Low-frequency filtering
    # --------------------------------------------------------

    elif augmentation_type == 4:

        augmented = random_low_frequency_filter(
            augmented,
            sample_rate,
            rng,
        )

        augmented = random_time_shift(
            augmented,
            sample_rate,
            rng,
        )

        augmented = random_gain(
            augmented,
            rng,
        )

    # --------------------------------------------------------
    # Type 5
    # Combined stronger variation
    # --------------------------------------------------------

    else:

        augmented = random_time_scale(
            augmented,
            rng,
        )

        augmented = random_high_frequency_filter(
            augmented,
            sample_rate,
            rng,
        )

        augmented = add_noise(
            augmented,
            rng,
        )

        augmented = random_gain(
            augmented,
            rng,
        )

    # --------------------------------------------------------
    # Random polarity
    # --------------------------------------------------------

    augmented = random_polarity(
        augmented,
        rng,
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

    # Keep values in a safe range
    augmented = np.clip(
        augmented,
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Ensure exactly 4 seconds
    # --------------------------------------------------------

    if len(augmented) > TARGET_SAMPLES:

        augmented = augmented[
            :TARGET_SAMPLES
        ]

    elif len(augmented) < TARGET_SAMPLES:

        padding = (
            TARGET_SAMPLES
            - len(augmented)
        )

        augmented = np.pad(
            augmented,
            (0, padding),
            mode="constant",
        )

    return augmented.astype(
        np.float32
    )


# ============================================================
# Mel extraction
# ============================================================

def waveform_to_mel(
    audio,
    sample_rate,
):
    """
    Convert augmented waveform to the exact same
    normalized Mel representation used by training.
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
            "Unexpected feature shape: "
            f"{mel_db.shape}"
        )

    return mel_db


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 75)
    print(
        "DeepShield Diverse VC Augmented Feature Generator"
    )
    print("=" * 75)

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
    # Load manifest
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        TRAIN_MANIFEST
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Only use actual VC training samples.
    #
    # A05 and A06 are the only VC attacks present
    # in train_manifest.csv.
    #
    # A17/A18/A19 are NOT loaded here.
    # --------------------------------------------------------

    vc_dataframe = dataframe[
        dataframe["label_id"] == 2
    ].copy()

    print()
    print(
        "Existing Voice Conversion samples:",
        len(vc_dataframe),
    )

    print()
    print(
        "Training VC attack distribution:"
    )

    print(
        vc_dataframe[
            "attack_id"
        ].value_counts().to_string()
    )

    # --------------------------------------------------------
    # Select samples
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

    print()
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

    for selected_index, row in tqdm(
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
                    "augmentation": "diverse_waveform",
                    "source_audio_id": original_id,
                }
            )

            continue

        try:

            # ------------------------------------------------
            # Load original
            # ------------------------------------------------

            audio, sample_rate = load_audio(
                row["audio_path"],
                sample_rate=TARGET_SAMPLE_RATE,
            )

            # ------------------------------------------------
            # Deterministic per-sample RNG
            #
            # This ensures reproducibility.
            # ------------------------------------------------

            local_seed = (
                RANDOM_SEED
                + selected_index
            )

            local_rng = np.random.default_rng(
                local_seed
            )

            # ------------------------------------------------
            # Select one of six augmentation families
            # ------------------------------------------------

            augmentation_type = int(
                local_rng.integers(
                    0,
                    6,
                )
            )

            augmented_audio = (
                augment_vc_waveform(
                    audio,
                    sample_rate,
                    local_rng,
                    augmentation_type,
                )
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
                    "augmentation": (
                        f"diverse_waveform_{augmentation_type}"
                    ),
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
    print("=" * 75)

    print(
        "Diverse VC augmentation completed."
    )

    print()

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
        len(
            augmented_dataframe
        ),
    )

    print()

    print(
        "Augmented attack distribution:"
    )

    if len(
        augmented_dataframe
    ) > 0:

        print(
            augmented_dataframe[
                "attack_id"
            ].value_counts().to_string()
        )

    print()

    print(
        "Augmentation distribution:"
    )

    if (
        len(
            augmented_dataframe
        ) > 0
        and "augmentation"
        in augmented_dataframe.columns
    ):

        print(
            augmented_dataframe[
                "augmentation"
            ].value_counts().to_string()
        )

    print()

    print(
        "Cache:"
    )

    print(
        AUG_CACHE_DIR
    )

    print()

    print(
        "Manifest:"
    )

    print(
        AUG_MANIFEST
    )

    print("=" * 75)