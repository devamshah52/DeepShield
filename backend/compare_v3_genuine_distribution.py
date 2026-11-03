from pathlib import Path
import csv
import io
import zipfile

import librosa
import numpy as np


# ============================================================
# DeepShield V3 - Genuine Distribution Comparison
# ============================================================
#
# PURPOSE:
# Compare:
#   1. Genuine ASVspoof training samples
#   2. Known genuine real-world recordings
#
# This script is READ-ONLY.
# It does NOT modify:
#   - V3 model
#   - V3 features
#   - manifests
#   - predict.py
#   - dataset archive
#
# ============================================================


# ============================================================
# Paths
# ============================================================

BACKEND_DIR = Path(
    r"C:\DeepShield\backend"
)

V3_DIR = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3"
)

TRAIN_MANIFEST = (
    V3_DIR
    / "dataset"
    / "manifests"
    / "v3_train.csv"
)

FEATURE_DIR = (
    V3_DIR
    / "features"
    / "train"
)

AUDIO_DIR = Path(
    r"C:\DeepShield\exam_test_audio"
)


# ============================================================
# Real-world genuine files
# ============================================================

REAL_GENUINE_FILES = [
    "sample-12s.wav",
    "voice-sample-44100hz-16bit.flac",
    "voice-sample.mp3",
    "voice-sample.ogg",
    "Recording_test.wav",
]


# ============================================================
# V3 preprocessing configuration
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


# ============================================================
# Output
# ============================================================

OUTPUT_FILE = (
    V3_DIR
    / "evaluation"
    / "v3_genuine_distribution_comparison.txt"
)


# ============================================================
# Helpers
# ============================================================

def separator(char="-", length=110):
    print(char * length)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def print_stat_block(
    title,
    values,
):
    print()
    print(title)
    print("-" * len(title))

    for key, value in values.items():

        if isinstance(value, float):
            print(
                f"{key:<28}: {value:.6f}"
            )
        else:
            print(
                f"{key:<28}: {value}"
            )


# ============================================================
# Feature extraction
# ============================================================

def extract_v3_mel(audio):
    """
    Exact V3 feature extraction.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # Raw statistics
    raw_mean = float(
        np.mean(mel_db)
    )

    raw_std = float(
        np.std(mel_db)
    )

    raw_min = float(
        np.min(mel_db)
    )

    raw_max = float(
        np.max(mel_db)
    )

    raw_p05 = float(
        np.percentile(
            mel_db,
            5,
        )
    )

    raw_p25 = float(
        np.percentile(
            mel_db,
            25,
        )
    )

    raw_p50 = float(
        np.percentile(
            mel_db,
            50,
        )
    )

    raw_p75 = float(
        np.percentile(
            mel_db,
            75,
        )
    )

    raw_p95 = float(
        np.percentile(
            mel_db,
            95,
        )
    )

    # V3 normalization
    mean = np.mean(mel_db)
    std = np.std(mel_db)

    if std > 0:
        normalized = (
            mel_db - mean
        ) / std
    else:
        normalized = (
            mel_db - mean
        )

    normalized = (
        normalized.astype(
            np.float32
        )
    )

    return {
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_p05": raw_p05,
        "raw_p25": raw_p25,
        "raw_p50": raw_p50,
        "raw_p75": raw_p75,
        "raw_p95": raw_p95,

        "norm_mean": float(
            np.mean(normalized)
        ),

        "norm_std": float(
            np.std(normalized)
        ),

        "norm_min": float(
            np.min(normalized)
        ),

        "norm_max": float(
            np.max(normalized)
        ),

        "norm_p05": float(
            np.percentile(
                normalized,
                5,
            )
        ),

        "norm_p25": float(
            np.percentile(
                normalized,
                25,
            )
        ),

        "norm_p50": float(
            np.percentile(
                normalized,
                50,
            )
        ),

        "norm_p75": float(
            np.percentile(
                normalized,
                75,
            )
        ),

        "norm_p95": float(
            np.percentile(
                normalized,
                95,
            )
        ),

        "shape": normalized.shape,
    }


# ============================================================
# Audio statistics
# ============================================================

def audio_statistics(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    mean = float(
        np.mean(audio)
    )

    std = float(
        np.std(audio)
    )

    threshold_40db = (
        10 ** (-40 / 20)
    )

    threshold_50db = (
        10 ** (-50 / 20)
    )

    abs_audio = np.abs(audio)

    silence_40 = float(
        np.mean(
            abs_audio
            < threshold_40db
        )
    )

    silence_50 = float(
        np.mean(
            abs_audio
            < threshold_50db
        )
    )

    return {
        "rms": rms,
        "peak": peak,
        "mean": mean,
        "std": std,
        "silence_40db": silence_40,
        "silence_50db": silence_50,
    }


# ============================================================
# Load real-world audio
# ============================================================

def load_real_audio(
    file_path,
):
    """
    Load real-world audio and apply
    exactly the V3 standardization.
    """

    audio, original_sr = librosa.load(
        str(file_path),
        sr=None,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    original_duration = (
        len(audio)
        / original_sr
    )

    original_stats = audio_statistics(
        audio
    )

    # Resample
    if original_sr != TARGET_SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=TARGET_SAMPLE_RATE,
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

    # Exactly 4 seconds
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

    standardized_stats = (
        audio_statistics(audio)
    )

    return (
        audio,
        original_sr,
        original_duration,
        original_stats,
        standardized_stats,
    )


# ============================================================
# Read V3 training manifest
# ============================================================

print()
print("=" * 110)
print("DeepShield V3 - Genuine Training Distribution Comparison")
print("=" * 110)

print()
print("V3 directory:")
print(V3_DIR)

print()
print("Training manifest:")
print(TRAIN_MANIFEST)

print()
print("Training feature directory:")
print(FEATURE_DIR)

print()
print("Real-world genuine directory:")
print(AUDIO_DIR)


if not TRAIN_MANIFEST.exists():
    raise FileNotFoundError(
        f"Training manifest not found:\n{TRAIN_MANIFEST}"
    )

if not FEATURE_DIR.exists():
    raise FileNotFoundError(
        f"Training feature directory not found:\n{FEATURE_DIR}"
    )


# ============================================================
# Load manifest
# ============================================================

print()
print("=" * 110)
print("READING V3 TRAINING MANIFEST")
print("=" * 110)

with open(
    TRAIN_MANIFEST,
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)


print(
    "Total training rows:",
    len(rows),
)


# ============================================================
# Identify genuine rows
# ============================================================

genuine_rows = []

for row in rows:

    label_value = str(
        row.get("label", "")
    ).strip()

    label_name = str(
        row.get("label_name", "")
    ).strip().lower()

    if (
        label_value == "0"
        or label_name == "genuine"
    ):

        genuine_rows.append(row)


print(
    "Genuine training rows:",
    len(genuine_rows),
)


if len(genuine_rows) == 0:
    raise RuntimeError(
        "No Genuine samples found in V3 training manifest."
    )


# ============================================================
# Determine feature path
# ============================================================

def find_feature_file(row):

    possible_columns = [
        "feature_path",
        "feature_file",
        "features",
        "feature",
    ]

    for column in possible_columns:

        value = row.get(column)

        if value:

            value = str(value).strip()

            if not value:
                continue

            candidate = Path(value)

            # Absolute path
            if candidate.is_absolute():
                if candidate.exists():
                    return candidate

            # Relative to V3 directory
            candidate = (
                V3_DIR
                / value
            )

            if candidate.exists():
                return candidate

            # Relative to train feature directory
            candidate = (
                FEATURE_DIR
                / Path(value).name
            )

            if candidate.exists():
                return candidate

    # Try audio_id
    audio_id = (
        row.get("audio_id")
        or row.get("id")
        or row.get("utt_id")
    )

    if audio_id:

        audio_id = str(
            audio_id
        ).strip()

        candidates = [
            FEATURE_DIR
            / f"{audio_id}.npy",

            FEATURE_DIR
            / f"{audio_id.replace('/', '_')}.npy",

            FEATURE_DIR
            / f"{Path(audio_id).name}.npy",
        ]

        for candidate in candidates:

            if candidate.exists():
                return candidate

    return None


# ============================================================
# Collect training feature statistics
# ============================================================

print()
print("=" * 110)
print("ANALYZING V3 GENUINE TRAINING FEATURES")
print("=" * 110)

training_feature_values = []

training_means = []
training_stds = []
training_mins = []
training_maxs = []

training_p05 = []
training_p25 = []
training_p50 = []
training_p75 = []
training_p95 = []

missing_features = 0
invalid_features = 0

MAX_TRAIN_SAMPLES = 1808

# Deterministic sampling is unnecessary because
# the manifest is already deterministic.
selected_rows = genuine_rows[
    :MAX_TRAIN_SAMPLES
]

print(
    "Samples to inspect:",
    len(selected_rows),
)

for index, row in enumerate(
    selected_rows,
    start=1,
):

    feature_path = find_feature_file(
        row
    )

    if feature_path is None:

        missing_features += 1
        continue

    try:

        feature = np.load(
            feature_path,
            allow_pickle=False,
        )

        feature = np.asarray(
            feature
        )

        if feature.shape != (
            N_MELS,
            251,
        ):

            invalid_features += 1
            continue

        if not np.isfinite(
            feature
        ).all():

            invalid_features += 1
            continue

        if feature.dtype != np.float32:

            feature = feature.astype(
                np.float32
            )

        # Per-file statistics
        training_means.append(
            float(
                np.mean(feature)
            )
        )

        training_stds.append(
            float(
                np.std(feature)
            )
        )

        training_mins.append(
            float(
                np.min(feature)
            )
        )

        training_maxs.append(
            float(
                np.max(feature)
            )
        )

        training_p05.append(
            float(
                np.percentile(
                    feature,
                    5,
                )
            )
        )

        training_p25.append(
            float(
                np.percentile(
                    feature,
                    25,
                )
            )
        )

        training_p50.append(
            float(
                np.percentile(
                    feature,
                    50,
                )
            )
        )

        training_p75.append(
            float(
                np.percentile(
                    feature,
                    75,
                )
            )
        )

        training_p95.append(
            float(
                np.percentile(
                    feature,
                    95,
                )
            )
        )

        # Keep a global sample of feature values
        flat = feature.reshape(-1)

        # Sample at most 1000 values/file
        if len(flat) > 1000:

            indices = np.linspace(
                0,
                len(flat) - 1,
                1000,
                dtype=np.int64,
            )

            flat = flat[
                indices
            ]

        training_feature_values.extend(
            flat.tolist()
        )

    except Exception:

        invalid_features += 1

    if index % 250 == 0:

        print(
            f"Processed {index}/{len(selected_rows)}"
        )


valid_training_count = len(
    training_means
)

print()
print(
    "Valid genuine feature files:",
    valid_training_count,
)

print(
    "Missing feature files:",
    missing_features,
)

print(
    "Invalid feature files:",
    invalid_features,
)


# ============================================================
# Aggregate training statistics
# ============================================================

training_feature_values = np.asarray(
    training_feature_values,
    dtype=np.float32,
)


def aggregate(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "Mean": float(
            np.mean(values)
        ),

        "Std": float(
            np.std(values)
        ),

        "Min": float(
            np.min(values)
        ),

        "Max": float(
            np.max(values)
        ),

        "P05": float(
            np.percentile(
                values,
                5,
            )
        ),

        "P25": float(
            np.percentile(
                values,
                25,
            )
        ),

        "P50": float(
            np.percentile(
                values,
                50,
            )
        ),

        "P75": float(
            np.percentile(
                values,
                75,
            )
        ),

        "P95": float(
            np.percentile(
                values,
                95,
            )
        ),
    }


training_aggregate = aggregate(
    training_feature_values
)


# ============================================================
# Print training distribution
# ============================================================

print()
print("=" * 110)
print("V3 GENUINE TRAINING FEATURE DISTRIBUTION")
print("=" * 110)

print_stat_block(
    "All normalized feature values",
    training_aggregate,
)

print()
print("Per-file normalized statistics:")
print(
    "Mean of file means :",
    np.mean(training_means),
)

print(
    "Mean of file stds  :",
    np.mean(training_stds),
)

print(
    "Mean of file mins  :",
    np.mean(training_mins),
)

print(
    "Mean of file maxs  :",
    np.mean(training_maxs),
)

print(
    "Mean of file P05   :",
    np.mean(training_p05),
)

print(
    "Mean of file P25   :",
    np.mean(training_p25),
)

print(
    "Mean of file P50   :",
    np.mean(training_p50),
)

print(
    "Mean of file P75   :",
    np.mean(training_p75),
)

print(
    "Mean of file P95   :",
    np.mean(training_p95),
)


# ============================================================
# Real-world genuine analysis
# ============================================================

print()
print("=" * 110)
print("REAL-WORLD GENUINE AUDIO ANALYSIS")
print("=" * 110)

real_results = []


for filename in REAL_GENUINE_FILES:

    separator()

    print(
        "FILE:",
        filename,
    )

    file_path = (
        AUDIO_DIR
        / filename
    )

    if not file_path.exists():

        print(
            "STATUS: FILE NOT FOUND"
        )

        continue

    try:

        (
            audio,
            original_sr,
            original_duration,
            original_stats,
            standardized_stats,
        ) = load_real_audio(
            file_path
        )

        feature_stats = (
            extract_v3_mel(
                audio
            )
        )

        print()
        print("[Original Audio]")

        print(
            "Sample Rate        :",
            original_sr,
            "Hz",
        )

        print(
            "Duration           :",
            f"{original_duration:.4f}",
            "sec",
        )

        print(
            "RMS                :",
            f"{original_stats['rms']:.6f}",
        )

        print(
            "Peak               :",
            f"{original_stats['peak']:.6f}",
        )

        print()
        print("[Standardized Audio]")

        print(
            "RMS                :",
            f"{standardized_stats['rms']:.6f}",
        )

        print(
            "Peak               :",
            f"{standardized_stats['peak']:.6f}",
        )

        print(
            "Silence < -40 dB  :",
            f"{standardized_stats['silence_40db'] * 100:.2f}%",
        )

        print(
            "Silence < -50 dB  :",
            f"{standardized_stats['silence_50db'] * 100:.2f}%",
        )

        print()
        print(
            "[Raw Mel Features - BEFORE normalization]"
        )

        print(
            "Mean               :",
            f"{feature_stats['raw_mean']:.6f}",
        )

        print(
            "Std                :",
            f"{feature_stats['raw_std']:.6f}",
        )

        print(
            "Min                :",
            f"{feature_stats['raw_min']:.6f}",
        )

        print(
            "Max                :",
            f"{feature_stats['raw_max']:.6f}",
        )

        print(
            "P05                :",
            f"{feature_stats['raw_p05']:.6f}",
        )

        print(
            "P25                :",
            f"{feature_stats['raw_p25']:.6f}",
        )

        print(
            "P50                :",
            f"{feature_stats['raw_p50']:.6f}",
        )

        print(
            "P75                :",
            f"{feature_stats['raw_p75']:.6f}",
        )

        print(
            "P95                :",
            f"{feature_stats['raw_p95']:.6f}",
        )

        print()
        print(
            "[Normalized V3 Features]"
        )

        print(
            "Mean               :",
            f"{feature_stats['norm_mean']:.6f}",
        )

        print(
            "Std                :",
            f"{feature_stats['norm_std']:.6f}",
        )

        print(
            "Min                :",
            f"{feature_stats['norm_min']:.6f}",
        )

        print(
            "Max                :",
            f"{feature_stats['norm_max']:.6f}",
        )

        print(
            "P05                :",
            f"{feature_stats['norm_p05']:.6f}",
        )

        print(
            "P25                :",
            f"{feature_stats['norm_p25']:.6f}",
        )

        print(
            "P50                :",
            f"{feature_stats['norm_p50']:.6f}",
        )

        print(
            "P75                :",
            f"{feature_stats['norm_p75']:.6f}",
        )

        print(
            "P95                :",
            f"{feature_stats['norm_p95']:.6f}",
        )

        real_results.append({
            "file": filename,
            "raw_mean": feature_stats[
                "raw_mean"
            ],
            "raw_std": feature_stats[
                "raw_std"
            ],
            "raw_p05": feature_stats[
                "raw_p05"
            ],
            "raw_p25": feature_stats[
                "raw_p25"
            ],
            "raw_p50": feature_stats[
                "raw_p50"
            ],
            "raw_p75": feature_stats[
                "raw_p75"
            ],
            "raw_p95": feature_stats[
                "raw_p95"
            ],
            "rms": standardized_stats[
                "rms"
            ],
            "peak": standardized_stats[
                "peak"
            ],
            "silence_40": standardized_stats[
                "silence_40db"
            ],
            "silence_50": standardized_stats[
                "silence_50db"
            ],
        })

        print()
        print(
            "STATUS: PASS"
        )

    except Exception as exc:

        print()
        print(
            "STATUS: ERROR"
        )

        print(
            "ERROR:",
            repr(exc),
        )


# ============================================================
# Comparison
# ============================================================

print()
print("=" * 110)
print("TRAINING vs REAL-WORLD COMPARISON")
print("=" * 110)

print()
print(
    "The normalized feature mean/std are expected to be"
)
print(
    "approximately 0/1 for both groups because V3 performs"
)
print(
    "per-file normalization."
)

print()
print(
    "Therefore, the more useful comparison is the RAW Mel"
)
print(
    "distribution and the audio-level statistics."
)


print()
print(
    f"{'File':40} "
    f"{'Raw Mean':>10} "
    f"{'Raw Std':>10} "
    f"{'P05':>10} "
    f"{'P50':>10} "
    f"{'P95':>10} "
    f"{'RMS':>10} "
    f"{'Silence40':>11}"
)

separator("-", 120)

for result in real_results:

    print(
        f"{result['file'][:40]:40} "
        f"{result['raw_mean']:10.2f} "
        f"{result['raw_std']:10.2f} "
        f"{result['raw_p05']:10.2f} "
        f"{result['raw_p50']:10.2f} "
        f"{result['raw_p95']:10.2f} "
        f"{result['rms']:10.4f} "
        f"{result['silence_40'] * 100:10.2f}%"
    )


# ============================================================
# Training raw distribution cannot be recovered from
# normalized .npy features alone.
# ============================================================

print()
print("=" * 110)
print("IMPORTANT LIMITATION")
print("=" * 110)

print(
    """
The existing V3 feature cache contains the NORMALIZED
Mel-spectrograms.

Because V3 applies per-file normalization:

    (mel_db - mean) / std

the original absolute Mel mean/std of the training files
cannot be reconstructed from the normalized .npy files.

Therefore this diagnostic compares:

1. normalized feature distributions
2. raw Mel distributions of the real-world files
3. audio-level characteristics of the real-world files

To compare RAW Mel distributions of the ASVspoof training
audio directly, we would need to read the original FLAC
files from LA.zip and calculate their raw Mel features.
"""
)


# ============================================================
# Save report
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "DeepShield V3 Genuine Distribution Comparison\n"
    )

    report.write(
        "=" * 100 + "\n\n"
    )

    report.write(
        f"Valid V3 genuine training features: "
        f"{valid_training_count}\n"
    )

    report.write(
        f"Missing features: {missing_features}\n"
    )

    report.write(
        f"Invalid features: {invalid_features}\n\n"
    )

    report.write(
        "Training normalized feature aggregate:\n"
    )

    for key, value in training_aggregate.items():

        report.write(
            f"{key}: {value}\n"
        )

    report.write(
        "\nReal-world genuine files:\n"
    )

    for result in real_results:

        report.write(
            "\n"
            + "-" * 80
            + "\n"
        )

        for key, value in result.items():

            report.write(
                f"{key}: {value}\n"
            )


print()
print("=" * 110)
print("REPORT SAVED")
print("=" * 110)

print(
    OUTPUT_FILE
)

print()
print("COMPARISON COMPLETE")
print("=" * 110)