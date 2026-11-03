import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

# ============================================================
# Backend path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.training.audio_processor import load_audio


# ============================================================
# Configuration
# ============================================================

TRAIN_MANIFEST = (
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "train_manifest.csv"
)

EVAL_MANIFEST = (
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "eval_manifest.csv"
)

SAMPLES_PER_GROUP = 100


# ============================================================
# Raw Mel extraction
# ============================================================

def extract_raw_mel(file_path):

    audio, sr = load_audio(
        file_path
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    return mel_db.astype(
        np.float32
    )


# ============================================================
# Statistics
# ============================================================

def get_statistics(file_path):

    mel = extract_raw_mel(
        file_path
    )

    return {
        "mel_mean": float(
            mel.mean()
        ),

        "mel_std": float(
            mel.std()
        ),

        "mel_min": float(
            mel.min()
        ),

        "mel_max": float(
            mel.max()
        ),

        "mel_p10": float(
            np.percentile(
                mel,
                10,
            )
        ),

        "mel_p25": float(
            np.percentile(
                mel,
                25,
            )
        ),

        "mel_median": float(
            np.percentile(
                mel,
                50,
            )
        ),

        "mel_p75": float(
            np.percentile(
                mel,
                75,
            )
        ),

        "mel_p90": float(
            np.percentile(
                mel,
                90,
            )
        ),
    }


# ============================================================
# Load manifests
# ============================================================

train = pd.read_csv(
    TRAIN_MANIFEST
)

eval_df = pd.read_csv(
    EVAL_MANIFEST
)


# ============================================================
# Select diagnostic groups
# ============================================================

groups = {

    "Genuine":
        train[
            train["attack_id"] == "-"
        ].head(
            SAMPLES_PER_GROUP
        ),

    "A05_KNOWN_VC":
        train[
            train["attack_id"] == "A05"
        ].head(
            SAMPLES_PER_GROUP
        ),

    "A06_KNOWN_VC":
        train[
            train["attack_id"] == "A06"
        ].head(
            SAMPLES_PER_GROUP
        ),

    "A17_UNSEEN_VC":
        eval_df[
            eval_df["attack_id"] == "A17"
        ].head(
            SAMPLES_PER_GROUP
        ),

    "A18_UNSEEN_VC":
        eval_df[
            eval_df["attack_id"] == "A18"
        ].head(
            SAMPLES_PER_GROUP
        ),

    "A19_UNSEEN_VC":
        eval_df[
            eval_df["attack_id"] == "A19"
        ].head(
            SAMPLES_PER_GROUP
        ),
}


# ============================================================
# Run diagnostic
# ============================================================

print()
print("=" * 100)
print(
    "DeepShield RAW Mel Spectrogram Diagnostic"
)
print("=" * 100)
print()

print(
    "These values are BEFORE the per-file mean/std normalization."
)

print()

all_results = []


for group_name, dataframe in groups.items():

    print(
        f"Processing {group_name}: "
        f"{len(dataframe)} samples"
    )

    for _, row in dataframe.iterrows():

        stats = get_statistics(
            row["audio_path"]
        )

        stats["group"] = group_name
        stats["audio_id"] = row["audio_id"]

        all_results.append(
            stats
        )


results = pd.DataFrame(
    all_results
)


# ============================================================
# Group summary
# ============================================================

numeric_columns = [
    "mel_mean",
    "mel_std",
    "mel_min",
    "mel_max",
    "mel_p10",
    "mel_p25",
    "mel_median",
    "mel_p75",
    "mel_p90",
]


summary = (
    results
    .groupby("group")[numeric_columns]
    .mean()
)


print()
print("=" * 100)
print(
    "RAW MEL FEATURE DISTRIBUTIONS"
)
print("=" * 100)
print()

print(
    summary.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# Known vs unseen VC
# ============================================================

known = results[
    results["group"].isin(
        [
            "A05_KNOWN_VC",
            "A06_KNOWN_VC",
        ]
    )
]

unseen = results[
    results["group"].isin(
        [
            "A17_UNSEEN_VC",
            "A18_UNSEEN_VC",
            "A19_UNSEEN_VC",
        ]
    )
]


print()
print("=" * 100)
print(
    "KNOWN VC vs UNSEEN VC — RAW MEL"
)
print("=" * 100)
print()

known_mean = (
    known[numeric_columns]
    .mean()
)

unseen_mean = (
    unseen[numeric_columns]
    .mean()
)

comparison = pd.DataFrame(
    {
        "Known_VC_A05_A06":
            known_mean,

        "Unseen_VC_A17_A18_A19":
            unseen_mean,

        "Absolute_Difference":
            unseen_mean - known_mean,
    }
)


print(
    comparison.to_string(
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# Save results
# ============================================================

output_file = (
    BACKEND_DIR
    / "ml"
    / "evaluation"
    / "raw_mel_diagnostic.csv"
)

results.to_csv(
    output_file,
    index=False,
)


print()
print("=" * 100)
print(
    "DIAGNOSTIC COMPLETED"
)
print("=" * 100)
print()

print(
    "Samples analyzed:",
    len(results)
)

print(
    "Results saved:",
    output_file
)

print()