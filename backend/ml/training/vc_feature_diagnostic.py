import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ============================================================
# Add backend directory to Python path
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

OUTPUT_FILE = (
    BACKEND_DIR
    / "ml"
    / "evaluation"
    / "vc_feature_diagnostic.csv"
)

SAMPLE_RATE = 16000

SAMPLES_PER_ATTACK = 100

RANDOM_STATE = 42


# ============================================================
# Feature extraction
# ============================================================

def extract_audio_statistics(file_path):

    audio, sr = load_audio(
        file_path,
        sample_rate=SAMPLE_RATE,
    )

    # --------------------------------------------------------
    # Basic signal statistics
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=256,
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=1024,
        hop_length=256,
    )[0]

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    stft = librosa.stft(
        audio,
        n_fft=1024,
        hop_length=256,
    )

    magnitude = np.abs(stft)

    # --------------------------------------------------------
    # Spectral features
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        S=magnitude,
        sr=sr,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude,
        sr=sr,
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sr,
        roll_percent=0.85,
    )[0]

    flatness = librosa.feature.spectral_flatness(
        S=magnitude,
    )[0]

    # --------------------------------------------------------
    # Spectral contrast
    # --------------------------------------------------------

    contrast = librosa.feature.spectral_contrast(
        S=magnitude,
        sr=sr,
    )

    # --------------------------------------------------------
    # Frequency-band energy
    # --------------------------------------------------------

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=1024,
    )

    power = magnitude ** 2

    bands = {
        "band_0_1k": (0, 1000),
        "band_1k_2k": (1000, 2000),
        "band_2k_4k": (2000, 4000),
        "band_4k_8k": (4000, 8000),
    }

    band_values = {}

    total_energy = (
        np.mean(power) + 1e-12
    )

    for name, (low, high) in bands.items():

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        if np.any(mask):

            energy = np.mean(
                power[mask]
            )

        else:

            energy = 0.0

        band_values[name] = (
            energy / total_energy
        )

    # --------------------------------------------------------
    # Return summary statistics
    # --------------------------------------------------------

    result = {
        "duration": len(audio) / sr,

        "rms_mean": float(
            np.mean(rms)
        ),

        "rms_std": float(
            np.std(rms)
        ),

        "zcr_mean": float(
            np.mean(zcr)
        ),

        "zcr_std": float(
            np.std(zcr)
        ),

        "spectral_centroid_mean": float(
            np.mean(centroid)
        ),

        "spectral_centroid_std": float(
            np.std(centroid)
        ),

        "spectral_bandwidth_mean": float(
            np.mean(bandwidth)
        ),

        "spectral_bandwidth_std": float(
            np.std(bandwidth)
        ),

        "spectral_rolloff_mean": float(
            np.mean(rolloff)
        ),

        "spectral_rolloff_std": float(
            np.std(rolloff)
        ),

        "spectral_flatness_mean": float(
            np.mean(flatness)
        ),

        "spectral_flatness_std": float(
            np.std(flatness)
        ),

        "spectral_contrast_mean": float(
            np.mean(contrast)
        ),

        "spectral_contrast_std": float(
            np.std(contrast)
        ),
    }

    result.update(band_values)

    return result


# ============================================================
# Load manifests
# ============================================================

print()
print("=" * 90)
print("DeepShield VC Feature Distribution Diagnostic")
print("=" * 90)

print()

train = pd.read_csv(
    TRAIN_MANIFEST
)

evaluation = pd.read_csv(
    EVAL_MANIFEST
)


# ============================================================
# Select attacks
# ============================================================

groups = {
    "Genuine": (
        train[
            train["attack_id"] == "-"
        ]
    ),

    "A01_TTS": (
        train[
            train["attack_id"] == "A01"
        ]
    ),

    "A02_TTS": (
        train[
            train["attack_id"] == "A02"
        ]
    ),

    "A03_TTS": (
        train[
            train["attack_id"] == "A03"
        ]
    ),

    "A04_TTS": (
        train[
            train["attack_id"] == "A04"
        ]
    ),

    "A05_KNOWN_VC": (
        train[
            train["attack_id"] == "A05"
        ]
    ),

    "A06_KNOWN_VC": (
        train[
            train["attack_id"] == "A06"
        ]
    ),

    "A17_UNSEEN_VC": (
        evaluation[
            evaluation["attack_id"] == "A17"
        ]
    ),

    "A18_UNSEEN_VC": (
        evaluation[
            evaluation["attack_id"] == "A18"
        ]
    ),

    "A19_UNSEEN_VC": (
        evaluation[
            evaluation["attack_id"] == "A19"
        ]
    ),
}


# ============================================================
# Run diagnostic
# ============================================================

rows = []

print(
    "Samples per group:",
    SAMPLES_PER_ATTACK
)

print()

for group_name, group_df in groups.items():

    n = min(
        SAMPLES_PER_ATTACK,
        len(group_df),
    )

    selected = group_df.sample(
        n=n,
        random_state=RANDOM_STATE,
    )

    print(
        f"{group_name:20} "
        f"Samples: {n}"
    )

    for _, row in selected.iterrows():

        try:

            features = extract_audio_statistics(
                row["audio_path"]
            )

            features.update(
                {
                    "group": group_name,
                    "attack_id": row["attack_id"],
                    "label": row["label"],
                    "audio_id": row["audio_id"],
                }
            )

            rows.append(
                features
            )

        except Exception as error:

            print(
                "ERROR:",
                row["audio_id"],
                error,
            )


results = pd.DataFrame(
    rows
)


# ============================================================
# Save raw results
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Summary
# ============================================================

numeric_columns = [
    column
    for column in results.columns
    if column not in [
        "group",
        "attack_id",
        "label",
        "audio_id",
    ]
]


summary = (
    results
    .groupby("group")[numeric_columns]
    .mean()
)


print()
print("=" * 90)
print("MEAN FEATURE VALUES")
print("=" * 90)

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    220,
)

print(
    summary.round(4).to_string()
)


# ============================================================
# VC comparison
# ============================================================

print()
print("=" * 90)
print("VC COMPARISON")
print("=" * 90)

vc_groups = [
    "A05_KNOWN_VC",
    "A06_KNOWN_VC",
    "A17_UNSEEN_VC",
    "A18_UNSEEN_VC",
    "A19_UNSEEN_VC",
]

vc_summary = (
    summary.loc[
        [
            group
            for group in vc_groups
            if group in summary.index
        ]
    ]
)

print(
    vc_summary.round(4).to_string()
)


# ============================================================
# Difference between known and unseen VC
# ============================================================

print()
print("=" * 90)
print("KNOWN VC vs UNSEEN VC")
print("=" * 90)

known_vc = results[
    results["group"].isin(
        [
            "A05_KNOWN_VC",
            "A06_KNOWN_VC",
        ]
    )
][numeric_columns].mean()

unseen_vc = results[
    results["group"].isin(
        [
            "A17_UNSEEN_VC",
            "A18_UNSEEN_VC",
            "A19_UNSEEN_VC",
        ]
    )
][numeric_columns].mean()


comparison = pd.DataFrame(
    {
        "Known_VC_A05_A06": known_vc,
        "Unseen_VC_A17_A18_A19": unseen_vc,
    }
)

comparison["Absolute_Difference"] = (
    comparison["Unseen_VC_A17_A18_A19"]
    - comparison["Known_VC_A05_A06"]
)

print(
    comparison.round(4).to_string()
)


# ============================================================
# Completion
# ============================================================

print()
print("=" * 90)
print("DIAGNOSTIC COMPLETED")
print("=" * 90)

print()
print(
    "Samples analyzed:",
    len(results)
)

print(
    "Results saved:",
    OUTPUT_FILE
)

print()