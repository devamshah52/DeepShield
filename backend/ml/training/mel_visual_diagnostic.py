import sys
from pathlib import Path

import matplotlib.pyplot as plt
import librosa.display
import pandas as pd


# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from ml.training.audio_processor import extract_mel_spectrogram


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = (
    BACKEND_DIR
    / "ml"
    / "dataset"
)

EVAL_MANIFEST = (
    DATASET_DIR
    / "eval_manifest.csv"
)

TRAIN_MANIFEST = (
    DATASET_DIR
    / "train_manifest.csv"
)

OUTPUT_DIR = (
    BACKEND_DIR
    / "ml"
    / "evaluation"
    / "mel_visual_diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Select representative samples
# ============================================================

train_df = pd.read_csv(
    TRAIN_MANIFEST
)

eval_df = pd.read_csv(
    EVAL_MANIFEST
)


samples = []


# Genuine from training set
row = train_df[
    train_df["attack_id"] == "-"
].iloc[0]

samples.append(
    (
        "Genuine",
        "TRAIN",
        row["audio_id"],
        row["audio_path"],
    )
)


# Known VC
for attack_id in ["A05", "A06"]:

    row = train_df[
        train_df["attack_id"] == attack_id
    ].iloc[0]

    samples.append(
        (
            attack_id,
            "KNOWN VC",
            row["audio_id"],
            row["audio_path"],
        )
    )


# Unseen VC
for attack_id in ["A17", "A18", "A19"]:

    row = eval_df[
        eval_df["attack_id"] == attack_id
    ].iloc[0]

    samples.append(
        (
            attack_id,
            "UNSEEN VC",
            row["audio_id"],
            row["audio_path"],
        )
    )


# ============================================================
# Print selected samples
# ============================================================

print()
print("=" * 100)
print("DeepShield Mel Spectrogram Visual Diagnostic")
print("=" * 100)
print()

for attack_id, category, audio_id, audio_path in samples:

    print(
        f"{attack_id:10} | "
        f"{category:10} | "
        f"{audio_id:20} | "
        f"{audio_path}"
    )


# ============================================================
# Generate individual spectrograms
# ============================================================

print()
print("=" * 100)
print("Generating spectrograms")
print("=" * 100)
print()


for attack_id, category, audio_id, audio_path in samples:

    print(
        f"Processing {attack_id}..."
    )

    features = extract_mel_spectrogram(
        audio_path
    )

    plt.figure(
        figsize=(14, 6)
    )

    librosa.display.specshow(
        features,
        sr=16000,
        hop_length=256,
        x_axis="time",
        y_axis="mel",
    )

    plt.colorbar(
        format="%+2.1f"
    )

    plt.title(
        f"DeepShield Mel Spectrogram - "
        f"{attack_id} ({category})\n"
        f"{audio_id}"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Mel Frequency"
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{attack_id}_{audio_id}.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Generate comparison image
# ============================================================

print()
print(
    "Generating comparison image..."
)


fig, axes = plt.subplots(
    6,
    1,
    figsize=(16, 20),
)


for index, (
    attack_id,
    category,
    audio_id,
    audio_path,
) in enumerate(samples):

    features = extract_mel_spectrogram(
        audio_path
    )

    image = axes[index].imshow(
        features,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    axes[index].set_title(
        f"{attack_id} - {category} - {audio_id}"
    )

    axes[index].set_ylabel(
        "Mel bins"
    )

    if index == len(samples) - 1:
        axes[index].set_xlabel(
            "Time frames"
        )

    fig.colorbar(
        image,
        ax=axes[index],
        fraction=0.02,
        pad=0.01,
    )


fig.suptitle(
    "DeepShield - Known vs Unseen Voice Conversion",
    fontsize=16,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.98]
)


comparison_file = (
    OUTPUT_DIR
    / "vc_comparison.png"
)

fig.savefig(
    comparison_file,
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Completed
# ============================================================

print()
print("=" * 100)
print("DIAGNOSTIC COMPLETED")
print("=" * 100)
print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Comparison image:"
)

print(
    comparison_file
)

print()