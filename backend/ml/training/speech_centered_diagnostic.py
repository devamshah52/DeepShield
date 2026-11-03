import sys
from pathlib import Path

import subprocess
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000
TARGET_SAMPLES = SAMPLE_RATE * 4

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

OUTPUT_DIR = (
    BACKEND_DIR
    / "ml"
    / "evaluation"
    / "speech_centered_diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load ORIGINAL audio
#
# IMPORTANT:
# This deliberately does NOT use audio_processor.load_audio()
# because that function already forces audio to 4 seconds.
# ============================================================

def load_original_audio(file_path):

    file_path = str(
        Path(file_path).resolve()
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        file_path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    audio = np.frombuffer(
        result.stdout,
        dtype=np.float32,
    ).copy()

    if audio.size == 0:
        raise ValueError(
            f"No readable audio data: {file_path}"
        )

    return audio


# ============================================================
# Mel extraction
# ============================================================

def extract_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    mean = mel_db.mean()
    std = mel_db.std()

    if std > 0:

        mel_db = (
            mel_db - mean
        ) / std

    else:

        mel_db = (
            mel_db - mean
        )

    return mel_db.astype(
        np.float32
    )


# ============================================================
# Current DeepShield preprocessing
#
# First 4 seconds + right padding
# ============================================================

def current_crop(audio):

    if len(audio) >= TARGET_SAMPLES:

        return audio[
            :TARGET_SAMPLES
        ]

    padding = (
        TARGET_SAMPLES
        - len(audio)
    )

    return np.pad(
        audio,
        (0, padding),
        mode="constant",
    )


# ============================================================
# Speech-centered preprocessing
# ============================================================

def speech_centered_crop(audio):

    # --------------------------------------------------------
    # Audio shorter than 4 seconds
    #
    # Center the original recording inside the 4-second
    # window instead of padding only on the right.
    # --------------------------------------------------------

    if len(audio) <= TARGET_SAMPLES:

        padding_total = (
            TARGET_SAMPLES
            - len(audio)
        )

        left_padding = (
            padding_total // 2
        )

        right_padding = (
            padding_total
            - left_padding
        )

        return np.pad(
            audio,
            (
                left_padding,
                right_padding,
            ),
            mode="constant",
        )

    # --------------------------------------------------------
    # RMS energy
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=256,
    )[0]

    # --------------------------------------------------------
    # Speech activity threshold
    # --------------------------------------------------------

    threshold = max(
        np.percentile(
            rms,
            30,
        ),
        rms.max() * 0.10,
    )

    active = np.where(
        rms >= threshold
    )[0]

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if len(active) == 0:

        start = max(
            0,
            (
                len(audio)
                - TARGET_SAMPLES
            ) // 2,
        )

        return audio[
            start:start + TARGET_SAMPLES
        ]

    # --------------------------------------------------------
    # Active speech boundaries
    # --------------------------------------------------------

    first_frame = int(
        active[0]
    )

    last_frame = int(
        active[-1]
    )

    speech_start = (
        first_frame
        * HOP_LENGTH
    )

    speech_end = min(
        len(audio),
        last_frame
        * HOP_LENGTH
        + 1024,
    )

    speech_center = (
        speech_start
        + speech_end
    ) // 2

    # --------------------------------------------------------
    # Center 4-second window
    # --------------------------------------------------------

    start = (
        speech_center
        - TARGET_SAMPLES // 2
    )

    start = max(
        0,
        min(
            start,
            len(audio)
            - TARGET_SAMPLES,
        ),
    )

    return audio[
        start:start + TARGET_SAMPLES
    ]


# ============================================================
# Select representative samples
# ============================================================

train = pd.read_csv(
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "train_manifest.csv"
)

evaluation = pd.read_csv(
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "eval_manifest.csv"
)


samples = []


# Genuine
row = train[
    train["attack_id"] == "-"
].iloc[0]

samples.append(
    (
        "Genuine",
        row["audio_id"],
        row["audio_path"],
    )
)


# Known VC
for attack_id in [
    "A05",
    "A06",
]:

    row = train[
        train["attack_id"] == attack_id
    ].iloc[0]

    samples.append(
        (
            attack_id,
            row["audio_id"],
            row["audio_path"],
        )
    )


# Unseen VC
for attack_id in [
    "A17",
    "A18",
    "A19",
]:

    row = evaluation[
        evaluation["attack_id"] == attack_id
    ].iloc[0]

    samples.append(
        (
            attack_id,
            row["audio_id"],
            row["audio_path"],
        )
    )


# ============================================================
# Process samples
# ============================================================

print()
print("=" * 100)
print(
    "DeepShield CORRECTED Speech-Centered Diagnostic"
)
print("=" * 100)
print()

results = []


for (
    attack_id,
    audio_id,
    audio_path,
) in samples:

    print(
        f"Processing {attack_id}: {audio_id}"
    )

    # --------------------------------------------------------
    # ORIGINAL duration
    # --------------------------------------------------------

    audio = load_original_audio(
        audio_path
    )

    original_duration = (
        len(audio)
        / SAMPLE_RATE
    )

    # --------------------------------------------------------
    # Current preprocessing
    # --------------------------------------------------------

    current = current_crop(
        audio
    )

    # --------------------------------------------------------
    # Speech-centered preprocessing
    # --------------------------------------------------------

    centered = speech_centered_crop(
        audio
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    current_mel = extract_mel(
        current
    )

    centered_mel = extract_mel(
        centered
    )

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    current_rms = float(
        np.sqrt(
            np.mean(
                current ** 2
            )
        )
    )

    centered_rms = float(
        np.sqrt(
            np.mean(
                centered ** 2
            )
        )
    )

    # --------------------------------------------------------
    # Non-zero occupancy
    # --------------------------------------------------------

    current_nonzero = (
        np.mean(
            np.abs(current)
            > 1e-5
        )
        * 100
    )

    centered_nonzero = (
        np.mean(
            np.abs(centered)
            > 1e-5
        )
        * 100
    )

    # --------------------------------------------------------
    # Record results
    # --------------------------------------------------------

    results.append(
        {
            "attack_id": attack_id,
            "audio_id": audio_id,
            "original_duration_sec":
                round(
                    original_duration,
                    3,
                ),
            "current_rms":
                round(
                    current_rms,
                    6,
                ),
            "centered_rms":
                round(
                    centered_rms,
                    6,
                ),
            "current_nonzero_percent":
                round(
                    current_nonzero,
                    2,
                ),
            "centered_nonzero_percent":
                round(
                    centered_nonzero,
                    2,
                ),
        }
    )

    # --------------------------------------------------------
    # Individual comparison
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
    )

    axes[0].imshow(
        current_mel,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    axes[0].set_title(
        f"{attack_id} - CURRENT "
        f"(first 4 seconds)"
    )

    axes[0].set_ylabel(
        "Mel bins"
    )

    axes[1].imshow(
        centered_mel,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    axes[1].set_title(
        f"{attack_id} - SPEECH-CENTERED"
    )

    axes[1].set_xlabel(
        "Time frames"
    )

    axes[1].set_ylabel(
        "Mel bins"
    )

    fig.suptitle(
        f"{attack_id} - {audio_id} | "
        f"Original duration: "
        f"{original_duration:.2f}s"
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{attack_id}_{audio_id}_comparison.png"
    )

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Save numerical results
# ============================================================

results_df = pd.DataFrame(
    results
)

csv_file = (
    OUTPUT_DIR
    / "corrected_speech_centered_comparison.csv"
)

results_df.to_csv(
    csv_file,
    index=False,
)


# ============================================================
# Combined comparison
# ============================================================

fig, axes = plt.subplots(
    len(samples),
    2,
    figsize=(16, 22),
)


for row_index, (
    attack_id,
    audio_id,
    audio_path,
) in enumerate(samples):

    audio = load_original_audio(
        audio_path
    )

    current = current_crop(
        audio
    )

    centered = speech_centered_crop(
        audio
    )

    current_mel = extract_mel(
        current
    )

    centered_mel = extract_mel(
        centered
    )

    axes[
        row_index,
        0
    ].imshow(
        current_mel,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    axes[
        row_index,
        0
    ].set_title(
        f"{attack_id} - Current"
    )

    axes[
        row_index,
        1
    ].imshow(
        centered_mel,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    axes[
        row_index,
        1
    ].set_title(
        f"{attack_id} - Speech-centered"
    )

    axes[
        row_index,
        0
    ].set_ylabel(
        "Mel bins"
    )


axes[
    -1,
    0
].set_xlabel(
    "Time frames"
)

axes[
    -1,
    1
].set_xlabel(
    "Time frames"
)

fig.suptitle(
    "DeepShield - Corrected Current vs Speech-Centered Preprocessing",
    fontsize=16,
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.98,
    ]
)

comparison_file = (
    OUTPUT_DIR
    / "corrected_speech_centered_comparison.png"
)

fig.savefig(
    comparison_file,
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 100)
print("ORIGINAL AUDIO / PREPROCESSING COMPARISON")
print("=" * 100)
print()

print(
    results_df.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("DIAGNOSTIC COMPLETED")
print("=" * 100)
print()

print(
    "Comparison image:"
)

print(
    comparison_file
)

print()
print(
    "CSV:"
)

print(
    csv_file
)