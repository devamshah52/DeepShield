import sys
from pathlib import Path

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


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

TARGET_SAMPLE_RATE = 16000

N_FFT = 1024

HOP_LENGTH = 256

N_MELS = 128


# ============================================================
# Mel Spectrogram Extraction
# ============================================================

def extract_mel_spectrogram(file_path):

    audio, sample_rate = load_audio(
        file_path,
        sample_rate=TARGET_SAMPLE_RATE,
    )

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max,
    )

    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-8
    )

    return mel_db.astype(
        np.float32
    )


# ============================================================
# Test + Visualization
# ============================================================

if __name__ == "__main__":

    test_file = (
        BACKEND_DIR
        / "ml"
        / "dataset"
        / "LA"
        / "ASVspoof2019_LA_train"
        / "flac"
        / "LA_T_1138215.flac"
    )

    features = extract_mel_spectrogram(
        test_file
    )

    print()
    print("=" * 60)
    print("Mel Spectrogram Test")
    print("=" * 60)

    print(
        "Shape:",
        features.shape
    )

    print(
        "Data type:",
        features.dtype
    )

    print(
        "Minimum:",
        features.min()
    )

    print(
        "Maximum:",
        features.max()
    )

    print(
        "Mean:",
        features.mean()
    )

    print(
        "Standard deviation:",
        features.std()
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Visualize
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    librosa.display.specshow(
        features,
        sr=TARGET_SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
    )

    plt.colorbar(
        format="%+2.0f"
    )

    plt.title(
        "DeepShield Mel Spectrogram"
    )

    plt.tight_layout()

    output_file = (
        BACKEND_DIR
        / "ml"
        / "features"
        / "mel_test.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print()
    print(
        "Spectrogram saved to:"
    )

    print(output_file)