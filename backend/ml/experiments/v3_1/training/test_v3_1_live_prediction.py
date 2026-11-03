from pathlib import Path
import sys

import librosa
import numpy as np
import torch


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

BACKEND_DIR = PROJECT_ROOT / "backend"

V3_DIR = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3"
)

V31_DIR = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3_1"
)

MODEL_FILE = (
    V3_DIR
    / "model"
    / "v3_model.py"
)

CHECKPOINT_PATH = (
    V31_DIR
    / "checkpoints"
    / "deepshield_v3_1_best.pth"
)


# ============================================================
# Import V3 Architecture
# ============================================================

sys.path.insert(
    0,
    str(MODEL_FILE.parent),
)

from v3_model import DeepShieldV3


# ============================================================
# Audio / Feature Configuration
# MUST MATCH PRODUCTION V3 PREPROCESSING
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
# Classes
# ============================================================

CLASS_NAMES = [
    "Genuine",
    "TTS",
    "Voice Conversion",
]

NUM_CLASSES = 3


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Header
# ============================================================

print()
print("=" * 70)
print("DeepShield V3.1 Live Prediction Integration Test")
print("=" * 70)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

print()
print("V3.1 Checkpoint:")
print(CHECKPOINT_PATH)


# ============================================================
# Verify Checkpoint
# ============================================================

if not CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        "V3.1 checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# Create Model
# ============================================================

model = DeepShieldV3(
    num_classes=NUM_CLASSES,
).to(DEVICE)


# ============================================================
# Load V3.1 Checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


# ============================================================
# Evaluation Mode
# ============================================================

model.eval()


print()
print("V3.1 model loaded successfully.")

print(
    "Model parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
    ),
)


# ============================================================
# Audio Loading
# ============================================================

def load_audio(
    file_path: str,
):

    audio, sample_rate = librosa.load(
        str(file_path),
        sr=TARGET_SAMPLE_RATE,
        mono=True,
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
    # Exactly 4 seconds
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
# Feature Extraction
# Same as production predict.py
# ============================================================

def extract_features(
    audio: np.ndarray,
    sample_rate: int,
):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )


    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
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
    # Validate feature shape
    # --------------------------------------------------------

    if (
        mel_db.shape
        != EXPECTED_FEATURE_SHAPE
    ):

        raise RuntimeError(
            "Unexpected feature shape: "
            f"{mel_db.shape}. "
            f"Expected "
            f"{EXPECTED_FEATURE_SHAPE}."
        )


    # --------------------------------------------------------
    # Validate values
    # --------------------------------------------------------

    if not np.isfinite(
        mel_db
    ).all():

        raise RuntimeError(
            "Features contain NaN "
            "or infinite values."
        )


    return mel_db


# ============================================================
# Prediction
# ============================================================

def predict_audio(
    file_path: str,
):

    file_path = str(
        Path(file_path).resolve()
    )


    if not Path(
        file_path
    ).exists():

        raise FileNotFoundError(
            "Audio file not found:\n"
            f"{file_path}"
        )


    print()
    print("-" * 70)
    print("Audio:")
    print(file_path)


    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sample_rate = load_audio(
        file_path
    )


    print(
        "Sample rate:",
        sample_rate,
    )

    print(
        "Samples:",
        len(audio),
    )


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    mel = extract_features(
        audio,
        sample_rate,
    )


    print(
        "Feature shape:",
        mel.shape,
    )


    # --------------------------------------------------------
    # Convert to PyTorch
    #
    # [128, 251]
    #      ↓
    # [1, 1, 128, 251]
    # --------------------------------------------------------

    features = torch.from_numpy(
        mel
    ).float()


    features = features.unsqueeze(
        0
    )

    features = features.unsqueeze(
        0
    )

    features = features.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            features
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predicted_index = int(
            torch.argmax(
                probabilities,
                dim=1,
            ).item()
        )


    # --------------------------------------------------------
    # CPU probabilities
    # --------------------------------------------------------

    probability_values = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )


    if not np.isfinite(
        probability_values
    ).all():

        raise RuntimeError(
            "Model produced invalid "
            "probabilities."
        )


    probability_sum = float(
        probability_values.sum()
    )


    if not np.isclose(
        probability_sum,
        1.0,
        atol=1e-3,
    ):

        raise RuntimeError(
            "Probabilities do not sum "
            f"to 1. Got "
            f"{probability_sum:.6f}"
        )


    # ========================================================
    # Results
    # ========================================================

    prediction = CLASS_NAMES[
        predicted_index
    ]


    confidence = (
        float(
            probability_values[
                predicted_index
            ]
        )
        * 100
    )


    print()
    print("Prediction:", prediction)
    print(
        "Confidence:",
        f"{confidence:.2f}%",
    )


    print()
    print("Class probabilities:")


    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        percentage = (
            float(
                probability_values[
                    index
                ]
            )
            * 100
        )

        print(
            f"  {class_name:<20}"
            f"{percentage:>8.2f}%"
        )


    print()
    print("Model:")
    print(
        CHECKPOINT_PATH.name
    )


    print()
    print("V3.1 LIVE PREDICTION TEST: PASS")
    print("-" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "Enter the FULL path to an audio file."
    )

    audio_path = input(
        "Audio path: "
    ).strip().strip('"')


    predict_audio(
        audio_path
    )