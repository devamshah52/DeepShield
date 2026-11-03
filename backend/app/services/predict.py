from pathlib import Path
import time

import librosa
import numpy as np
import torch

from ml.experiments.v3.model.v3_model import (
    DeepShieldV3,
)


# ============================================================
# Configuration
# ============================================================

BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

V31_DIR = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3_1"
)

MODEL_PATH = (
    V31_DIR
    / "checkpoints"
    / "deepshield_v3_2_weighted_best.pth"
)


# ============================================================
# V3.2 Audio / Feature Configuration
#
# MUST MATCH V3 / V3.1 / V3.2 TRAINING PREPROCESSING EXACTLY
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
# Prediction Service Startup
# ============================================================

print()
print("=" * 70)
print("DeepShield V3.2 Weighted Prediction Service")
print("=" * 70)

print(
    "Device:",
    DEVICE,
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

print(
    "V3.2 Weighted Model:",
    MODEL_PATH,
)


# ============================================================
# Verify Model Exists
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"DeepShield V3.2 weighted model not found:\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# Create V3.2 Model
# ============================================================

model = DeepShieldV3(
    num_classes=NUM_CLASSES,
).to(DEVICE)


# ============================================================
# Load V3.2 Weighted Checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)


# ------------------------------------------------------------
# Support checkpoint dictionary
# ------------------------------------------------------------

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


print(
    "V3.2 weighted model loaded successfully."
)

print(
    "Model parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
    ),
)

print("=" * 70)


# ============================================================
# Audio Loading
# ============================================================

def load_audio(
    file_path: str,
):
    """
    Load uploaded audio and standardize it to:

        Sample rate : 16 kHz
        Channels    : Mono
        Duration    : Exactly 4 seconds

    This matches the V3 / V3.1 / V3.2 training
    preprocessing.
    """

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
# V3.2 Mel Spectrogram Extraction
# ============================================================

def extract_v3_2_features(
    audio: np.ndarray,
    sample_rate: int,
):
    """
    Convert standardized audio into the exact
    V3.2 Mel-spectrogram representation.

    Output:
        float32
        shape = (128, 251)
    """

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    # --------------------------------------------------------
    # Power -> dB
    # --------------------------------------------------------

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # --------------------------------------------------------
    # Per-file normalization
    #
    # Same as V3 / V3.1 / V3.2 training.
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
    # Validate shape
    # --------------------------------------------------------

    if (
        mel_db.shape
        != EXPECTED_FEATURE_SHAPE
    ):

        raise RuntimeError(
            "Unexpected V3.2 Mel feature shape: "
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
            "V3.2 Mel features contain "
            "NaN or infinite values."
        )

    return mel_db


# ============================================================
# CUDA Timing Helper
# ============================================================

def synchronize_cuda():
    """
    Synchronize CUDA so GPU timing is accurate.
    """

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# Prediction
# ============================================================

def predict_audio(
    file_path: str,
):

    total_start = time.perf_counter()

    file_path = str(
        Path(file_path).resolve()
    )

    # --------------------------------------------------------
    # Validate uploaded file
    # --------------------------------------------------------

    if not Path(
        file_path
    ).exists():

        raise FileNotFoundError(
            f"Audio file not found:\n"
            f"{file_path}"
        )

    # --------------------------------------------------------
    # Audio loading timing
    # --------------------------------------------------------

    audio_start = time.perf_counter()

    audio, sample_rate = load_audio(
        file_path
    )

    audio_time = (
        time.perf_counter()
        - audio_start
    )

    # --------------------------------------------------------
    # Feature extraction timing
    # --------------------------------------------------------

    feature_start = time.perf_counter()

    mel = extract_v3_2_features(
        audio,
        sample_rate,
    )

    feature_time = (
        time.perf_counter()
        - feature_start
    )

    # --------------------------------------------------------
    # NumPy -> PyTorch
    #
    # Mel:
    #     [128, 251]
    #
    # V3.2 model input:
    #     [batch, channel, mel, time]
    #
    # Result:
    #     [1, 1, 128, 251]
    # --------------------------------------------------------

    tensor_start = time.perf_counter()

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

    # Synchronize because .to(cuda) may be asynchronous.
    synchronize_cuda()

    tensor_time = (
        time.perf_counter()
        - tensor_start
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    synchronize_cuda()

    inference_start = time.perf_counter()

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

    synchronize_cuda()

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # --------------------------------------------------------
    # Convert probabilities to CPU
    # --------------------------------------------------------

    result_start = time.perf_counter()

    probability_values = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Validate probabilities
    # --------------------------------------------------------

    if not np.isfinite(
        probability_values
    ).all():

        raise RuntimeError(
            "Model produced invalid "
            "probability values."
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
            "Model probabilities do not "
            f"sum to 1. Got "
            f"{probability_sum:.6f}."
        )

    # --------------------------------------------------------
    # Probability dictionary
    # --------------------------------------------------------

    probability_dict = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probability_dict[
            class_name
        ] = round(
            float(
                probability_values[
                    index
                ]
            ) * 100,
            2,
        )

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    prediction = CLASS_NAMES[
        predicted_index
    ]

    confidence = round(
        float(
            probability_values[
                predicted_index
            ]
        ) * 100,
        2,
    )

    result_time = (
        time.perf_counter()
        - result_start
    )

    # --------------------------------------------------------
    # Total timing
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # Performance log
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DeepShield Prediction Timing")
    print("=" * 70)

    print(
        f"Audio loading       : {audio_time:.4f} sec"
    )

    print(
        f"Feature extraction  : {feature_time:.4f} sec"
    )

    print(
        f"Tensor preparation  : {tensor_time:.4f} sec"
    )

    print(
        f"Model inference     : {inference_time:.4f} sec"
    )

    print(
        f"Result processing   : {result_time:.4f} sec"
    )

    print("-" * 70)

    print(
        f"TOTAL               : {total_time:.4f} sec"
    )

    print("-" * 70)

    print(
        f"Prediction          : {prediction}"
    )

    print(
        f"Confidence          : {confidence:.2f}%"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "prediction": prediction,

        "confidence": confidence,

        "probabilities": probability_dict,

        "feature_shape": list(
            mel.shape
        ),

        "device": str(
            DEVICE
        ),

        "model": MODEL_PATH.name,
    }