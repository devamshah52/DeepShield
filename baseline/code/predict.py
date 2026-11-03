from pathlib import Path

import torch

from ml.training.audio_processor import (
    extract_mel_spectrogram,
)

from ml.training.model_pool import (
    DeepShieldCNNPool,
)


# ============================================================
# Configuration
# ============================================================

BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_PATH = (
    BACKEND_DIR
    / "ml"
    / "models"
    / "deepshield_cnn_pool_augmented_best.pth"
)

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
# Load model once
# ============================================================

print()
print("=" * 70)
print("DeepShield Prediction Service")
print("=" * 70)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Model:",
    MODEL_PATH
)


# ============================================================
# Verify model exists
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"DeepShield model not found:\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# Create dual-pooling model
# ============================================================

model = DeepShieldCNNPool(
    num_classes=NUM_CLASSES
).to(DEVICE)


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
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
# Evaluation mode
# ============================================================

model.eval()


print(
    "Model loaded successfully."
)

print("=" * 70)


# ============================================================
# Prediction
# ============================================================

def predict_audio(
    file_path: str,
):

    file_path = str(
        Path(file_path).resolve()
    )

    # --------------------------------------------------------
    # Extract standardized Mel spectrogram
    # --------------------------------------------------------

    mel = extract_mel_spectrogram(
        file_path
    )

    # --------------------------------------------------------
    # Convert NumPy -> PyTorch
    #
    # Mel:
    # [128, 251]
    #
    # Model input:
    # [batch, channel, mel, time]
    #
    # Result:
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
    # Inference
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
    # Convert probabilities
    # --------------------------------------------------------

    probability_values = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    probability_dict = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probability_dict[class_name] = round(
            float(
                probability_values[index]
            ) * 100,
            2,
        )

    # --------------------------------------------------------
    # Prediction
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