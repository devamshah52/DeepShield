import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# DeepShield imports
# ============================================================

from ml.training.model import DeepShieldCNN

from ml.training.audio_processor import (
    extract_mel_spectrogram,
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    BACKEND_DIR
    / "ml"
    / "models"
    / "deepshield_cnn_best.pth"
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
# Load model
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )


    model = DeepShieldCNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )


    # --------------------------------------------------------
    # Support both:
    #
    # 1. Plain state_dict
    # 2. Checkpoint containing model_state_dict
    # --------------------------------------------------------

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


    model.eval()


    return model


# ============================================================
# Predict audio
# ============================================================

def predict_audio(
    audio_path,
    model=None,
):

    audio_path = Path(
        audio_path
    ).resolve()


    # --------------------------------------------------------
    # Validate audio file
    # --------------------------------------------------------

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n"
            f"{audio_path}"
        )


    if not audio_path.is_file():

        raise ValueError(
            f"Audio path is not a file:\n"
            f"{audio_path}"
        )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if model is None:

        model = load_model()


    # --------------------------------------------------------
    # Extract standardized features
    #
    # IMPORTANT:
    #
    # This uses the SAME preprocessing as the website/API:
    #
    # ml/training/audio_processor.py
    #
    # Therefore:
    #
    # FFmpeg
    # 16 kHz
    # Mono
    # 4 seconds
    # n_fft = 1024
    # hop_length = 256
    # n_mels = 128
    # dB conversion
    # Standardization
    #
    # --------------------------------------------------------

    mel_spectrogram = (
        extract_mel_spectrogram(
            str(audio_path)
        )
    )


    # --------------------------------------------------------
    # Convert Mel spectrogram to tensor
    #
    # Current shape:
    #
    # [128, time]
    #
    # Model expects:
    #
    # [batch, channel, mel_bins, time]
    #
    # Therefore:
    #
    # [128, time]
    #      ↓
    # [1, 128, time]
    #      ↓
    # [1, 1, 128, time]
    #
    # --------------------------------------------------------

    features = torch.from_numpy(
        mel_spectrogram
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

        logits = model(
            features
        )


        probabilities = F.softmax(
            logits,
            dim=1,
        )


        predicted_class = (
            torch.argmax(
                probabilities,
                dim=1,
            )
            .item()
        )


    # --------------------------------------------------------
    # Convert probabilities
    # --------------------------------------------------------

    probability_values = (
        probabilities[
            0
        ]
        .detach()
        .cpu()
        .numpy()
    )


    predicted_label = (
        CLASS_NAMES[
            predicted_class
        ]
    )


    confidence = float(
        probability_values[
            predicted_class
        ]
    )


    class_probabilities = {

        CLASS_NAMES[index]:
            float(
                probability_values[
                    index
                ]
            )

        for index in range(
            NUM_CLASSES
        )

    }


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "prediction":
            predicted_label,

        "confidence":
            confidence,

        "probabilities":
            class_probabilities,

        "device":
            str(DEVICE),

        "sample_rate":
            16000,

        "duration":
            4.0,

        "feature_shape":
            list(
                mel_spectrogram.shape
            ),

    }


# ============================================================
# Test from command line
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 70)

    print(
        "DeepShield Inference Test"
    )

    print("=" * 70)


    print()

    print(
        "Device:",
        DEVICE,
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # --------------------------------------------------------
    # Require audio path
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()

        print(
            "Usage:"
        )

        print(
            "python .\\ml\\inference.py "
            "path\\to\\audio.wav"
        )

        sys.exit(1)


    audio_path = sys.argv[1]


    print()

    print(
        "Audio:",
        audio_path,
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()

    print(
        "Loading model..."
    )


    model = load_model()


    print(
        "Model loaded successfully."
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print()

    print(
        "Running inference..."
    )


    result = predict_audio(
        audio_path,
        model=model,
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "Prediction Result"
    )

    print("=" * 70)


    print()

    print(
        "Prediction:",
        result["prediction"]
    )


    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )


    print()

    print(
        "Class probabilities:"
    )


    for (
        label,
        probability
    ) in result[
        "probabilities"
    ].items():

        print(
            f"  {label}: "
            f"{probability * 100:.2f}%"
        )


    print()

    print(
        "Feature shape:",
        result[
            "feature_shape"
        ]
    )


    print(
        "Device:",
        result["device"]
    )


    print()

    print(
        "Sample rate:",
        result["sample_rate"]
    )


    print(
        "Duration:",
        f"{result['duration']:.1f} seconds"
    )


    print()

    print("=" * 70)