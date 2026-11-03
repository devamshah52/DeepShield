from pathlib import Path
import sys

import librosa
import numpy as np
import torch


# ============================================================
# Add DeepShield backend to Python import path
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(
    0,
    str(BACKEND_DIR),
)


from ml.experiments.v3.model.v3_model import DeepShieldV3


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

V32_WEIGHTED_CHECKPOINT = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "experiments"
    / "v3_1"
    / "checkpoints"
    / "deepshield_v3_2_weighted_best.pth"
)

AUDIO_DIR = (
    PROJECT_ROOT
    / "v3_1_dataset_inspection"
    / "gtts_blind_100"
)

TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLES = 64000

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

EXPECTED_FEATURE_SHAPE = (128, 251)

CLASS_NAMES = [
    "Genuine",
    "TTS",
    "Voice Conversion",
]


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

print("=" * 70)
print("DeepShield V3.2 Weighted - 100 Sample gTTS Blind Evaluation")
print("=" * 70)

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
print("Checkpoint:")
print(V32_WEIGHTED_CHECKPOINT)

print()


if not V32_WEIGHTED_CHECKPOINT.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{V32_WEIGHTED_CHECKPOINT}"
    )


model = DeepShieldV3(
    num_classes=3
).to(DEVICE)


checkpoint = torch.load(
    V32_WEIGHTED_CHECKPOINT,
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


model.eval()

print(
    "V3.2 weighted model loaded successfully."
)

print(
    "Model parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    ),
)

print("-" * 70)


# ============================================================
# Feature extraction
# ============================================================

def extract_features(
    file_path: Path,
):

    audio, _ = librosa.load(
        str(file_path),
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if len(audio) == 0:

        raise ValueError(
            "Audio contains no samples."
        )

    # Exactly 4 seconds
    if len(audio) > TARGET_SAMPLES:

        audio = audio[
            :TARGET_SAMPLES
        ]

    elif len(audio) < TARGET_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                TARGET_SAMPLES - len(audio),
            ),
            mode="constant",
        )

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    # Power -> dB
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # Same normalization as V3.1 / V3.2
    mean = np.mean(mel_db)
    std = np.std(mel_db)

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

    if (
        mel_db.shape
        != EXPECTED_FEATURE_SHAPE
    ):

        raise RuntimeError(
            f"Unexpected feature shape: "
            f"{mel_db.shape}"
        )

    if not np.isfinite(
        mel_db
    ).all():

        raise RuntimeError(
            "Features contain NaN or infinite values."
        )

    return mel_db


# ============================================================
# Find blind-test files
# ============================================================

audio_files = sorted(
    AUDIO_DIR.glob("*.mp3")
)

print()
print(
    "Independent gTTS blind files:",
    len(audio_files),
)

if len(audio_files) != 100:

    raise RuntimeError(
        f"Expected exactly 100 blind-test files, "
        f"but found {len(audio_files)}."
    )

print("-" * 70)


# ============================================================
# Evaluate
# ============================================================

tts_correct = 0
genuine_predictions = 0
vc_predictions = 0

results = []


for index, audio_file in enumerate(
    audio_files,
    start=1,
):

    try:

        mel = extract_features(
            audio_file
        )

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

        probability_values = (
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
        )

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

        genuine_probability = (
            float(
                probability_values[0]
            )
            * 100
        )

        tts_probability = (
            float(
                probability_values[1]
            )
            * 100
        )

        vc_probability = (
            float(
                probability_values[2]
            )
            * 100
        )

        if prediction == "TTS":

            tts_correct += 1

        elif prediction == "Genuine":

            genuine_predictions += 1

        elif prediction == "Voice Conversion":

            vc_predictions += 1

        results.append(
            {
                "file": audio_file.name,
                "prediction": prediction,
                "confidence": confidence,
                "genuine": genuine_probability,
                "tts": tts_probability,
                "voice_conversion": vc_probability,
            }
        )

        print(
            f"{index:03d}. "
            f"{audio_file.name:<18} "
            f"{prediction:<18} "
            f"{confidence:6.2f}%"
        )

    except Exception as error:

        print(
            f"{index:03d}. "
            f"{audio_file.name:<18} "
            f"ERROR: {error}"
        )


# ============================================================
# Summary
# ============================================================

total = len(results)

accuracy = (
    (tts_correct / total) * 100
    if total > 0
    else 0
)

print()
print("=" * 70)
print("V3.2 Weighted - 100 Sample gTTS Blind Evaluation Summary")
print("=" * 70)

print(
    f"Total gTTS samples     : {total}"
)

print(
    f"Predicted as TTS       : {tts_correct}"
)

print(
    f"Predicted as Genuine   : {genuine_predictions}"
)

print(
    f"Predicted as VC        : {vc_predictions}"
)

print(
    f"gTTS detection rate    : {accuracy:.2f}%"
)

print("=" * 70)

print()
print("Expected class for every sample: TTS")

if total > 0 and tts_correct == total:

    print(
        "STATUS: ALL 100 gTTS SAMPLES DETECTED"
    )

elif total > 0:

    print(
        "STATUS: SOME gTTS SAMPLES WERE MISCLASSIFIED"
    )

else:

    print(
        "STATUS: NO TEST FILES FOUND"
    )