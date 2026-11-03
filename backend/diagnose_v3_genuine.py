from pathlib import Path
import sys

import librosa
import numpy as np
import torch


# ============================================================
# DeepShield V3 Genuine Audio Diagnostic
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
V3_DIR = BACKEND_DIR / "ml" / "experiments" / "v3"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.experiments.v3.model.v3_model import DeepShieldV3


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    V3_DIR
    / "checkpoints"
    / "deepshield_v3_best.pth"
)

AUDIO_DIR = Path(
    r"C:\DeepShield\exam_test_audio"
)

TEST_FILES = [
    "sample-12s.wav",
    "voice-sample-44100hz-16bit.flac",
    "voice-sample.mp3",
    "voice-sample.ogg",
    "Recording_test.wav",
]

TARGET_SAMPLE_RATE = 16000
TARGET_DURATION = 4
TARGET_SAMPLES = TARGET_SAMPLE_RATE * TARGET_DURATION

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

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
# Utility
# ============================================================

def separator(char="-", length=110):
    print(char * length)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


# ============================================================
# Load model
# ============================================================

print()
print("=" * 110)
print("DeepShield V3 - Genuine Audio Diagnostic")
print("=" * 110)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

print("Model:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"V3 model not found:\n{MODEL_PATH}"
    )


model = DeepShieldV3(
    num_classes=3
).to(DEVICE)


checkpoint = torch.load(
    MODEL_PATH,
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
    model.load_state_dict(checkpoint)


model.eval()

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Model loaded successfully.")
print("Parameters:", parameter_count)

print("=" * 110)


# ============================================================
# Audio loading
# ============================================================

def load_and_standardize(file_path):
    """
    Load audio exactly like V3 inference:

        16 kHz
        mono
        exactly 4 seconds
    """

    audio, original_sr = librosa.load(
        str(file_path),
        sr=None,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError(
            "Audio contains no samples."
        )

    original_duration = (
        len(audio) / original_sr
    )

    original_rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    original_peak = float(
        np.max(
            np.abs(audio)
        )
    )

    # --------------------------------------------------------
    # Resample to V3 sample rate
    # --------------------------------------------------------

    if original_sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=TARGET_SAMPLE_RATE,
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
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
        audio,
        original_sr,
        original_duration,
        original_rms,
        original_peak,
    )


# ============================================================
# Feature extraction
# ============================================================

def extract_features(audio):
    """
    Exact V3 feature extraction.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # Raw feature statistics
    raw_mean = float(
        np.mean(mel_db)
    )

    raw_std = float(
        np.std(mel_db)
    )

    raw_min = float(
        np.min(mel_db)
    )

    raw_max = float(
        np.max(mel_db)
    )

    raw_p05 = float(
        np.percentile(
            mel_db,
            5,
        )
    )

    raw_p25 = float(
        np.percentile(
            mel_db,
            25,
        )
    )

    raw_p50 = float(
        np.percentile(
            mel_db,
            50,
        )
    )

    raw_p75 = float(
        np.percentile(
            mel_db,
            75,
        )
    )

    raw_p95 = float(
        np.percentile(
            mel_db,
            95,
        )
    )

    # --------------------------------------------------------
    # V3 per-file normalization
    # --------------------------------------------------------

    mean = np.mean(mel_db)
    std = np.std(mel_db)

    if std > 0:

        mel_normalized = (
            mel_db - mean
        ) / std

    else:

        mel_normalized = (
            mel_db - mean
        )

    mel_normalized = (
        mel_normalized.astype(
            np.float32
        )
    )

    # Normalized statistics
    norm_mean = float(
        np.mean(mel_normalized)
    )

    norm_std = float(
        np.std(mel_normalized)
    )

    norm_min = float(
        np.min(mel_normalized)
    )

    norm_max = float(
        np.max(mel_normalized)
    )

    norm_p05 = float(
        np.percentile(
            mel_normalized,
            5,
        )
    )

    norm_p25 = float(
        np.percentile(
            mel_normalized,
            25,
        )
    )

    norm_p50 = float(
        np.percentile(
            mel_normalized,
            50,
        )
    )

    norm_p75 = float(
        np.percentile(
            mel_normalized,
            75,
        )
    )

    norm_p95 = float(
        np.percentile(
            mel_normalized,
            95,
        )
    )

    return {
        "mel": mel_normalized,

        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_p05": raw_p05,
        "raw_p25": raw_p25,
        "raw_p50": raw_p50,
        "raw_p75": raw_p75,
        "raw_p95": raw_p95,

        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "norm_p05": norm_p05,
        "norm_p25": norm_p25,
        "norm_p50": norm_p50,
        "norm_p75": norm_p75,
        "norm_p95": norm_p95,
    }


# ============================================================
# Audio statistics
# ============================================================

def calculate_audio_statistics(audio):

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    mean = float(
        np.mean(audio)
    )

    std = float(
        np.std(audio)
    )

    abs_audio = np.abs(audio)

    # Percentage of samples below -40 dBFS
    threshold_40db = (
        10 ** (-40 / 20)
    )

    silence_ratio_40db = float(
        np.mean(
            abs_audio
            < threshold_40db
        )
    )

    # Percentage below -50 dBFS
    threshold_50db = (
        10 ** (-50 / 20)
    )

    silence_ratio_50db = float(
        np.mean(
            abs_audio
            < threshold_50db
        )
    )

    return {
        "rms": rms,
        "peak": peak,
        "mean": mean,
        "std": std,
        "silence_40db": silence_ratio_40db,
        "silence_50db": silence_ratio_50db,
    }


# ============================================================
# Prediction
# ============================================================

def predict(mel):

    features = torch.from_numpy(
        mel
    ).float()

    # [128, 251]
    features = features.unsqueeze(0)

    # [1, 128, 251]
    features = features.unsqueeze(0)

    # [1, 1, 128, 251]
    features = features.to(DEVICE)

    with torch.no_grad():

        logits = model(
            features
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    logits_np = (
        logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    probs_np = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    predicted_index = int(
        np.argmax(probs_np)
    )

    return (
        logits_np,
        probs_np,
        predicted_index,
    )


# ============================================================
# Run diagnostics
# ============================================================

all_results = []

for filename in TEST_FILES:

    separator()

    print("FILE:", filename)

    file_path = (
        AUDIO_DIR
        / filename
    )

    if not file_path.exists():

        print("STATUS: FILE NOT FOUND")
        continue

    try:

        (
            audio,
            original_sr,
            original_duration,
            original_rms,
            original_peak,
        ) = load_and_standardize(
            file_path
        )

        print()
        print("[Original Audio]")
        print(
            "Sample Rate        :",
            original_sr,
            "Hz",
        )

        print(
            "Duration           :",
            f"{original_duration:.4f}",
            "sec",
        )

        print(
            "RMS                :",
            fmt(original_rms),
        )

        print(
            "Peak               :",
            fmt(original_peak),
        )

        audio_stats = (
            calculate_audio_statistics(
                audio
            )
        )

        print()
        print("[Standardized Audio]")
        print(
            "Sample Rate        :",
            TARGET_SAMPLE_RATE,
            "Hz",
        )

        print(
            "Samples            :",
            len(audio),
        )

        print(
            "Duration           :",
            f"{len(audio) / TARGET_SAMPLE_RATE:.2f}",
            "sec",
        )

        print(
            "RMS                :",
            fmt(audio_stats["rms"]),
        )

        print(
            "Peak               :",
            fmt(audio_stats["peak"]),
        )

        print(
            "Mean               :",
            fmt(audio_stats["mean"]),
        )

        print(
            "Std                :",
            fmt(audio_stats["std"]),
        )

        print(
            "Silence < -40 dB  :",
            f"{audio_stats['silence_40db'] * 100:.2f}%",
        )

        print(
            "Silence < -50 dB  :",
            f"{audio_stats['silence_50db'] * 100:.2f}%",
        )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        feature_stats = extract_features(
            audio
        )

        mel = feature_stats["mel"]

        print()
        print("[Raw Mel Features - BEFORE normalization]")

        print(
            "Shape              :",
            mel.shape,
        )

        print(
            "Mean               :",
            fmt(feature_stats["raw_mean"]),
        )

        print(
            "Std                :",
            fmt(feature_stats["raw_std"]),
        )

        print(
            "Min                :",
            fmt(feature_stats["raw_min"]),
        )

        print(
            "Max                :",
            fmt(feature_stats["raw_max"]),
        )

        print(
            "P05                :",
            fmt(feature_stats["raw_p05"]),
        )

        print(
            "P25                :",
            fmt(feature_stats["raw_p25"]),
        )

        print(
            "P50                :",
            fmt(feature_stats["raw_p50"]),
        )

        print(
            "P75                :",
            fmt(feature_stats["raw_p75"]),
        )

        print(
            "P95                :",
            fmt(feature_stats["raw_p95"]),
        )

        print()
        print("[Normalized V3 Features]")

        print(
            "Mean               :",
            fmt(feature_stats["norm_mean"]),
        )

        print(
            "Std                :",
            fmt(feature_stats["norm_std"]),
        )

        print(
            "Min                :",
            fmt(feature_stats["norm_min"]),
        )

        print(
            "Max                :",
            fmt(feature_stats["norm_max"]),
        )

        print(
            "P05                :",
            fmt(feature_stats["norm_p05"]),
        )

        print(
            "P25                :",
            fmt(feature_stats["norm_p25"]),
        )

        print(
            "P50                :",
            fmt(feature_stats["norm_p50"]),
        )

        print(
            "P75                :",
            fmt(feature_stats["norm_p75"]),
        )

        print(
            "P95                :",
            fmt(feature_stats["norm_p95"]),
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        logits, probabilities, predicted_index = (
            predict(mel)
        )

        print()
        print("[V3 Model Output]")

        print(
            "Logit - Genuine    :",
            fmt(float(logits[0])),
        )

        print(
            "Logit - TTS        :",
            fmt(float(logits[1])),
        )

        print(
            "Logit - VC         :",
            fmt(float(logits[2])),
        )

        print()
        print(
            "Genuine            :",
            f"{probabilities[0] * 100:.4f}%",
        )

        print(
            "TTS                :",
            f"{probabilities[1] * 100:.4f}%",
        )

        print(
            "Voice Conversion   :",
            f"{probabilities[2] * 100:.4f}%",
        )

        print()
        print(
            "Prediction         :",
            CLASS_NAMES[
                predicted_index
            ],
        )

        print(
            "Confidence         :",
            f"{probabilities[predicted_index] * 100:.4f}%",
        )

        result = {
            "file": filename,
            "prediction": CLASS_NAMES[
                predicted_index
            ],
            "genuine_probability": float(
                probabilities[0]
            ),
            "tts_probability": float(
                probabilities[1]
            ),
            "vc_probability": float(
                probabilities[2]
            ),
            "raw_mel_mean": feature_stats[
                "raw_mean"
            ],
            "raw_mel_std": feature_stats[
                "raw_std"
            ],
            "raw_mel_min": feature_stats[
                "raw_min"
            ],
            "raw_mel_max": feature_stats[
                "raw_max"
            ],
            "rms": audio_stats[
                "rms"
            ],
            "peak": audio_stats[
                "peak"
            ],
            "silence_40db": audio_stats[
                "silence_40db"
            ],
        }

        all_results.append(result)

        print()
        print("STATUS: PASS")

    except Exception as exc:

        print()
        print("STATUS: ERROR")
        print(
            "ERROR:",
            repr(exc),
        )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 110)
print("DIAGNOSTIC SUMMARY")
print("=" * 110)

if not all_results:

    print("No files were successfully analyzed.")

else:

    print()

    print(
        f"{'File':40} "
        f"{'Prediction':18} "
        f"{'Genuine':>10} "
        f"{'TTS':>10} "
        f"{'VC':>10}"
    )

    separator("-", 110)

    for result in all_results:

        print(
            f"{result['file'][:40]:40} "
            f"{result['prediction'][:18]:18} "
            f"{result['genuine_probability'] * 100:9.2f}% "
            f"{result['tts_probability'] * 100:9.2f}% "
            f"{result['vc_probability'] * 100:9.2f}%"
        )

print()
print("=" * 110)
print("DIAGNOSTIC COMPLETE")
print("=" * 110)