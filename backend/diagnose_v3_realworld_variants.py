from __future__ import annotations

import io
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import soundfile as sf


# ============================================================
# PATHS
# ============================================================

BACKEND_ROOT = Path(r"C:\DeepShield\backend")
V3_ROOT = BACKEND_ROOT / "ml" / "experiments" / "v3"

CHECKPOINT_PATH = (
    V3_ROOT / "checkpoints" / "deepshield_v3_best.pth"
)

AUDIO_DIR = Path(r"C:\DeepShield\exam_test_audio")


# ============================================================
# V3 CONFIG
# ============================================================

SAMPLE_RATE = 16000
DURATION = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

CLASS_NAMES = {
    0: "Genuine",
    1: "TTS",
    2: "Voice_Conversion",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL IMPORT
# ============================================================

sys.path.insert(0, str(V3_ROOT))

from model.v3_model import DeepShieldV3


# ============================================================
# MODEL
# ============================================================

print("=" * 80)
print("DeepShield V3 — Real-World Genuine Preprocessing Diagnostic")
print("=" * 80)

print(f"Device     : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU        : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(f"Checkpoint : {CHECKPOINT_PATH}")
print(f"Audio dir  : {AUDIO_DIR}")
print("=" * 80)


print("\nLoading V3 model...")

model = DeepShieldV3(
    num_classes=3,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path: Path):

    """
    First try librosa's normal decoder.

    If that fails, use soundfile as a fallback.

    The purpose here is diagnostic only.
    """

    try:

        audio, sr = librosa.load(
            str(path),
            sr=None,
            mono=True,
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        return audio, int(sr)

    except Exception:

        data, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )

        data = np.asarray(
            data,
            dtype=np.float32,
        )

        if data.ndim == 2:
            data = data.mean(axis=1)

        return data, int(sr)


# ============================================================
# RESAMPLE
# ============================================================

def resample_audio(audio, sr):

    if sr == SAMPLE_RATE:
        return audio

    return librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=SAMPLE_RATE,
    ).astype(np.float32)


# ============================================================
# NORMALIZE VARIANTS
# ============================================================

def peak_normalize(audio):

    peak = np.max(
        np.abs(audio)
    )

    if peak < 1e-8:
        return audio.copy()

    return (
        audio / peak
    ).astype(np.float32)


def rms_normalize(audio, target_rms=0.1):

    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    if rms < 1e-8:
        return audio.copy()

    normalized = (
        audio * (target_rms / rms)
    )

    # Prevent clipping.
    peak = np.max(
        np.abs(normalized)
    )

    if peak > 0.999:

        normalized = (
            normalized / peak * 0.999
        )

    return normalized.astype(np.float32)


def trim_silence(audio):

    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=40,
    )

    return trimmed.astype(np.float32)


def trim_silence_30(audio):

    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=30,
    )

    return trimmed.astype(np.float32)


# ============================================================
# FIXED 4 SECOND WINDOW
# ============================================================

def fixed_four_seconds(audio):

    if len(audio) < NUM_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                NUM_SAMPLES - len(audio),
            ),
            mode="constant",
        )

    else:

        audio = audio[:NUM_SAMPLES]

    return audio.astype(np.float32)


# ============================================================
# MEL FEATURE
# ============================================================

def make_feature(audio):

    audio = fixed_four_seconds(
        audio
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    mean = float(
        mel_db.mean()
    )

    std = float(
        mel_db.std()
    )

    if std < 1e-8:

        raise ValueError(
            "Mel spectrogram std too small"
        )

    feature = (
        mel_db - mean
    ) / std

    feature = np.asarray(
        feature,
        dtype=np.float32,
    )

    if feature.shape != (128, 251):

        raise ValueError(
            f"Unexpected feature shape: "
            f"{feature.shape}"
        )

    return feature, mel_db


# ============================================================
# PREDICTION
# ============================================================

@torch.inference_mode()
def predict(feature):

    x = torch.from_numpy(
        feature
    )

    x = x.unsqueeze(0)
    x = x.unsqueeze(0)

    x = x.to(DEVICE)

    logits = model(x)

    probs = torch.softmax(
        logits,
        dim=1,
    )[0]

    probs = (
        probs
        .detach()
        .cpu()
        .numpy()
    )

    prediction = int(
        np.argmax(probs)
    )

    return prediction, probs


# ============================================================
# AUDIO STATISTICS
# ============================================================

def stats(audio):

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                audio ** 2
            )
        )
    )

    if peak > 0:

        silence_40 = float(
            np.mean(
                np.abs(audio)
                < peak * (10 ** (-40 / 20))
            )
        )

    else:

        silence_40 = 1.0

    return rms, peak, silence_40


# ============================================================
# ONE VARIANT
# ============================================================

def run_variant(
    original_audio,
    name,
    transform,
):

    try:

        audio = transform(
            original_audio.copy()
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        audio = resample_audio(
            audio,
            ORIGINAL_SR,
        )

        feature, mel_db = make_feature(
            audio
        )

        prediction, probs = predict(
            feature
        )

        rms, peak, silence = stats(
            fixed_four_seconds(audio)
        )

        print(
            f"  {name:24s} | "
            f"{CLASS_NAMES[prediction]:16s} | "
            f"G={probs[0] * 100:6.2f}% "
            f"TTS={probs[1] * 100:6.2f}% "
            f"VC={probs[2] * 100:6.2f}% | "
            f"RMS={rms:.4f} "
            f"Peak={peak:.4f} "
            f"Sil40={silence * 100:5.1f}%"
        )

        return {
            "name": name,
            "prediction": prediction,
            "genuine": float(probs[0]),
            "tts": float(probs[1]),
            "vc": float(probs[2]),
        }

    except Exception as e:

        print(
            f"  {name:24s} | ERROR | {e}"
        )

        return None


# ============================================================
# FILE LIST
# ============================================================

AUDIO_FILES = [
    "sample-12s.wav",
    "voice-sample-44100hz-16bit.flac",
    "voice-sample.mp3",
    "voice-sample.ogg",
    "Recording_test.wav",
]


# ============================================================
# MAIN
# ============================================================

all_results = []


for filename in AUDIO_FILES:

    path = AUDIO_DIR / filename

    if not path.exists():

        print(
            f"\nSKIPPING missing file: {filename}"
        )

        continue

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    try:

        original_audio, ORIGINAL_SR = load_audio(
            path
        )

        print(
            f"Original sample rate : "
            f"{ORIGINAL_SR} Hz"
        )

        print(
            f"Original duration    : "
            f"{len(original_audio) / ORIGINAL_SR:.2f} sec"
        )

        original_rms, original_peak, original_silence = stats(
            original_audio
        )

        print(
            f"Original RMS         : "
            f"{original_rms:.6f}"
        )

        print(
            f"Original peak        : "
            f"{original_peak:.6f}"
        )

        print(
            f"Original silence <40dB: "
            f"{original_silence * 100:.2f}%"
        )

    except Exception as e:

        print(
            f"ERROR loading {filename}: {e}"
        )

        continue

    print("\nVariants:")

    variants = [

        (
            "Original",
            lambda x: x,
        ),

        (
            "Peak normalized",
            peak_normalize,
        ),

        (
            "RMS normalized",
            rms_normalize,
        ),

        (
            "Trim silence 40dB",
            trim_silence,
        ),

        (
            "Trim silence 30dB",
            trim_silence_30,
        ),

        (
            "Trim + peak norm",
            lambda x: peak_normalize(
                trim_silence(x)
            ),
        ),

        (
            "Trim + RMS norm",
            lambda x: rms_normalize(
                trim_silence(x)
            ),
        ),
    ]

    for name, transform in variants:

        result = run_variant(
            original_audio,
            name,
            transform,
        )

        if result is not None:

            result["file"] = filename

            all_results.append(
                result
            )


# ============================================================
# FINAL INTERPRETATION DATA
# ============================================================

print("\n" + "=" * 80)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 80)

for filename in AUDIO_FILES:

    results = [
        r
        for r in all_results
        if r["file"] == filename
    ]

    if not results:
        continue

    print(f"\n{filename}")

    for r in results:

        print(
            f"  {r['name']:24s} -> "
            f"{CLASS_NAMES[r['prediction']]:16s} "
            f"(G={r['genuine'] * 100:.2f}% "
            f"TTS={r['tts'] * 100:.2f}% "
            f"VC={r['vc'] * 100:.2f}%)"
        )

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)