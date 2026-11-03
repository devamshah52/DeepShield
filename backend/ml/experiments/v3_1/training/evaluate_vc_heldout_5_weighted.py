from pathlib import Path
import sys

import librosa
import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "experiments"
    / "v3_1"
    / "checkpoints"
    / "deepshield_v3_2_weighted_best.pth"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "experiments"
    / "v3"
    / "model"
)

VC_DIR = PROJECT_ROOT / "exam_demo_audio" / "Voice Conversion"

# ============================================================
# IMPORT V3 MODEL
# ============================================================

sys.path.insert(0, str(MODEL_DIR))

from v3_model import DeepShieldV3


# ============================================================
# CONFIG
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

EXPECTED_SHAPE = (128, 251)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HELPER: LOAD AUDIO
# ============================================================

def load_audio(audio_path):
    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if len(audio) < NUM_SAMPLES:
        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio)),
            mode="constant",
        )
    else:
        audio = audio[:NUM_SAMPLES]

    return audio.astype(np.float32)


# ============================================================
# HELPER: CREATE FEATURE
# ============================================================

def extract_feature(audio_path):
    audio = load_audio(audio_path)

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

    if std < 1e-8:
        raise ValueError(
            f"Feature standard deviation too small: {audio_path}"
        )

    feature = (mel_db - mean) / std
    feature = feature.astype(np.float32)

    if feature.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected feature shape {feature.shape} "
            f"for {audio_path}; expected {EXPECTED_SHAPE}"
        )

    return feature


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{MODEL_FILE}"
        )

    model = DeepShieldV3(
        num_classes=3
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("DEEPSHIELD V3.2 WEIGHTED")
    print("HELD-OUT VOICE CONVERSION EVALUATION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nCheckpoint:")
    print(MODEL_FILE)

    print(f"\nVC directory:")
    print(VC_DIR)

    if not VC_DIR.exists():
        raise FileNotFoundError(
            f"VC directory not found:\n{VC_DIR}"
        )

    # --------------------------------------------------------
    # Find VC files
    # --------------------------------------------------------

    audio_files = sorted(
        VC_DIR.glob("*.flac")
    )

    print(f"\nFound FLAC files: {len(audio_files)}")

    if len(audio_files) == 0:
        raise RuntimeError(
            "No FLAC files found."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    predictions = []
    confidences = []

    print("\n" + "-" * 70)
    print("PER-FILE RESULTS")
    print("-" * 70)

    with torch.no_grad():

        for audio_path in audio_files:

            try:
                feature = extract_feature(audio_path)

                tensor = torch.from_numpy(feature)
                tensor = tensor.unsqueeze(0).unsqueeze(0)
                tensor = tensor.to(DEVICE)

                logits = model(tensor)

                probabilities = F.softmax(
                    logits,
                    dim=1,
                )

                confidence, prediction = torch.max(
                    probabilities,
                    dim=1,
                )

                prediction = int(prediction.item())
                confidence = float(confidence.item())

                predictions.append(prediction)
                confidences.append(confidence)

                predicted_name = CLASS_NAMES[prediction]

                print(
                    f"{audio_path.name:<28} "
                    f"-> {predicted_name:<16} "
                    f"{confidence * 100:>7.2f}%"
                )

            except Exception as exc:

                print(
                    f"{audio_path.name:<28} "
                    f"-> ERROR: {exc}"
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(predictions)

    vc_predictions = sum(
        1 for p in predictions
        if p == 2
    )

    genuine_predictions = sum(
        1 for p in predictions
        if p == 0
    )

    tts_predictions = sum(
        1 for p in predictions
        if p == 1
    )

    detection_rate = (
        vc_predictions / total * 100
        if total > 0
        else 0.0
    )

    mean_confidence = (
        np.mean(confidences) * 100
        if confidences
        else 0.0
    )

    median_confidence = (
        np.median(confidences) * 100
        if confidences
        else 0.0
    )

    print("\n" + "=" * 70)
    print("HELD-OUT VC EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal files evaluated : {total}")

    print(
        f"Voice Conversion      : "
        f"{vc_predictions} "
        f"({vc_predictions / total * 100:.2f}%)"
        if total > 0
        else "Voice Conversion      : 0"
    )

    print(
        f"TTS                   : "
        f"{tts_predictions} "
        f"({tts_predictions / total * 100:.2f}%)"
        if total > 0
        else "TTS                   : 0"
    )

    print(
        f"Genuine               : "
        f"{genuine_predictions} "
        f"({genuine_predictions / total * 100:.2f}%)"
        if total > 0
        else "Genuine               : 0"
    )

    print(
        f"\nVC detection rate     : "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Mean confidence       : "
        f"{mean_confidence:.2f}%"
    )

    print(
        f"Median confidence     : "
        f"{median_confidence:.2f}%"
    )

    print("\nExpected class:")
    print("Voice_Conversion")

    print("\nInterpretation:")

    if total == 5 and vc_predictions == 5:
        print(
            "PASS - All 5 held-out Voice Conversion files "
            "were classified as Voice Conversion."
        )
    elif total > 0:
        print(
            "Evaluation completed, but not all held-out "
            "Voice Conversion files were classified correctly."
        )
    else:
        print(
            "FAIL - No files were successfully evaluated."
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()