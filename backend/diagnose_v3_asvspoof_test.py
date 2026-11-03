from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

import librosa
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

BACKEND_ROOT = Path(r"C:\DeepShield\backend")
V3_ROOT = BACKEND_ROOT / "ml" / "experiments" / "v3"

ZIP_PATH = BACKEND_ROOT / "ml" / "dataset" / "LA.zip"

MANIFEST_PATH = (
    V3_ROOT
    / "dataset"
    / "manifests"
    / "v3_test.csv"
)

CHECKPOINT_PATH = (
    V3_ROOT
    / "checkpoints"
    / "deepshield_v3_best.pth"
)

OUTPUT_PATH = (
    V3_ROOT
    / "evaluation"
    / "v3_asvspoof_heldout_diagnostic.csv"
)

# Number of deterministic samples per class.
# 100 x 3 = 300 total.
NUM_PER_CLASS = 100
SEED = 42

# ============================================================
# V3 PREPROCESSING
# ============================================================

SAMPLE_RATE = 16000
DURATION = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

EXPECTED_SHAPE = (128, 251)

CLASS_NAMES = {
    0: "Genuine",
    1: "TTS",
    2: "Voice_Conversion",
}


# ============================================================
# IMPORT V3 MODEL
# ============================================================

sys.path.insert(0, str(V3_ROOT))

from model.v3_model import DeepShieldV3


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 75)
print("DeepShield V3 — Held-Out Test Diagnostic")
print("=" * 75)

print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA         : "
        f"{torch.version.cuda}"
    )

print(f"Checkpoint   : {CHECKPOINT_PATH}")
print(f"Test manifest: {MANIFEST_PATH}")
print(f"Archive      : {ZIP_PATH}")

print("=" * 75)


# ============================================================
# LOAD MODEL
# ============================================================

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
# READ MANIFEST
# ============================================================

def read_manifest():

    print("\nReading V3 test manifest...")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    rows = []

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "audio_id",
            "audio_path",
            "speaker_id",
            "attack_id",
            "label_name",
            "label_id",
            "split",
        }

        missing = required_columns - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "Manifest is missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            if row["split"].strip().lower() != "test":
                continue

            label_id = int(row["label_id"])

            if label_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unexpected label_id: {label_id}"
                )

            rows.append(row)

    print(f"Test rows: {len(rows)}")

    return rows


# ============================================================
# SELECT 100 PER CLASS
# ============================================================

def select_samples(rows):

    rng = np.random.default_rng(SEED)

    groups = {
        0: [],
        1: [],
        2: [],
    }

    for row in rows:

        label_id = int(row["label_id"])

        groups[label_id].append(row)

    print("\nManifest class distribution:")

    for label_id in range(3):

        print(
            f"  {CLASS_NAMES[label_id]:18s}: "
            f"{len(groups[label_id])}"
        )

    selected = []

    for label_id in range(3):

        if len(groups[label_id]) < NUM_PER_CLASS:

            raise RuntimeError(
                f"Not enough samples for "
                f"{CLASS_NAMES[label_id]}"
            )

        indices = rng.choice(
            len(groups[label_id]),
            size=NUM_PER_CLASS,
            replace=False,
        )

        for index in indices:

            selected.append(
                groups[label_id][index]
            )

    rng.shuffle(selected)

    print(
        f"\nSelected {len(selected)} samples "
        f"({NUM_PER_CLASS} per class)."
    )

    return selected


# ============================================================
# AUDIO -> V3 FEATURE
# ============================================================

def audio_bytes_to_feature(audio_bytes):

    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError("Empty audio")

    # --------------------------------------------------------
    # Exactly 4 seconds
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # V3 Mel spectrogram
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Per-file normalization
    # --------------------------------------------------------

    mean = float(mel_db.mean())
    std = float(mel_db.std())

    if std < 1e-8:

        raise ValueError(
            "Mel spectrogram has near-zero std"
        )

    mel_norm = (
        mel_db - mean
    ) / std

    mel_norm = np.asarray(
        mel_norm,
        dtype=np.float32,
    )

    if mel_norm.shape != EXPECTED_SHAPE:

        raise ValueError(
            f"Unexpected feature shape: "
            f"{mel_norm.shape}"
        )

    if not np.isfinite(mel_norm).all():

        raise ValueError(
            "Feature contains NaN/Inf"
        )

    return (
        mel_norm,
        audio,
        mel_db,
    )


# ============================================================
# PREDICTION
# ============================================================

@torch.inference_mode()
def predict_feature(feature):

    x = torch.from_numpy(feature)

    # (128,251)
    # -> (1,1,128,251)

    x = x.unsqueeze(0)
    x = x.unsqueeze(0)

    x = x.to(DEVICE)

    logits = model(x)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    predicted = int(
        torch.argmax(probabilities).item()
    )

    probs = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    return predicted, probs


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if not ZIP_PATH.exists():

        raise FileNotFoundError(
            f"LA.zip not found:\n{ZIP_PATH}"
        )

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    rows = read_manifest()

    selected = select_samples(rows)

    # --------------------------------------------------------
    # Open archive
    # --------------------------------------------------------

    print("\nOpening LA.zip...")

    results = []

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as zf:

        archive_members = set(
            zf.namelist()
        )

        print(
            f"Archive members: "
            f"{len(archive_members)}"
        )

        print("\nRunning inference...\n")

        for index, row in enumerate(
            selected,
            start=1,
        ):

            audio_id = row["audio_id"]
            audio_path = row["audio_path"]

            expected = int(
                row["label_id"]
            )

            # ------------------------------------------------
            # Check archive member
            # ------------------------------------------------

            if audio_path not in archive_members:

                print(
                    f"[{index:03d}] MISSING "
                    f"{audio_id}"
                )

                continue

            try:

                audio_bytes = zf.read(
                    audio_path
                )

                (
                    feature,
                    audio,
                    mel_db,
                ) = audio_bytes_to_feature(
                    audio_bytes
                )

                predicted, probs = predict_feature(
                    feature
                )

                correct = (
                    predicted == expected
                )

                result = {
                    "audio_id": audio_id,
                    "audio_path": audio_path,
                    "speaker_id": row[
                        "speaker_id"
                    ],
                    "attack_id": row[
                        "attack_id"
                    ],
                    "expected": expected,
                    "expected_name": CLASS_NAMES[
                        expected
                    ],
                    "predicted": predicted,
                    "predicted_name": CLASS_NAMES[
                        predicted
                    ],
                    "genuine_prob": float(
                        probs[0]
                    ),
                    "tts_prob": float(
                        probs[1]
                    ),
                    "vc_prob": float(
                        probs[2]
                    ),
                    "confidence": float(
                        probs[predicted]
                    ),
                    "correct": correct,
                    "audio_rms": float(
                        np.sqrt(
                            np.mean(
                                audio ** 2
                            )
                        )
                    ),
                    "audio_peak": float(
                        np.max(
                            np.abs(audio)
                        )
                    ),
                    "mel_mean": float(
                        mel_db.mean()
                    ),
                    "mel_std": float(
                        mel_db.std()
                    ),
                }

                results.append(result)

                # ------------------------------------------------
                # Print only errors
                # ------------------------------------------------

                if not correct:

                    print(
                        f"[{index:03d}] ERROR | "
                        f"{audio_id} | "
                        f"attack={row['attack_id']} | "
                        f"expected="
                        f"{CLASS_NAMES[expected]:16s} | "
                        f"predicted="
                        f"{CLASS_NAMES[predicted]:16s} | "
                        f"G="
                        f"{probs[0] * 100:6.2f}% "
                        f"TTS="
                        f"{probs[1] * 100:6.2f}% "
                        f"VC="
                        f"{probs[2] * 100:6.2f}%"
                    )

            except Exception as e:

                print(
                    f"[{index:03d}] FAILED | "
                    f"{audio_id} | "
                    f"{e}"
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("HELD-OUT V3 TEST DIAGNOSTIC SUMMARY")
    print("=" * 75)

    for class_id in range(3):

        class_results = [
            r
            for r in results
            if r["expected"] == class_id
        ]

        if not class_results:
            continue

        correct = sum(
            r["correct"]
            for r in class_results
        )

        accuracy = (
            correct
            / len(class_results)
            * 100
        )

        avg_confidence = (
            np.mean(
                [
                    r["confidence"]
                    for r in class_results
                ]
            )
            * 100
        )

        print(
            f"{CLASS_NAMES[class_id]:18s} | "
            f"{correct:3d}/"
            f"{len(class_results):3d} correct | "
            f"Accuracy "
            f"{accuracy:6.2f}% | "
            f"Avg confidence "
            f"{avg_confidence:6.2f}%"
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    total_correct = sum(
        r["correct"]
        for r in results
    )

    total = len(results)

    overall_accuracy = (
        total_correct
        / total
        * 100
        if total
        else 0
    )

    print("-" * 75)

    print(
        f"Overall            | "
        f"{total_correct:3d}/"
        f"{total:3d} correct | "
        f"Accuracy "
        f"{overall_accuracy:6.2f}%"
    )

    # ========================================================
    # ERROR BREAKDOWN
    # ========================================================

    errors = [
        r
        for r in results
        if not r["correct"]
    ]

    print("\nError breakdown:")

    if not errors:

        print("  No errors found.")

    else:

        transitions = {}

        for r in errors:

            key = (
                r["expected_name"],
                r["predicted_name"],
            )

            transitions[key] = (
                transitions.get(key, 0)
                + 1
            )

        for (
            expected_name,
            predicted_name,
        ), count in sorted(
            transitions.items()
        ):

            print(
                f"  {expected_name} -> "
                f"{predicted_name}: "
                f"{count}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "audio_id",
        "audio_path",
        "speaker_id",
        "attack_id",
        "expected",
        "expected_name",
        "predicted",
        "predicted_name",
        "genuine_prob",
        "tts_prob",
        "vc_prob",
        "confidence",
        "correct",
        "audio_rms",
        "audio_peak",
        "mel_mean",
        "mel_std",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        "\nDetailed results saved to:"
    )

    print(OUTPUT_PATH)

    print("\n" + "=" * 75)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()