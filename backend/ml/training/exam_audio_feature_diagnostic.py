import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa


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
# Exam files
# ============================================================

EXAM_FILES = {
    "Genuine_Recording": (
        Path(r"C:\DeepShield\exam_test_audio\Recording.m4a")
    ),

    "VC_A17": (
        Path(r"C:\DeepShield\exam_test_audio\VC_A17.flac")
    ),

    "TTS_sample": (
        Path(
            r"C:\DeepShield\exam_test_audio"
            r"\voice-sample-44100hz-16bit.flac"
        )
    ),
}


# ============================================================
# Extract RAW Mel features
# ============================================================

def extract_raw_mel(file_path):

    audio, sample_rate = load_audio(
        file_path,
        sample_rate=TARGET_SAMPLE_RATE,
    )

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

    return mel_db.astype(np.float32)


# ============================================================
# Extract normalized Mel features
# ============================================================

def extract_normalized_mel(file_path):

    mel_db = extract_raw_mel(file_path)

    mean = mel_db.mean()
    std = mel_db.std()

    normalized = (
        mel_db - mean
    ) / (
        std + 1e-8
    )

    return normalized.astype(np.float32)


# ============================================================
# Calculate statistics
# ============================================================

def calculate_statistics(
    raw_mel,
    normalized_mel,
):

    return {
        "raw_mean": float(raw_mel.mean()),
        "raw_std": float(raw_mel.std()),
        "raw_min": float(raw_mel.min()),
        "raw_max": float(raw_mel.max()),
        "raw_p10": float(np.percentile(raw_mel, 10)),
        "raw_p25": float(np.percentile(raw_mel, 25)),
        "raw_median": float(np.percentile(raw_mel, 50)),
        "raw_p75": float(np.percentile(raw_mel, 75)),
        "raw_p90": float(np.percentile(raw_mel, 90)),
        "normalized_mean": float(normalized_mel.mean()),
        "normalized_std": float(normalized_mel.std()),
        "normalized_min": float(normalized_mel.min()),
        "normalized_max": float(normalized_mel.max()),
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 100)
    print("DeepShield Exam Audio Feature Distribution Diagnostic")
    print("=" * 100)

    results = []

    for name, file_path in EXAM_FILES.items():

        print()
        print("-" * 100)
        print(f"Processing: {name}")
        print(f"File: {file_path}")
        print("-" * 100)

        if not file_path.exists():

            print(
                f"ERROR: File not found: {file_path}"
            )

            continue

        try:

            raw_mel = extract_raw_mel(
                file_path
            )

            normalized_mel = extract_normalized_mel(
                file_path
            )

            statistics = calculate_statistics(
                raw_mel,
                normalized_mel,
            )

            statistics["file"] = name
            statistics["path"] = str(file_path)
            statistics["shape"] = str(
                raw_mel.shape
            )

            results.append(
                statistics
            )

            print(
                "Feature shape:",
                raw_mel.shape,
            )

            print()
            print("RAW MEL")
            print(
                "Mean:",
                round(statistics["raw_mean"], 4),
            )
            print(
                "Std:",
                round(statistics["raw_std"], 4),
            )
            print(
                "Min:",
                round(statistics["raw_min"], 4),
            )
            print(
                "Max:",
                round(statistics["raw_max"], 4),
            )

            print()
            print("Normalized MEL")
            print(
                "Mean:",
                round(
                    statistics["normalized_mean"],
                    4,
                ),
            )
            print(
                "Std:",
                round(
                    statistics["normalized_std"],
                    4,
                ),
            )
            print(
                "Min:",
                round(
                    statistics["normalized_min"],
                    4,
                ),
            )
            print(
                "Max:",
                round(
                    statistics["normalized_max"],
                    4,
                ),
            )

        except Exception as error:

            print(
                "ERROR:",
                error,
            )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 100)
    print("EXAM AUDIO FEATURE SUMMARY")
    print("=" * 100)

    if results:

        dataframe = pd.DataFrame(
            results
        )

        columns = [
            "file",
            "shape",
            "raw_mean",
            "raw_std",
            "raw_min",
            "raw_max",
            "raw_p10",
            "raw_p25",
            "raw_median",
            "raw_p75",
            "raw_p90",
            "normalized_mean",
            "normalized_std",
            "normalized_min",
            "normalized_max",
        ]

        print(
            dataframe[
                columns
            ].to_string(
                index=False
            )
        )

        output_file = (
            BACKEND_DIR
            / "ml"
            / "evaluation"
            / "exam_audio_feature_diagnostic.csv"
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            output_file,
            index=False,
        )

        print()
        print(
            "Results saved:"
        )
        print(output_file)

    else:

        print(
            "No files were successfully processed."
        )

    print()
    print("=" * 100)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 100)