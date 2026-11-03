import shutil
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from app.services.predict import predict_audio


# ============================================================
# Configuration
# ============================================================

MANIFEST_PATH = (
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "eval_manifest.csv"
)

OUTPUT_DIR = (
    BACKEND_DIR.parent
    / "exam_demo_audio"
)

RESULTS_PATH = (
    BACKEND_DIR
    / "ml"
    / "evaluation"
    / "exam_demo_selection.csv"
)

SAMPLES_PER_CLASS = 5

CLASS_NAMES = [
    "Genuine",
    "TTS",
    "Voice Conversion",
]


# ============================================================
# Helpers
# ============================================================

def normalize_label(label):

    label = str(label).strip()

    if label.lower() == "genuine":
        return "Genuine"

    if label.lower() == "tts":
        return "TTS"

    if label.lower() in [
        "voice_conversion",
        "voice conversion",
        "voice-conversion",
    ]:
        return "Voice Conversion"

    return label


def find_audio_path(row):

    # --------------------------------------------------------
    # Prefer audio_path if present
    # --------------------------------------------------------

    if "audio_path" in row.index:

        value = str(
            row["audio_path"]
        ).strip()

        if value and value.lower() != "nan":

            path = Path(value)

            if path.exists():
                return path

    # --------------------------------------------------------
    # Try common ASVspoof locations
    # --------------------------------------------------------

    audio_id = str(
        row["audio_id"]
    ).strip()

    candidates = [

        BACKEND_DIR
        / "ml"
        / "dataset"
        / "LA"
        / "ASVspoof2019_LA_eval"
        / "flac"
        / f"{audio_id}.flac",

        BACKEND_DIR
        / "ml"
        / "dataset"
        / "LA"
        / "ASVspoof2019_LA_train"
        / "flac"
        / f"{audio_id}.flac",

    ]

    for path in candidates:

        if path.exists():
            return path

    return None


def safe_filename(
    audio_id,
    extension,
):

    return (
        f"{audio_id}{extension.lower()}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 90)
    print("DeepShield Verified Exam Demo Set Generator")
    print("=" * 90)

    print()
    print("Manifest:")
    print(MANIFEST_PATH)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Verify manifest
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Evaluation manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print()
    print(
        "Evaluation samples:",
        len(df),
    )

    print()
    print(
        "Manifest columns:"
    )

    print(
        list(df.columns)
    )

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    required_columns = [
        "audio_id",
        "label",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: "
                f"{column}"
            )

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    df["expected_label"] = (
        df["label"]
        .apply(normalize_label)
    )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    for class_name in CLASS_NAMES:

        (
            OUTPUT_DIR
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Track selected samples
    # --------------------------------------------------------

    selected = {
        class_name: []
        for class_name in CLASS_NAMES
    }

    diagnostic_rows = []

    # --------------------------------------------------------
    # Process samples
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("Searching for correctly classified samples")
    print("=" * 90)

    # Shuffle deterministically so the same set can be
    # reproduced later.
    candidates = df.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(
        drop=True
    )

    processed = 0

    for _, row in candidates.iterrows():

        expected_label = row[
            "expected_label"
        ]

        # Ignore labels outside our 3-class system
        if expected_label not in CLASS_NAMES:
            continue

        # Stop when all classes are full
        if all(
            len(selected[class_name])
            >= SAMPLES_PER_CLASS
            for class_name in CLASS_NAMES
        ):
            break

        # Skip a class that already has enough samples
        if (
            len(selected[expected_label])
            >= SAMPLES_PER_CLASS
        ):
            continue

        audio_path = find_audio_path(
            row
        )

        if audio_path is None:

            continue

        processed += 1

        try:

            result = predict_audio(
                str(audio_path)
            )

            predicted_label = result[
                "prediction"
            ]

            confidence = float(
                result["confidence"]
            )

            correct = (
                predicted_label
                == expected_label
            )

            diagnostic_rows.append(
                {
                    "audio_id": row[
                        "audio_id"
                    ],
                    "attack_id": row.get(
                        "attack_id",
                        "",
                    ),
                    "expected_label":
                        expected_label,
                    "predicted_label":
                        predicted_label,
                    "confidence":
                        confidence,
                    "correct":
                        correct,
                    "source_path":
                        str(audio_path),
                }
            )

            if not correct:
                continue

            # ------------------------------------------------
            # Copy correctly classified file
            # ------------------------------------------------

            extension = (
                audio_path.suffix
            )

            destination = (
                OUTPUT_DIR
                / expected_label
                / safe_filename(
                    str(
                        row["audio_id"]
                    ),
                    extension,
                )
            )

            shutil.copy2(
                audio_path,
                destination,
            )

            selected[
                expected_label
            ].append(
                {
                    "audio_id": str(
                        row["audio_id"]
                    ),
                    "attack_id": str(
                        row.get(
                            "attack_id",
                            "",
                        )
                    ),
                    "expected_label":
                        expected_label,
                    "predicted_label":
                        predicted_label,
                    "confidence":
                        confidence,
                    "source_path":
                        str(audio_path),
                    "demo_path":
                        str(destination),
                }
            )

            print()
            print(
                f"[SELECTED] "
                f"{expected_label:<18} "
                f"{row['audio_id']} "
                f"-> "
                f"{confidence:.2f}%"
            )

        except Exception as error:

            diagnostic_rows.append(
                {
                    "audio_id": row[
                        "audio_id"
                    ],
                    "attack_id": row.get(
                        "attack_id",
                        "",
                    ),
                    "expected_label":
                        expected_label,
                    "predicted_label":
                        "ERROR",
                    "confidence":
                        0.0,
                    "correct":
                        False,
                    "source_path":
                        str(audio_path),
                    "error":
                        str(error),
                }
            )

    # --------------------------------------------------------
    # Save diagnostics
    # --------------------------------------------------------

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )

    diagnostic_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print()
    print("=" * 90)
    print("VERIFIED EXAM DEMO SET")
    print("=" * 90)

    complete = True

    for class_name in CLASS_NAMES:

        count = len(
            selected[class_name]
        )

        print(
            f"{class_name:<20}: "
            f"{count}/{SAMPLES_PER_CLASS}"
        )

        if count < SAMPLES_PER_CLASS:
            complete = False

    print()
    print(
        "Total selected:",
        sum(
            len(value)
            for value in selected.values()
        ),
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Selection report:"
    )

    print(
        RESULTS_PATH
    )

    print()
    print("=" * 90)

    if complete:

        print(
            "SUCCESS: "
            "15 verified demo samples created."
        )

    else:

        print(
            "WARNING: "
            "Could not find enough correctly "
            "classified samples for every class."
        )

    print("=" * 90)