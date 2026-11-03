from pathlib import Path
import csv


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    BASE_DIR
    / "ml"
    / "dataset"
    / "LA"
)


TRAIN_AUDIO_DIR = (
    DATASET_DIR
    / "ASVspoof2019_LA_train"
    / "flac"
)


DEV_AUDIO_DIR = (
    DATASET_DIR
    / "ASVspoof2019_LA_dev"
    / "flac"
)


EVAL_AUDIO_DIR = (
    DATASET_DIR
    / "ASVspoof2019_LA_eval"
    / "flac"
)


PROTOCOL_DIR = (
    DATASET_DIR
    / "ASVspoof2019_LA_cm_protocols"
)


TRAIN_PROTOCOL = (
    PROTOCOL_DIR
    / "ASVspoof2019.LA.cm.train.trn.txt"
)


DEV_PROTOCOL = (
    PROTOCOL_DIR
    / "ASVspoof2019.LA.cm.dev.trl.txt"
)


EVAL_PROTOCOL = (
    PROTOCOL_DIR
    / "ASVspoof2019.LA.cm.eval.trl.txt"
)


OUTPUT_DIR = (
    BASE_DIR
    / "ml"
    / "dataset"
)


TRAIN_MANIFEST = (
    OUTPUT_DIR
    / "train_manifest.csv"
)


DEV_MANIFEST = (
    OUTPUT_DIR
    / "dev_manifest.csv"
)


EVAL_MANIFEST = (
    OUTPUT_DIR
    / "eval_manifest.csv"
)


# ============================================================
# ASVspoof attack mapping
# ============================================================
#
# DeepShield currently has 3 classes:
#
# 0 = Genuine
# 1 = TTS
# 2 = Voice Conversion
#
# ASVspoof 2019 LA contains:
#
# A01-A04  TTS
# A05-A06  VC
# A07-A12  TTS
# A13-A15  TTS + VC
# A16      TTS
# A17-A19  VC
#
# Because our current DeepShield model has only 3 classes,
# TTS_VC attacks are assigned to the TTS class for now.
#
# We will later be able to introduce a dedicated
# TTS_VC / DeepFake category if we expand the model.
# ============================================================


TTS_ATTACKS = {
    "A01",
    "A02",
    "A03",
    "A04",
    "A07",
    "A08",
    "A09",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
    "A15",
    "A16",
}


VC_ATTACKS = {
    "A05",
    "A06",
    "A17",
    "A18",
    "A19",
}


# ============================================================
# Convert ASVspoof label to DeepShield class
# ============================================================

def map_label(
    attack_id,
    key,
):

    # --------------------------------------------------------
    # Genuine speech
    # --------------------------------------------------------

    if key == "bonafide":

        return "Genuine", 0


    # --------------------------------------------------------
    # TTS / TTS-VC
    # --------------------------------------------------------

    if attack_id in TTS_ATTACKS:

        return "TTS", 1


    # --------------------------------------------------------
    # Voice Conversion
    # --------------------------------------------------------

    if attack_id in VC_ATTACKS:

        return "Voice_Conversion", 2


    # --------------------------------------------------------
    # Unknown attack
    # --------------------------------------------------------

    raise ValueError(
        f"Unknown attack ID: {attack_id}"
    )


# ============================================================
# Create manifest
# ============================================================

def create_manifest(
    protocol_file,
    audio_directory,
    output_file,
    split_name,
):

    rows = []

    missing_files = 0

    unknown_attacks = set()


    # ========================================================
    # Read protocol
    # ========================================================

    with open(
        protocol_file,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()


            if not line:

                continue


            parts = line.split()


            # ------------------------------------------------
            # Expected format:
            #
            # SPEAKER_ID
            # AUDIO_FILE_NAME
            # -
            # SYSTEM_ID
            # KEY
            # ------------------------------------------------

            if len(parts) != 5:

                print(
                    "Skipping unexpected line:",
                    line,
                )

                continue


            speaker_id = parts[0]

            audio_id = parts[1]

            attack_id = parts[3]

            key = parts[4]


            # =================================================
            # Convert label
            # =================================================

            try:

                label_name, label_id = map_label(
                    attack_id,
                    key,
                )

            except ValueError as error:

                unknown_attacks.add(
                    attack_id
                )

                print(error)

                continue


            # =================================================
            # Audio path
            # =================================================

            audio_path = (
                audio_directory
                / f"{audio_id}.flac"
            )


            if not audio_path.exists():

                missing_files += 1

                continue


            # =================================================
            # Add row
            # =================================================

            rows.append(
                {
                    "audio_path": str(
                        audio_path.resolve()
                    ),

                    "audio_id": audio_id,

                    "speaker_id": speaker_id,

                    "attack_id": attack_id,

                    "original_label": key,

                    "label": label_name,

                    "label_id": label_id,

                    "split": split_name,
                }
            )


    # ========================================================
    # Create output directory
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Write CSV
    # ========================================================

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "audio_path",
            "audio_id",
            "speaker_id",
            "attack_id",
            "original_label",
            "label",
            "label_id",
            "split",
        ]


        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )


        writer.writeheader()

        writer.writerows(rows)


    # ========================================================
    # Display results
    # ========================================================

    print()

    print("=" * 60)

    print(
        f"{split_name.upper()} MANIFEST CREATED"
    )

    print("=" * 60)


    print(
        f"Output: {output_file}"
    )


    print(
        f"Total files: {len(rows)}"
    )


    print(
        f"Missing files: {missing_files}"
    )


    # ========================================================
    # Label counts
    # ========================================================

    counts = {}


    for row in rows:

        label = row["label"]


        counts[label] = (
            counts.get(
                label,
                0,
            )
            + 1
        )


    print()


    for label in [
        "Genuine",
        "TTS",
        "Voice_Conversion",
    ]:

        print(
            f"{label}: "
            f"{counts.get(label, 0)}"
        )


    # ========================================================
    # Attack distribution
    # ========================================================

    attack_counts = {}


    for row in rows:

        attack = row["attack_id"]


        attack_counts[attack] = (
            attack_counts.get(
                attack,
                0,
            )
            + 1
        )


    print()

    print(
        "Attack distribution:"
    )


    for attack in sorted(
        attack_counts.keys()
    ):

        print(
            f"{attack}: "
            f"{attack_counts[attack]}"
        )


    # ========================================================
    # Unknown attacks
    # ========================================================

    if unknown_attacks:

        print()

        print(
            "WARNING: Unknown attack IDs:"
        )

        for attack in sorted(
            unknown_attacks
        ):

            print(
                f"  {attack}"
            )


    print("=" * 60)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "DeepShield Dataset Manifest Generator"
    )

    print()

    print(
        "Dataset:",
        DATASET_DIR,
    )

    print()


    # ========================================================
    # Training manifest
    # ========================================================

    create_manifest(
        protocol_file=TRAIN_PROTOCOL,

        audio_directory=TRAIN_AUDIO_DIR,

        output_file=TRAIN_MANIFEST,

        split_name="train",
    )


    # ========================================================
    # Development manifest
    # ========================================================

    create_manifest(
        protocol_file=DEV_PROTOCOL,

        audio_directory=DEV_AUDIO_DIR,

        output_file=DEV_MANIFEST,

        split_name="dev",
    )


    # ========================================================
    # Evaluation manifest
    # ========================================================

    create_manifest(
        protocol_file=EVAL_PROTOCOL,

        audio_directory=EVAL_AUDIO_DIR,

        output_file=EVAL_MANIFEST,

        split_name="eval",
    )


    print()

    print(
        "Manifest generation completed."
    )