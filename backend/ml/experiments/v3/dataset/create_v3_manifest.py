"""
DeepShield V3 - Balanced ASVspoof 2019 LA Manifest Builder

V3 classes:
    0 = Genuine
    1 = TTS
    2 = Voice Conversion

Source:
    ASVspoof 2019 LA training set inside LA.zip

Source distribution:
    Genuine = 2,580
    A01 = 3,800  -> TTS
    A02 = 3,800  -> TTS
    A03 = 3,800  -> TTS
    A04 = 3,800  -> TTS
    A05 = 3,800  -> Voice Conversion
    A06 = 3,800  -> Voice Conversion

Important design:
    1. Speakers are assigned to exactly one split.
    2. Train/dev/test are therefore speaker-disjoint.
    3. Each split is independently balanced across the three classes.
    4. Larger classes are downsampled to the smallest class in
       that split.
    5. All attack systems A01-A06 remain represented where possible.
    6. No audio is extracted.
    7. LA.zip is never modified.
    8. Existing V1/V2 datasets are never modified.

Random seed:
    42
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

BACKEND_DIR = PROJECT_ROOT / "backend"

DATASET_DIR = (
    BACKEND_DIR
    / "ml"
    / "dataset"
)

LA_ZIP = DATASET_DIR / "LA.zip"

V3_DIR = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3"
)

MANIFEST_DIR = (
    V3_DIR
    / "dataset"
    / "manifests"
)

METADATA_DIR = (
    V3_DIR
    / "dataset"
    / "metadata"
)


# ============================================================
# ASVspoof paths
# ============================================================

PROTOCOL_MEMBER = (
    "LA/"
    "ASVspoof2019_LA_cm_protocols/"
    "ASVspoof2019.LA.cm.train.trn.txt"
)

AUDIO_PREFIX = (
    "LA/"
    "ASVspoof2019_LA_train/"
    "flac/"
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
TEST_RATIO = 0.15

SPLITS = (
    "train",
    "dev",
    "test",
)


# ============================================================
# Utility functions
# ============================================================

def stable_hash(value: str) -> int:
    """
    Deterministic integer hash.
    """

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:16],
        16,
    )


def assign_speaker_split(
    speaker_id: str,
) -> str:
    """
    Deterministically assign a speaker.

    Approximately:
        70% train
        15% dev
        15% test
    """

    value = (
        stable_hash(
            f"DeepShield-V3-{speaker_id}"
        )
        % 10000
    ) / 10000.0

    if value < TRAIN_RATIO:
        return "train"

    if value < (
        TRAIN_RATIO
        + DEV_RATIO
    ):
        return "dev"

    return "test"


def deterministic_shuffle(
    records,
    seed_text: str,
):
    """
    Deterministically shuffle records.
    """

    records = list(records)

    rng = random.Random(
        SEED
        + stable_hash(seed_text)
    )

    rng.shuffle(records)

    return records


def ensure_directories():
    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Read ASVspoof protocol
# ============================================================

def read_protocol(
    archive: zipfile.ZipFile,
):
    """
    Read:

        speaker_id
        audio_id
        system_id
        attack_id
        label
    """

    print()
    print("=" * 70)
    print("Reading ASVspoof 2019 LA training protocol")
    print("=" * 70)

    with archive.open(
        PROTOCOL_MEMBER,
        "r",
    ) as file:

        text = file.read().decode(
            "utf-8"
        )

    records = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:

            raise RuntimeError(
                "Unexpected protocol format "
                f"at line {line_number}:\n"
                f"{line}"
            )

        (
            speaker_id,
            audio_id,
            system_id,
            attack_id,
            label,
        ) = parts

        records.append(
            {
                "speaker_id": speaker_id,
                "audio_id": audio_id,
                "system_id": system_id,
                "attack_id": attack_id,
                "label": label,
            }
        )

    return records


# ============================================================
# Source validation
# ============================================================

def validate_source(
    records,
    archive_members,
):
    print()
    print("=" * 70)
    print("Validating source dataset")
    print("=" * 70)

    label_counts = Counter(
        record["label"]
        for record in records
    )

    print(
        "Genuine protocol entries : "
        f"{label_counts['bonafide']}"
    )

    print(
        "Spoof protocol entries   : "
        f"{label_counts['spoof']}"
    )

    expected_total = 25380

    if len(records) != expected_total:

        raise RuntimeError(
            "Unexpected protocol size: "
            f"{len(records)} "
            f"(expected {expected_total})"
        )

    expected_attack_counts = {
        "A01": 3800,
        "A02": 3800,
        "A03": 3800,
        "A04": 3800,
        "A05": 3800,
        "A06": 3800,
    }

    attack_counts = Counter(
        record["attack_id"]
        for record in records
        if record["label"] == "spoof"
    )

    print()
    print("Attack distribution:")

    for attack_id in sorted(
        attack_counts
    ):

        print(
            f"  {attack_id}: "
            f"{attack_counts[attack_id]}"
        )

    if dict(attack_counts) != (
        expected_attack_counts
    ):

        raise RuntimeError(
            "Unexpected attack distribution."
        )

    # --------------------------------------------------------
    # Verify every protocol audio file exists
    # --------------------------------------------------------

    print()
    print(
        "Checking audio members..."
    )

    archive_member_set = set(
        archive_members
    )

    missing = []

    for record in records:

        member = (
            AUDIO_PREFIX
            + record["audio_id"]
            + ".flac"
        )

        if member not in archive_member_set:

            missing.append(
                member
            )

    if missing:

        print(
            f"Missing audio files: "
            f"{len(missing)}"
        )

        for member in missing[:10]:

            print(
                f"  {member}"
            )

        raise RuntimeError(
            "Protocol/audio mismatch."
        )

    print(
        "All protocol audio files verified."
    )


# ============================================================
# Label mapping
# ============================================================

def map_label(
    record,
):
    """
    DeepShield V3 label mapping.

    0 = Genuine
    1 = TTS
    2 = Voice Conversion
    """

    if record["label"] == "bonafide":

        return (
            0,
            "Genuine",
        )

    attack_id = record["attack_id"]

    if attack_id in {
        "A01",
        "A02",
        "A03",
        "A04",
    }:

        return (
            1,
            "TTS",
        )

    if attack_id in {
        "A05",
        "A06",
    }:

        return (
            2,
            "Voice_Conversion",
        )

    raise RuntimeError(
        f"Unknown attack ID: {attack_id}"
    )


# ============================================================
# Enrich records
# ============================================================

def enrich_records(
    records,
    speaker_to_split,
):
    enriched = []

    for record in records:

        label_id, label_name = map_label(
            record
        )

        audio_path = (
            AUDIO_PREFIX
            + record["audio_id"]
            + ".flac"
        )

        enriched.append(
            {
                **record,
                "split": speaker_to_split[
                    record["speaker_id"]
                ],
                "label_id": label_id,
                "label_name": label_name,
                "audio_path": audio_path,
            }
        )

    return enriched


# ============================================================
# Speaker split
# ============================================================

def create_speaker_splits(
    records,
):
    """
    Create a deterministic speaker-disjoint split.

    ASVspoof 2019 LA training contains 20 speakers.

    We explicitly assign:
        14 speakers -> train
        3 speakers  -> dev
        3 speakers  -> test

    This gives an exact 70/15/15 speaker split.

    A seeded shuffle is used so the assignment is:
        - deterministic
        - reproducible
        - guaranteed to contain all three splits

    Speakers are never shared between splits.
    """

    print()
    print("=" * 70)
    print("Creating speaker-disjoint splits")
    print("=" * 70)

    speakers = sorted(
        {
            record["speaker_id"]
            for record in records
        }
    )

    total_speakers = len(
        speakers
    )

    if total_speakers != 20:

        raise RuntimeError(
            "Expected 20 training speakers, "
            f"but found {total_speakers}."
        )

    # --------------------------------------------------------
    # Deterministic shuffle
    # --------------------------------------------------------

    rng = random.Random(
        SEED
    )

    shuffled_speakers = list(
        speakers
    )

    rng.shuffle(
        shuffled_speakers
    )

    # --------------------------------------------------------
    # Exact speaker allocation
    #
    # 20 speakers:
    #
    # 14 -> train
    # 3  -> dev
    # 3  -> test
    # --------------------------------------------------------

    train_speakers = (
        shuffled_speakers[:14]
    )

    dev_speakers = (
        shuffled_speakers[14:17]
    )

    test_speakers = (
        shuffled_speakers[17:20]
    )

    speaker_to_split = {}

    for speaker_id in train_speakers:

        speaker_to_split[
            speaker_id
        ] = "train"

    for speaker_id in dev_speakers:

        speaker_to_split[
            speaker_id
        ] = "dev"

    for speaker_id in test_speakers:

        speaker_to_split[
            speaker_id
        ] = "test"

    # --------------------------------------------------------
    # Safety validation
    # --------------------------------------------------------

    if len(speaker_to_split) != 20:

        raise RuntimeError(
            "Speaker assignment does not "
            "contain exactly 20 speakers."
        )

    if (
        set(train_speakers)
        & set(dev_speakers)
    ):

        raise RuntimeError(
            "Speaker leakage between "
            "train and dev."
        )

    if (
        set(train_speakers)
        & set(test_speakers)
    ):

        raise RuntimeError(
            "Speaker leakage between "
            "train and test."
        )

    if (
        set(dev_speakers)
        & set(test_speakers)
    ):

        raise RuntimeError(
            "Speaker leakage between "
            "dev and test."
        )

    # --------------------------------------------------------
    # Display assignment
    # --------------------------------------------------------

    print()
    print(
        "Speaker allocation:"
    )

    print(
        f"  train: {len(train_speakers)} speakers"
    )

    print(
        f"  dev  : {len(dev_speakers)} speakers"
    )

    print(
        f"  test : {len(test_speakers)} speakers"
    )

    print()
    print(
        "TRAIN speakers:"
    )

    print(
        "  "
        + ", ".join(
            sorted(train_speakers)
        )
    )

    print()
    print(
        "DEV speakers:"
    )

    print(
        "  "
        + ", ".join(
            sorted(dev_speakers)
        )
    )

    print()
    print(
        "TEST speakers:"
    )

    print(
        "  "
        + ", ".join(
            sorted(test_speakers)
        )
    )

    # --------------------------------------------------------
    # Final split count
    # --------------------------------------------------------

    counts = Counter(
        speaker_to_split.values()
    )

    print()
    print(
        "Final speaker distribution:"
    )

    for split in SPLITS:

        print(
            f"  {split:5s}: "
            f"{counts[split]} speakers"
        )

    return speaker_to_split


# ============================================================
# Display available class counts
# ============================================================

def show_available_counts(
    enriched,
):
    print()
    print("=" * 70)
    print("Available samples per split")
    print("=" * 70)

    for split in SPLITS:

        print()
        print(
            split.upper()
        )

        for label in (
            "Genuine",
            "TTS",
            "Voice_Conversion",
        ):

            count = sum(
                1
                for record in enriched
                if (
                    record["split"] == split
                    and record["label_name"] == label
                )
            )

            print(
                f"  {label:18s}: "
                f"{count}"
            )


# ============================================================
# Select balanced split
# ============================================================

def select_balanced_split(
    records,
    split,
):
    """
    Balance a single split.

    The smallest class determines the final
    number of samples per class.

    Example:
        Genuine       1417
        TTS           5000
        VC            3000

    becomes:

        Genuine       1417
        TTS           1417
        VC            1417
    """

    print()
    print(
        f"Balancing {split.upper()}..."
    )

    class_records = {}

    for label in (
        "Genuine",
        "TTS",
        "Voice_Conversion",
    ):

        candidates = [
            record
            for record in records
            if (
                record["split"] == split
                and record["label_name"] == label
            )
        ]

        class_records[label] = (
            candidates
        )

    available = {
        label: len(
            class_records[label]
        )
        for label in class_records
    }

    target = min(
        available.values()
    )

    print(
        "Available:"
    )

    for label in (
        "Genuine",
        "TTS",
        "Voice_Conversion",
    ):

        print(
            f"  {label:18s}: "
            f"{available[label]}"
        )

    print(
        f"Balanced target per class: "
        f"{target}"
    )

    selected = []

    # --------------------------------------------------------
    # Genuine
    # --------------------------------------------------------

    chosen = deterministic_shuffle(
        class_records["Genuine"],
        f"{split}-Genuine",
    )[:target]

    selected.extend(
        chosen
    )

    # --------------------------------------------------------
    # TTS
    #
    # Preserve A01-A04 as evenly as possible.
    # --------------------------------------------------------

    tts_selected = select_attack_balanced(
        class_records["TTS"],
        attack_ids=[
            "A01",
            "A02",
            "A03",
            "A04",
        ],
        target=target,
        seed_prefix=f"{split}-TTS",
    )

    selected.extend(
        tts_selected
    )

    # --------------------------------------------------------
    # VC
    #
    # Preserve A05-A06 as evenly as possible.
    # --------------------------------------------------------

    vc_selected = select_attack_balanced(
        class_records["Voice_Conversion"],
        attack_ids=[
            "A05",
            "A06",
        ],
        target=target,
        seed_prefix=f"{split}-VC",
    )

    selected.extend(
        vc_selected
    )

    return selected


# ============================================================
# Attack-balanced selection
# ============================================================

def select_attack_balanced(
    records,
    attack_ids,
    target,
    seed_prefix,
):
    """
    Select exactly 'target' samples while distributing
    samples as evenly as possible across attack systems.

    If one attack has fewer available samples, the remaining
    quota is redistributed to the other attacks.
    """

    by_attack = {}

    for attack_id in attack_ids:

        by_attack[attack_id] = (
            deterministic_shuffle(
                [
                    record
                    for record in records
                    if record["attack_id"]
                    == attack_id
                ],
                f"{seed_prefix}-{attack_id}",
            )
        )

    available = {
        attack_id: len(
            by_attack[attack_id]
        )
        for attack_id in attack_ids
    }

    selected = []

    # --------------------------------------------------------
    # Start with an equal quota
    # --------------------------------------------------------

    base_quota = (
        target // len(attack_ids)
    )

    remainder = (
        target
        % len(attack_ids)
    )

    quotas = {
        attack_id: base_quota
        for attack_id in attack_ids
    }

    # Give remainder to attacks in deterministic order.
    for attack_id in attack_ids:

        if remainder <= 0:
            break

        quotas[attack_id] += 1
        remainder -= 1

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    for attack_id in attack_ids:

        take = min(
            quotas[attack_id],
            available[attack_id],
        )

        selected.extend(
            by_attack[attack_id][
                :take
            ]
        )

        quotas[attack_id] -= take

    # --------------------------------------------------------
    # Redistribute unmet quota
    # --------------------------------------------------------

    remaining = (
        target
        - len(selected)
    )

    if remaining > 0:

        leftovers = []

        for attack_id in attack_ids:

            already_selected = sum(
                1
                for record in selected
                if record["attack_id"]
                == attack_id
            )

            leftovers.extend(
                by_attack[attack_id][
                    already_selected:
                ]
            )

        leftovers = deterministic_shuffle(
            leftovers,
            f"{seed_prefix}-redistribution",
        )

        selected.extend(
            leftovers[:remaining]
        )

    if len(selected) != target:

        raise RuntimeError(
            f"Could not select {target} "
            f"samples for {seed_prefix}. "
            f"Selected {len(selected)}."
        )

    return selected


# ============================================================
# Validate selected dataset
# ============================================================

def validate_selected(
    selected,
):
    print()
    print("=" * 70)
    print("Validating final V3 dataset")
    print("=" * 70)

    # --------------------------------------------------------
    # Audio uniqueness
    # --------------------------------------------------------

    audio_ids = [
        record["audio_id"]
        for record in selected
    ]

    duplicate_audio = [
        audio_id
        for audio_id, count
        in Counter(audio_ids).items()
        if count > 1
    ]

    if duplicate_audio:

        raise RuntimeError(
            "Duplicate audio IDs detected: "
            f"{len(duplicate_audio)}"
        )

    print(
        "Duplicate audio check: PASS"
    )

    # --------------------------------------------------------
    # Speaker leakage
    # --------------------------------------------------------

    speakers_by_split = {
        split: {
            record["speaker_id"]
            for record in selected
            if record["split"] == split
        }
        for split in SPLITS
    }

    for first, second in (
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ):

        overlap = (
            speakers_by_split[first]
            & speakers_by_split[second]
        )

        if overlap:

            raise RuntimeError(
                "Speaker leakage detected "
                f"between {first} and {second}: "
                f"{sorted(overlap)}"
            )

    print(
        "Speaker leakage check: PASS"
    )

    # --------------------------------------------------------
    # Per-split class balance
    # --------------------------------------------------------

    print()
    print(
        "Final class distribution:"
    )

    for split in SPLITS:

        print()
        print(
            split.upper()
        )

        counts = Counter(
            record["label_name"]
            for record in selected
            if record["split"] == split
        )

        values = []

        for label in (
            "Genuine",
            "TTS",
            "Voice_Conversion",
        ):

            count = counts[label]

            values.append(
                count
            )

            print(
                f"  {label:18s}: "
                f"{count}"
            )

        if len(
            set(values)
        ) != 1:

            raise RuntimeError(
                f"{split} is not balanced."
            )

    print()
    print(
        "Per-split class balance: PASS"
    )

    # --------------------------------------------------------
    # Overall distribution
    # --------------------------------------------------------

    overall = Counter(
        record["label_name"]
        for record in selected
    )

    print()
    print(
        "Overall:"
    )

    for label in (
        "Genuine",
        "TTS",
        "Voice_Conversion",
    ):

        print(
            f"  {label:18s}: "
            f"{overall[label]}"
        )

    total = len(selected)

    print(
        f"  {'TOTAL':18s}: "
        f"{total}"
    )

    if not (
        overall["Genuine"]
        == overall["TTS"]
        == overall["Voice_Conversion"]
    ):

        raise RuntimeError(
            "Overall class balance failed."
        )

    print()
    print(
        "Overall class balance: PASS"
    )


# ============================================================
# Attack distribution report
# ============================================================

def print_attack_distribution(
    selected,
):
    print()
    print("=" * 70)
    print("Attack distribution in V3")
    print("=" * 70)

    for split in SPLITS:

        print()
        print(
            split.upper()
        )

        counts = Counter(
            record["attack_id"]
            for record in selected
            if record["split"] == split
        )

        for attack_id in sorted(
            counts
        ):

            print(
                f"  {attack_id}: "
                f"{counts[attack_id]}"
            )


# ============================================================
# Write manifest
# ============================================================

def write_manifest(
    records,
    path,
):
    fieldnames = [
        "audio_id",
        "audio_path",
        "speaker_id",
        "system_id",
        "attack_id",
        "label",
        "label_name",
        "label_id",
        "split",
    ]

    ordered = sorted(
        records,
        key=lambda record: (
            record["speaker_id"],
            record["audio_id"],
        ),
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in ordered:

            writer.writerow(
                {
                    field: record[field]
                    for field in fieldnames
                }
            )

    print(
        f"Created:\n{path}"
    )


# ============================================================
# Metadata
# ============================================================

def create_metadata(
    selected,
    speaker_to_split,
):
    summary = {}

    summary["source"] = {
        "dataset": "ASVspoof 2019 LA",
        "archive": str(LA_ZIP),
        "protocol": PROTOCOL_MEMBER,
        "audio_prefix": AUDIO_PREFIX,
    }

    summary["configuration"] = {
        "seed": SEED,
        "speaker_disjoint": True,
        "train_ratio": TRAIN_RATIO,
        "dev_ratio": DEV_RATIO,
        "test_ratio": TEST_RATIO,
        "classes": {
            "0": "Genuine",
            "1": "TTS",
            "2": "Voice Conversion",
        },
    }

    summary["total_samples"] = len(
        selected
    )

    summary["class_counts"] = dict(
        Counter(
            record["label_name"]
            for record in selected
        )
    )

    summary["split_counts"] = dict(
        Counter(
            record["split"]
            for record in selected
        )
    )

    summary["speaker_counts"] = {
        split: len(
            {
                record["speaker_id"]
                for record in selected
                if record["split"] == split
            }
        )
        for split in SPLITS
    }

    summary["attack_counts"] = dict(
        Counter(
            record["attack_id"]
            for record in selected
        )
    )

    summary["attack_counts_by_split"] = {}

    for split in SPLITS:

        summary[
            "attack_counts_by_split"
        ][split] = dict(
            Counter(
                record["attack_id"]
                for record in selected
                if record["split"] == split
            )
        )

    summary[
        "speaker_assignments"
    ] = speaker_to_split

    json_path = (
        METADATA_DIR
        / "v3_dataset_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    txt_path = (
        METADATA_DIR
        / "v3_dataset_summary.txt"
    )

    with txt_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "DeepShield V3 Dataset Summary\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Source: ASVspoof 2019 LA\n"
        )

        file.write(
            f"Archive: {LA_ZIP}\n\n"
        )

        file.write(
            "Classes:\n"
        )

        file.write(
            "  0 = Genuine\n"
        )

        file.write(
            "  1 = TTS\n"
        )

        file.write(
            "  2 = Voice Conversion\n\n"
        )

        file.write(
            "Total samples: "
            f"{len(selected)}\n\n"
        )

        file.write(
            "Class counts:\n"
        )

        for label in (
            "Genuine",
            "TTS",
            "Voice_Conversion",
        ):

            count = summary[
                "class_counts"
            ].get(
                label,
                0,
            )

            file.write(
                f"  {label}: {count}\n"
            )

        file.write(
            "\nSplit counts:\n"
        )

        for split in SPLITS:

            file.write(
                f"  {split}: "
                f"{summary['split_counts'].get(split, 0)}\n"
            )

        file.write(
            "\nSpeaker counts:\n"
        )

        for split in SPLITS:

            file.write(
                f"  {split}: "
                f"{summary['speaker_counts'][split]}\n"
            )

        file.write(
            "\nAttack counts:\n"
        )

        for attack_id in sorted(
            summary["attack_counts"]
        ):

            file.write(
                f"  {attack_id}: "
                f"{summary['attack_counts'][attack_id]}\n"
            )

        file.write(
            "\nDesign:\n"
        )

        file.write(
            "  Speaker-disjoint splits\n"
        )

        file.write(
            "  Per-split class balancing\n"
        )

        file.write(
            "  Deterministic sampling\n"
        )

        file.write(
            f"  Random seed = {SEED}\n"
        )

        file.write(
            "\nExisting V1/V2 datasets were not modified.\n"
        )

        file.write(
            "LA.zip was not modified.\n"
        )

    print()
    print(
        f"Created:\n{json_path}"
    )

    print(
        f"Created:\n{txt_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DeepShield V3 - Balanced Dataset Builder")
    print("=" * 70)

    print()
    print(
        f"LA archive:\n{LA_ZIP}"
    )

    if not LA_ZIP.exists():

        raise FileNotFoundError(
            f"LA.zip not found:\n{LA_ZIP}"
        )

    ensure_directories()

    # --------------------------------------------------------
    # Open archive
    # --------------------------------------------------------

    print()
    print(
        "Opening LA.zip..."
    )

    with zipfile.ZipFile(
        LA_ZIP,
        "r",
    ) as archive:

        archive_members = (
            archive.namelist()
        )

        if PROTOCOL_MEMBER not in (
            archive_members
        ):

            raise FileNotFoundError(
                "Training protocol not found:\n"
                f"{PROTOCOL_MEMBER}"
            )

        records = read_protocol(
            archive
        )

        validate_source(
            records,
            archive_members,
        )

    # --------------------------------------------------------
    # Speaker splits
    # --------------------------------------------------------

    speaker_to_split = (
        create_speaker_splits(
            records
        )
    )

    # --------------------------------------------------------
    # Enrich records
    # --------------------------------------------------------

    enriched = enrich_records(
        records,
        speaker_to_split,
    )

    # --------------------------------------------------------
    # Show availability
    # --------------------------------------------------------

    show_available_counts(
        enriched
    )

    # --------------------------------------------------------
    # Balance each split
    # --------------------------------------------------------

    selected = []

    for split in SPLITS:

        split_selected = (
            select_balanced_split(
                enriched,
                split,
            )
        )

        selected.extend(
            split_selected
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_selected(
        selected
    )

    print_attack_distribution(
        selected
    )

    # --------------------------------------------------------
    # Separate manifests
    # --------------------------------------------------------

    train_records = [
        record
        for record in selected
        if record["split"] == "train"
    ]

    dev_records = [
        record
        for record in selected
        if record["split"] == "dev"
    ]

    test_records = [
        record
        for record in selected
        if record["split"] == "test"
    ]

    # --------------------------------------------------------
    # Write manifests
    # --------------------------------------------------------

    write_manifest(
        train_records,
        MANIFEST_DIR
        / "v3_train.csv",
    )

    write_manifest(
        dev_records,
        MANIFEST_DIR
        / "v3_dev.csv",
    )

    write_manifest(
        test_records,
        MANIFEST_DIR
        / "v3_test.csv",
    )

    write_manifest(
        selected,
        MANIFEST_DIR
        / "v3_all.csv",
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    create_metadata(
        selected,
        speaker_to_split,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V3 DATASET CREATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Total samples: "
        f"{len(selected)}"
    )

    print(
        f"Train samples: "
        f"{len(train_records)}"
    )

    print(
        f"Dev samples:   "
        f"{len(dev_records)}"
    )

    print(
        f"Test samples:  "
        f"{len(test_records)}"
    )

    print()
    print(
        "Output:"
    )

    print(
        MANIFEST_DIR
    )

    print()
    print(
        "LA.zip was NOT modified."
    )

    print(
        "Existing V1/V2 datasets were NOT modified."
    )


if __name__ == "__main__":
    main()