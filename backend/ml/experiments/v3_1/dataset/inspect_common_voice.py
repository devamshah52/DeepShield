"""
DeepShield V3.1 - Common Voice 26 American English Male Dataset Inspection

Purpose:
    Inspect Common Voice metadata directly from the .tar.gz archive
    without extracting the 10+ GB dataset.

Project:
    C:\DeepShield

Server:
    OFF

This script does NOT:
    - extract the archive
    - modify V3
    - modify the original ASVspoof dataset
    - train any model
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ARCHIVE = Path(
    r"C:\DeepShield\external_datasets"
    r"\\1784540315873-cv26-american-english-male.tar.gz"
)

OUTPUT_DIR = Path(
    r"C:\DeepShield\backend\ml\experiments\v3_1\dataset\metadata"
)

TSV_FILES = ["train.tsv", "dev.tsv", "test.tsv"]


# ============================================================
# HELPERS
# ============================================================

def safe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def print_header(title: str) -> None:
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def inspect_split(tar: tarfile.TarFile, split_name: str) -> dict:
    print_header(f"INSPECTING {split_name.upper()}")

    member = tar.getmember(split_name)

    speakers = set()
    clips_per_speaker = Counter()

    ages = Counter()
    accents = Counter()
    genders = Counter()
    locales = Counter()

    vote_up = 0
    vote_down = 0

    rows = 0
    empty_client_id = 0
    empty_path = 0
    empty_sentence = 0

    min_votes = None
    max_votes = None

    with tar.extractfile(member) as raw:
        if raw is None:
            raise RuntimeError(f"Could not read {split_name}")

        text_stream = io.TextIOWrapper(
            raw,
            encoding="utf-8",
            errors="replace",
            newline=""
        )

        reader = csv.DictReader(text_stream, delimiter="\t")

        columns = reader.fieldnames or []

        print(f"Columns: {columns}")

        required = ["client_id", "path", "sentence"]
        missing = [c for c in required if c not in columns]

        if missing:
            raise RuntimeError(
                f"{split_name}: missing required columns: {missing}"
            )

        for row in reader:
            rows += 1

            client_id = normalize(row.get("client_id"))
            path = normalize(row.get("path"))
            sentence = normalize(row.get("sentence"))

            age = normalize(row.get("age"))
            gender = normalize(row.get("gender"))
            accent = normalize(row.get("accents"))
            locale = normalize(row.get("locale"))

            up = safe_int(row.get("up_votes", "0"))
            down = safe_int(row.get("down_votes", "0"))

            if client_id:
                speakers.add(client_id)
                clips_per_speaker[client_id] += 1
            else:
                empty_client_id += 1

            if not path:
                empty_path += 1

            if not sentence:
                empty_sentence += 1

            ages[age or "<empty>"] += 1
            genders[gender or "<empty>"] += 1
            accents[accent or "<empty>"] += 1
            locales[locale or "<empty>"] += 1

            vote_up += up
            vote_down += down

            total_votes = up + down

            if min_votes is None or total_votes < min_votes:
                min_votes = total_votes

            if max_votes is None or total_votes > max_votes:
                max_votes = total_votes

    result = {
        "split": split_name,
        "rows": rows,
        "unique_speakers": len(speakers),
        "empty_client_id": empty_client_id,
        "empty_path": empty_path,
        "empty_sentence": empty_sentence,
        "age_distribution": dict(ages),
        "gender_distribution": dict(genders),
        "accent_distribution": dict(accents),
        "locale_distribution": dict(locales),
        "total_up_votes": vote_up,
        "total_down_votes": vote_down,
        "minimum_total_votes": min_votes,
        "maximum_total_votes": max_votes,
        "speaker_clip_statistics": {
            "minimum": min(clips_per_speaker.values())
            if clips_per_speaker else 0,
            "maximum": max(clips_per_speaker.values())
            if clips_per_speaker else 0,
            "average": (
                sum(clips_per_speaker.values()) / len(clips_per_speaker)
                if clips_per_speaker else 0
            ),
        },
        "_speaker_ids": sorted(speakers),
    }

    print(f"Rows/clips       : {rows:,}")
    print(f"Unique speakers   : {len(speakers):,}")
    print(f"Empty client IDs  : {empty_client_id:,}")
    print(f"Empty paths       : {empty_path:,}")
    print(f"Empty sentences   : {empty_sentence:,}")

    print()
    print("Age distribution:")
    for key, value in ages.most_common():
        print(f"  {key}: {value:,}")

    print()
    print("Gender distribution:")
    for key, value in genders.most_common():
        print(f"  {key}: {value:,}")

    print()
    print("Top accents:")
    for key, value in accents.most_common(20):
        print(f"  {key or '<empty>'}: {value:,}")

    print()
    print("Locales:")
    for key, value in locales.most_common():
        print(f"  {key or '<empty>'}: {value:,}")

    print()
    print("Votes:")
    print(f"  Up votes   : {vote_up:,}")
    print(f"  Down votes : {vote_down:,}")

    print()
    print("Clips per speaker:")
    print(
        f"  Minimum : {result['speaker_clip_statistics']['minimum']:,}"
    )
    print(
        f"  Maximum : {result['speaker_clip_statistics']['maximum']:,}"
    )
    print(
        f"  Average : "
        f"{result['speaker_clip_statistics']['average']:.2f}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 75)
    print("DeepShield V3.1 - Common Voice 26 Dataset Inspection")
    print("=" * 75)

    print(f"Archive : {ARCHIVE}")
    print(f"Output  : {OUTPUT_DIR}")

    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Archive not found:\n{ARCHIVE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    archive_size_gb = ARCHIVE.stat().st_size / (1024 ** 3)

    print(f"Archive size: {archive_size_gb:.2f} GB")
    print()
    print("Extraction: NO")
    print("Training:   NO")
    print("V3 changes: NO")

    all_results = {}

    print_header("OPENING ARCHIVE")

    with tarfile.open(ARCHIVE, mode="r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}

        print(f"Archive members: {len(members):,}")

        missing = [name for name in TSV_FILES if name not in members]

        if missing:
            raise RuntimeError(
                f"Missing metadata files: {missing}"
            )

        # Inspect each split.
        for split_name in TSV_FILES:
            all_results[split_name] = inspect_split(
                tar,
                split_name
            )

    # ========================================================
    # SPEAKER OVERLAP
    # ========================================================

    print_header("SPEAKER OVERLAP CHECK")

    speaker_sets = {
        split: set(result["_speaker_ids"])
        for split, result in all_results.items()
    }

    train_speakers = speaker_sets["train.tsv"]
    dev_speakers = speaker_sets["dev.tsv"]
    test_speakers = speaker_sets["test.tsv"]

    train_dev = train_speakers & dev_speakers
    train_test = train_speakers & test_speakers
    dev_test = dev_speakers & test_speakers

    print(f"Train speakers : {len(train_speakers):,}")
    print(f"Dev speakers   : {len(dev_speakers):,}")
    print(f"Test speakers  : {len(test_speakers):,}")

    print()
    print(f"Train ∩ Dev    : {len(train_dev):,}")
    print(f"Train ∩ Test   : {len(train_test):,}")
    print(f"Dev ∩ Test     : {len(dev_test):,}")

    if not train_dev and not train_test and not dev_test:
        print()
        print("SPEAKER SPLIT STATUS: PASS")
    else:
        print()
        print("SPEAKER SPLIT STATUS: OVERLAP DETECTED")

    # ========================================================
    # TOTALS
    # ========================================================

    total_clips = sum(
        result["rows"]
        for result in all_results.values()
    )

    total_speakers = len(
        train_speakers | dev_speakers | test_speakers
    )

    print_header("COMMON VOICE TOTALS")

    print(f"Total clips across splits : {total_clips:,}")
    print(f"Total unique speakers     : {total_speakers:,}")

    # ========================================================
    # SAVE JSON
    # ========================================================

    # Remove internal speaker lists from the main summary.
    clean_results = {}

    for split, result in all_results.items():
        clean_results[split] = {
            key: value
            for key, value in result.items()
            if key != "_speaker_ids"
        }

    summary = {
        "dataset": "Common Voice Scripted Speech 26.0 - American English Male",
        "archive": str(ARCHIVE),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_size_gb": archive_size_gb,
        "splits": clean_results,
        "totals": {
            "clips": total_clips,
            "unique_speakers": total_speakers,
        },
        "speaker_overlap": {
            "train_dev": len(train_dev),
            "train_test": len(train_test),
            "dev_test": len(dev_test),
            "status": (
                "PASS"
                if not train_dev and
                not train_test and
                not dev_test
                else "OVERLAP_DETECTED"
            ),
        },
    }

    output_json = OUTPUT_DIR / "common_voice_26_american_male_summary.json"

    with output_json.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(f"Summary saved:")
    print(output_json)

    # ========================================================
    # FINAL
    # ========================================================

    print_header("INSPECTION COMPLETE")

    print("COMMON VOICE INSPECTION: PASS")
    print()
    print("No files were extracted.")
    print("No V3 files were modified.")
    print("No model was trained.")
    print()
    print("Next step: choose a controlled speaker-disjoint")
    print("Common Voice Genuine subset for V3.1.")


if __name__ == "__main__":
    main()