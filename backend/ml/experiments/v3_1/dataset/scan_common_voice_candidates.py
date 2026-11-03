"""
DeepShield V3.1 - Fast Common Voice Candidate Pool Scan

Reads Common Voice metadata and audio directly from the .tar.gz archive.

IMPORTANT:
    - The 9.68 GB archive is NOT extracted.
    - Existing V3 is NOT modified.
    - No model training.
    - Server should be OFF.

Optimization:
    The archive member names are indexed ONCE.
    We do not repeatedly call tar.getmember() for every audio file.
"""

from __future__ import annotations

import csv
import io
import json
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import librosa


# ============================================================
# CONFIGURATION
# ============================================================

ARCHIVE = Path(
    r"C:\DeepShield\external_datasets"
    r"\1784540315873-cv26-american-english-male.tar.gz"
)

OUTPUT_DIR = Path(
    r"C:\DeepShield\backend\ml\experiments\v3_1\dataset\metadata"
)

OUTPUT_JSON = (
    OUTPUT_DIR /
    "common_voice_candidate_pool_scan.json"
)

SPLIT = "train.tsv"

SEED = 42

# Candidate speakers.
CANDIDATE_SPEAKERS = 400

# Maximum audio files checked per speaker.
MAX_AUDIO_CHECKS_PER_SPEAKER = 20

# Minimum duration required by our V3 preprocessing.
MIN_DURATION_SECONDS = 4.0

# Desired Common Voice Genuine samples later.
TARGET_COMMON_VOICE = 1800


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def safe_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# ============================================================
# LOAD TSV
# ============================================================

def load_metadata(
    tar: tarfile.TarFile
) -> list[dict]:

    print_header("READING TRAIN.TSV")

    member = tar.getmember(SPLIT)

    rows = []

    extracted = tar.extractfile(member)

    if extracted is None:
        raise RuntimeError(
            "Could not read train.tsv"
        )

    with extracted:
        text_stream = io.TextIOWrapper(
            extracted,
            encoding="utf-8",
            errors="replace",
            newline=""
        )

        reader = csv.DictReader(
            text_stream,
            delimiter="\t"
        )

        for row in reader:
            rows.append(row)

    print(
        f"Metadata rows loaded: "
        f"{len(rows):,}"
    )

    return rows


# ============================================================
# BUILD SPEAKER INDEX
# ============================================================

def build_speaker_index(
    rows: list[dict]
) -> dict[str, list[dict]]:

    print_header("BUILDING SPEAKER INDEX")

    by_speaker: dict[str, list[dict]] = defaultdict(list)

    for row in rows:

        client_id = (
            row.get("client_id") or ""
        ).strip()

        path = (
            row.get("path") or ""
        ).strip()

        if not client_id or not path:
            continue

        by_speaker[client_id].append(row)

    print(
        f"Unique speakers: "
        f"{len(by_speaker):,}"
    )

    return dict(by_speaker)


# ============================================================
# BUILD ARCHIVE INDEX
# ============================================================

def build_archive_index(
    tar: tarfile.TarFile
) -> dict[str, tarfile.TarInfo]:

    print_header("BUILDING ARCHIVE MEMBER INDEX")

    archive_index = {}

    count = 0

    for member in tar:

        count += 1

        # We only need MP3 files.
        if (
            member.isfile()
            and member.name.startswith("clips/")
            and member.name.lower().endswith(".mp3")
        ):
            archive_index[member.name] = member

        if count % 50000 == 0:
            print(
                f"Archive members indexed: "
                f"{count:,}"
            )

    print()
    print(
        f"Total archive members visited: "
        f"{count:,}"
    )

    print(
        f"MP3 members indexed: "
        f"{len(archive_index):,}"
    )

    return archive_index


# ============================================================
# SELECT SPEAKERS
# ============================================================

def select_speakers(
    by_speaker: dict[str, list[dict]]
) -> list[str]:

    print_header("SELECTING CANDIDATE SPEAKERS")

    rng = random.Random(SEED)

    speakers = list(by_speaker.keys())

    rng.shuffle(speakers)

    selected = speakers[
        :CANDIDATE_SPEAKERS
    ]

    print(
        f"Candidate speakers: "
        f"{len(selected):,}"
    )

    return selected


# ============================================================
# INSPECT ONE AUDIO FILE
# ============================================================

def inspect_audio(
    tar: tarfile.TarFile,
    archive_index: dict[str, tarfile.TarInfo],
    row: dict
) -> dict:

    client_id = (
        row.get("client_id") or ""
    ).strip()

    path = (
        row.get("path") or ""
    ).strip()

    archive_path = (
        f"clips/{path}"
    )

    result = {
        "client_id": client_id,
        "path": path,
        "archive_path": archive_path,
        "age": (
            row.get("age") or ""
        ).strip(),
        "accent": (
            row.get("accents") or ""
        ).strip(),
        "up_votes": safe_int(
            row.get("up_votes")
        ),
        "down_votes": safe_int(
            row.get("down_votes")
        ),
        "status": "UNKNOWN",
    }

    member = archive_index.get(
        archive_path
    )

    if member is None:
        result["status"] = "MISSING"
        return result

    try:

        extracted = tar.extractfile(
            member
        )

        if extracted is None:
            result["status"] = "READ_ERROR"
            return result

        with extracted:
            audio_bytes = extracted.read()

        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=None,
            mono=False
        )

        if audio.ndim == 1:
            samples = len(audio)
            channels = 1
        else:
            samples = audio.shape[-1]
            channels = audio.shape[0]

        duration = (
            samples / float(sr)
        )

        result.update({
            "status": (
                "PASS"
                if duration >= MIN_DURATION_SECONDS
                else "TOO_SHORT"
            ),
            "sample_rate": int(sr),
            "channels": int(channels),
            "samples": int(samples),
            "duration_seconds": float(
                duration
            ),
        })

        return result

    except Exception as exc:

        result["status"] = (
            "DECODE_ERROR"
        )

        result["error"] = str(exc)

        return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 75)
    print(
        "DeepShield V3.1 - Fast Common Voice "
        "Candidate Pool Scan"
    )
    print("=" * 75)

    print(
        f"Archive       : {ARCHIVE}"
    )

    print(
        f"Split         : {SPLIT}"
    )

    print(
        f"Seed          : {SEED}"
    )

    print(
        f"Candidate speakers : "
        f"{CANDIDATE_SPEAKERS}"
    )

    print(
        f"Max clips/speaker : "
        f"{MAX_AUDIO_CHECKS_PER_SPEAKER}"
    )

    print(
        f"Minimum duration : "
        f"{MIN_DURATION_SECONDS:.1f}s"
    )

    print()
    print("Extraction: NO")
    print("Training:   NO")
    print("V3 changes: NO")

    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Archive not found:\n{ARCHIVE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # OPEN ARCHIVE
    # ========================================================

    print_header("OPENING ARCHIVE")

    with tarfile.open(
        ARCHIVE,
        mode="r:gz"
    ) as tar:

        print(
            "Reading metadata and indexing "
            "archive members..."
        )

        rows = load_metadata(tar)

        by_speaker = (
            build_speaker_index(rows)
        )

        selected_speakers = (
            select_speakers(by_speaker)
        )

        # IMPORTANT:
        # Build archive index once.
        archive_index = (
            build_archive_index(tar)
        )

        # ====================================================
        # AUDIO SCAN
        # ====================================================

        print_header(
            "SCANNING CANDIDATE AUDIO"
        )

        speaker_results = {}

        total_checked = 0
        total_pass = 0
        total_short = 0
        total_errors = 0

        sample_rates = Counter()
        channels = Counter()

        durations = []

        rng = random.Random(SEED)

        for speaker_index, speaker in enumerate(
            selected_speakers,
            start=1
        ):

            clips = list(
                by_speaker[speaker]
            )

            rng.shuffle(clips)

            clips = clips[
                :MAX_AUDIO_CHECKS_PER_SPEAKER
            ]

            speaker_results[speaker] = {
                "metadata_clip_count":
                    len(by_speaker[speaker]),

                "checked_clip_count":
                    len(clips),

                "usable_clip_count":
                    0,

                "too_short_count":
                    0,

                "error_count":
                    0,

                "clips":
                    [],
            }

            for row in clips:

                result = inspect_audio(
                    tar,
                    archive_index,
                    row
                )

                speaker_results[
                    speaker
                ]["clips"].append(result)

                total_checked += 1

                status = result["status"]

                if status == "PASS":

                    total_pass += 1

                    speaker_results[
                        speaker
                    ]["usable_clip_count"] += 1

                elif status == "TOO_SHORT":

                    total_short += 1

                    speaker_results[
                        speaker
                    ]["too_short_count"] += 1

                else:

                    total_errors += 1

                    speaker_results[
                        speaker
                    ]["error_count"] += 1

                if "duration_seconds" in result:

                    durations.append(
                        result[
                            "duration_seconds"
                        ]
                    )

                if "sample_rate" in result:

                    sample_rates[
                        result["sample_rate"]
                    ] += 1

                if "channels" in result:

                    channels[
                        result["channels"]
                    ] += 1

            # Progress every 25 speakers.
            if (
                speaker_index % 25 == 0
                or
                speaker_index ==
                len(selected_speakers)
            ):

                print(
                    f"Speakers processed: "
                    f"{speaker_index}/"
                    f"{len(selected_speakers)} | "
                    f"audio checked: "
                    f"{total_checked:,} | "
                    f"usable: "
                    f"{total_pass:,}"
                )

    # ========================================================
    # SPEAKER SUMMARY
    # ========================================================

    print_header(
        "SPEAKER-LEVEL SUMMARY"
    )

    usable_counts = [
        data["usable_clip_count"]
        for data in speaker_results.values()
    ]

    for threshold in [5, 10, 15, 20]:

        count = sum(
            x >= threshold
            for x in usable_counts
        )

        print(
            f"Speakers with >= {threshold} "
            f"usable clips : {count:,}"
        )

    # ========================================================
    # AUDIO SUMMARY
    # ========================================================

    print_header("AUDIO SUMMARY")

    print(
        f"Audio files checked : "
        f"{total_checked:,}"
    )

    print(
        f"Usable >= 4 sec     : "
        f"{total_pass:,}"
    )

    print(
        f"Too short           : "
        f"{total_short:,}"
    )

    print(
        f"Errors              : "
        f"{total_errors:,}"
    )

    if total_checked:

        print(
            f"Usable percentage   : "
            f"{total_pass / total_checked * 100:.2f}%"
        )

    print()
    print("Sample rates:")

    for sr, count in (
        sample_rates.most_common()
    ):

        print(
            f"  {sr} Hz: {count:,}"
        )

    print()
    print("Channels:")

    for ch, count in (
        channels.most_common()
    ):

        print(
            f"  {ch}: {count:,}"
        )

    if durations:

        durations.sort()

        def percentile(
            p: float
        ) -> float:

            index = int(
                (len(durations) - 1) * p
            )

            return durations[index]

        print()
        print(
            "Duration statistics:"
        )

        print(
            f"  Minimum : "
            f"{min(durations):.3f}s"
        )

        print(
            f"  P05     : "
            f"{percentile(0.05):.3f}s"
        )

        print(
            f"  P25     : "
            f"{percentile(0.25):.3f}s"
        )

        print(
            f"  Median  : "
            f"{percentile(0.50):.3f}s"
        )

        print(
            f"  P75     : "
            f"{percentile(0.75):.3f}s"
        )

        print(
            f"  P95     : "
            f"{percentile(0.95):.3f}s"
        )

        print(
            f"  Maximum : "
            f"{max(durations):.3f}s"
        )

    # ========================================================
    # CAPACITY ESTIMATES
    # ========================================================

    print_header(
        "V3.1 SUBSET CAPACITY"
    )

    capacities = {}

    for max_per_speaker in [
        5,
        10,
        15,
        20,
    ]:

        capacity = sum(
            min(
                count,
                max_per_speaker
            )
            for count in usable_counts
        )

        capacities[
            str(max_per_speaker)
        ] = capacity

        print(
            f"Max {max_per_speaker:2d}/speaker : "
            f"{capacity:,} usable clips"
        )

    print()
    print(
        f"Target Common Voice Genuine : "
        f"{TARGET_COMMON_VOICE:,}"
    )

    possible = any(
        capacity >= TARGET_COMMON_VOICE
        for capacity in capacities.values()
    )

    print(
        "Target achievable from "
        "candidate pool: "
        f"{'YES' if possible else 'NO'}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    summary = {
        "dataset": (
            "Common Voice Scripted Speech 26.0 "
            "American English Male"
        ),

        "archive": str(ARCHIVE),

        "split": SPLIT,

        "seed": SEED,

        "configuration": {
            "candidate_speakers":
                CANDIDATE_SPEAKERS,

            "max_audio_checks_per_speaker":
                MAX_AUDIO_CHECKS_PER_SPEAKER,

            "minimum_duration_seconds":
                MIN_DURATION_SECONDS,

            "target_common_voice":
                TARGET_COMMON_VOICE,
        },

        "summary": {
            "candidate_speakers":
                len(selected_speakers),

            "audio_checked":
                total_checked,

            "usable_audio":
                total_pass,

            "too_short":
                total_short,

            "errors":
                total_errors,

            "usable_percentage":
                (
                    total_pass /
                    total_checked *
                    100
                    if total_checked
                    else 0
                ),

            "sample_rates": {
                str(k): v
                for k, v in
                sample_rates.items()
            },

            "channels": {
                str(k): v
                for k, v in
                channels.items()
            },

            "capacity": capacities,
        },

        "speaker_results":
            speaker_results,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "Results saved:"
    )

    print(
        OUTPUT_JSON
    )

    print_header(
        "SCAN COMPLETE"
    )

    print(
        "COMMON VOICE CANDIDATE SCAN: PASS"
    )

    print()
    print(
        "Archive extracted : NO"
    )

    print(
        "V3 modified       : NO"
    )

    print(
        "Model trained     : NO"
    )

    print()
    print(
        "Next step: construct the "
        "controlled V3.1 Genuine manifest."
    )


if __name__ == "__main__":
    main()