"""
DeepShield V3.1 - Common Voice Audio Inspection

Purpose:
    Inspect a controlled sample of Common Voice MP3 files directly
    from the .tar.gz archive without extracting the dataset.

Project:
    C:\\DeepShield

Server:
    OFF

This script does NOT:
    - extract the full Common Voice archive
    - modify V3
    - modify the original ASVspoof dataset
    - train a model
    - create the final V3.1 manifest
"""

from __future__ import annotations

import csv
import io
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

import librosa
import numpy as np


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

OUTPUT_JSON = OUTPUT_DIR / "common_voice_audio_inspection.json"

SPLIT = "train.tsv"

# Deterministic seed so the same sample can be reproduced.
SEED = 42

# Number of speakers to inspect.
NUM_SPEAKERS = 50

# Maximum clips inspected per speaker.
MAX_CLIPS_PER_SPEAKER = 5

# Maximum total audio files inspected.
MAX_AUDIO_FILES = NUM_SPEAKERS * MAX_CLIPS_PER_SPEAKER

# V3 preprocessing target.
TARGET_SR = 16000

# V3 requires 4 seconds.
TARGET_SECONDS = 4.0

# Accept clips at least this long.
MIN_DURATION_SECONDS = 4.0


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


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), p))


# ============================================================
# READ TSV FROM ARCHIVE
# ============================================================

def read_metadata(tar: tarfile.TarFile) -> list[dict]:
    print_header("READING TRAIN METADATA")

    member = tar.getmember(SPLIT)

    rows = []

    with tar.extractfile(member) as raw:
        if raw is None:
            raise RuntimeError(f"Could not read {SPLIT}")

        text_stream = io.TextIOWrapper(
            raw,
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

    print(f"Metadata rows loaded : {len(rows):,}")

    return rows


# ============================================================
# SELECT CONTROLLED SAMPLE
# ============================================================

def select_sample(rows: list[dict]) -> list[dict]:
    print_header("SELECTING CONTROLLED AUDIO SAMPLE")

    by_speaker: dict[str, list[dict]] = {}

    for row in rows:
        speaker = (row.get("client_id") or "").strip()
        path = (row.get("path") or "").strip()

        if not speaker or not path:
            continue

        by_speaker.setdefault(speaker, []).append(row)

    print(f"Available speakers : {len(by_speaker):,}")

    rng = random.Random(SEED)

    speakers = sorted(by_speaker.keys())
    rng.shuffle(speakers)

    selected_speakers = speakers[:NUM_SPEAKERS]

    selected_rows = []

    for speaker in selected_speakers:
        clips = by_speaker[speaker]

        # Deterministically shuffle clips for this speaker.
        clips = list(clips)
        rng.shuffle(clips)

        selected_rows.extend(
            clips[:MAX_CLIPS_PER_SPEAKER]
        )

    print(f"Selected speakers  : {len(selected_speakers):,}")
    print(f"Selected clips     : {len(selected_rows):,}")

    return selected_rows


# ============================================================
# INSPECT AUDIO
# ============================================================

def inspect_audio(
    tar: tarfile.TarFile,
    row: dict,
) -> dict:

    client_id = (row.get("client_id") or "").strip()
    relative_path = (row.get("path") or "").strip()

    # Common Voice metadata contains:
    # clips/foo.mp3 inside the archive,
    # while TSV path is usually foo.mp3.
    archive_path = f"clips/{relative_path}"

    result = {
        "client_id": client_id,
        "path": relative_path,
        "archive_path": archive_path,
        "age": (row.get("age") or "").strip(),
        "accent": (row.get("accents") or "").strip(),
        "up_votes": safe_int(row.get("up_votes")),
        "down_votes": safe_int(row.get("down_votes")),
        "status": "UNKNOWN",
    }

    try:
        member = tar.getmember(archive_path)

    except KeyError:
        result["status"] = "MISSING_ARCHIVE_MEMBER"
        return result

    try:
        extracted = tar.extractfile(member)

        if extracted is None:
            result["status"] = "EXTRACT_FAILED"
            return result

        audio_bytes = extracted.read()

        # Decode directly from memory.
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=None,
            mono=False
        )

        # Determine original channel count.
        if audio.ndim == 1:
            channels = 1
            samples = len(audio)
        else:
            channels = audio.shape[0]
            samples = audio.shape[-1]

        duration = samples / float(sr)

        # Convert to mono only for statistics.
        if audio.ndim == 1:
            mono_audio = audio.astype(np.float32)
        else:
            mono_audio = np.mean(
                audio.astype(np.float32),
                axis=0
            )

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(mono_audio)
                )
            )
        )

        peak = float(
            np.max(
                np.abs(mono_audio)
            )
        )

        result.update({
            "status": "PASS" if duration >= MIN_DURATION_SECONDS else "TOO_SHORT",
            "sample_rate": int(sr),
            "channels": int(channels),
            "samples": int(samples),
            "duration_seconds": float(duration),
            "rms": rms,
            "peak": peak,
            "file_size_bytes": len(audio_bytes),
        })

    except Exception as exc:
        result["status"] = "DECODE_ERROR"
        result["error"] = str(exc)

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 75)
    print("DeepShield V3.1 - Common Voice Audio Inspection")
    print("=" * 75)

    print(f"Archive       : {ARCHIVE}")
    print(f"Split         : {SPLIT}")
    print(f"Seed          : {SEED}")
    print(f"Speakers      : {NUM_SPEAKERS}")
    print(f"Clips/speaker : {MAX_CLIPS_PER_SPEAKER}")
    print(f"Max clips     : {MAX_AUDIO_FILES:,}")

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

    print_header("OPENING COMMON VOICE ARCHIVE")

    with tarfile.open(
        ARCHIVE,
        mode="r:gz"
    ) as tar:

        print(f"Archive members: {len(tar.getmembers()):,}")

        rows = read_metadata(tar)

        selected = select_sample(rows)

        print_header("DECODING AUDIO")

        results = []

        for index, row in enumerate(
            selected,
            start=1
        ):

            result = inspect_audio(
                tar,
                row
            )

            results.append(result)

            status = result["status"]

            if status == "PASS":
                print(
                    f"[{index:03d}/{len(selected):03d}] "
                    f"PASS       "
                    f"{result['duration_seconds']:.2f}s "
                    f"{result['sample_rate']}Hz "
                    f"{result['channels']}ch"
                )

            elif status == "TOO_SHORT":
                print(
                    f"[{index:03d}/{len(selected):03d}] "
                    f"TOO SHORT  "
                    f"{result['duration_seconds']:.2f}s"
                )

            else:
                print(
                    f"[{index:03d}/{len(selected):03d}] "
                    f"{status}"
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_header("AUDIO INSPECTION SUMMARY")

    status_counts = Counter(
        r["status"]
        for r in results
    )

    durations = [
        r["duration_seconds"]
        for r in results
        if "duration_seconds" in r
    ]

    sample_rates = Counter(
        r["sample_rate"]
        for r in results
        if "sample_rate" in r
    )

    channels = Counter(
        r["channels"]
        for r in results
        if "channels" in r
    )

    rms_values = [
        r["rms"]
        for r in results
        if "rms" in r
    ]

    peak_values = [
        r["peak"]
        for r in results
        if "peak" in r
    ]

    print("Status:")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count:,}")

    print()
    print("Sample rates:")
    for sr, count in sample_rates.most_common():
        print(f"  {sr} Hz: {count:,}")

    print()
    print("Channels:")
    for ch, count in channels.most_common():
        print(f"  {ch}: {count:,}")

    if durations:
        print()
        print("Duration statistics:")
        print(f"  Minimum : {min(durations):.3f}s")
        print(f"  P05     : {percentile(durations, 5):.3f}s")
        print(f"  P25     : {percentile(durations, 25):.3f}s")
        print(f"  Median  : {percentile(durations, 50):.3f}s")
        print(f"  P75     : {percentile(durations, 75):.3f}s")
        print(f"  P95     : {percentile(durations, 95):.3f}s")
        print(f"  Maximum : {max(durations):.3f}s")

        usable = sum(
            d >= MIN_DURATION_SECONDS
            for d in durations
        )

        print()
        print(
            f"Clips >= {MIN_DURATION_SECONDS:.1f}s: "
            f"{usable:,}/{len(durations):,} "
            f"({usable / len(durations) * 100:.2f}%)"
        )

    if rms_values:
        print()
        print("RMS:")
        print(f"  Mean   : {np.mean(rms_values):.6f}")
        print(f"  Median : {np.median(rms_values):.6f}")
        print(f"  P05    : {percentile(rms_values, 5):.6f}")
        print(f"  P95    : {percentile(rms_values, 95):.6f}")

    if peak_values:
        print()
        print("Peak:")
        print(f"  Mean   : {np.mean(peak_values):.6f}")
        print(f"  Median : {np.median(peak_values):.6f}")
        print(f"  P05    : {percentile(peak_values, 5):.6f}")
        print(f"  P95    : {percentile(peak_values, 95):.6f}")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {
        "dataset": (
            "Common Voice Scripted Speech 26.0 "
            "American English Male"
        ),
        "archive": str(ARCHIVE),
        "split": SPLIT,
        "seed": SEED,
        "configuration": {
            "num_speakers": NUM_SPEAKERS,
            "max_clips_per_speaker": MAX_CLIPS_PER_SPEAKER,
            "max_audio_files": MAX_AUDIO_FILES,
            "target_sample_rate": TARGET_SR,
            "target_seconds": TARGET_SECONDS,
            "minimum_duration_seconds": MIN_DURATION_SECONDS,
        },
        "summary": {
            "selected_audio_files": len(results),
            "status_counts": dict(status_counts),
            "sample_rates": {
                str(k): v
                for k, v in sample_rates.items()
            },
            "channels": {
                str(k): v
                for k, v in channels.items()
            },
            "duration": {
                "minimum": min(durations) if durations else None,
                "p05": percentile(durations, 5),
                "p25": percentile(durations, 25),
                "median": percentile(durations, 50),
                "p75": percentile(durations, 75),
                "p95": percentile(durations, 95),
                "maximum": max(durations) if durations else None,
            },
            "usable_at_least_4_seconds": sum(
                d >= MIN_DURATION_SECONDS
                for d in durations
            ),
        },
        "files": results,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(f"Detailed results saved to:")
    print(OUTPUT_JSON)

    print_header("INSPECTION COMPLETE")

    print("COMMON VOICE AUDIO INSPECTION: PASS")
    print()
    print("The archive was NOT extracted.")
    print("The existing V3 was NOT modified.")
    print("No model was trained.")
    print()
    print("Next step: use these measurements to design")
    print("the controlled V3.1 Genuine subset.")


if __name__ == "__main__":
    main()