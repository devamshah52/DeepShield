from pathlib import Path
import csv
import io
import random
import zipfile

import librosa
import numpy as np


# ============================================================
# DeepShield V3
# ASVspoof Genuine vs Real-World Genuine Comparison
# ============================================================
#
# READ-ONLY DIAGNOSTIC
#
# Compares raw audio / raw Mel characteristics of:
#
#   A) Genuine ASVspoof 2019 LA training samples
#   B) Known genuine real-world recordings
#
# It reads ASVspoof FLAC files directly from LA.zip.
#
# It does NOT modify:
#   - LA.zip
#   - V3 manifests
#   - V3 feature cache
#   - V3 model
#   - predict.py
#
# ============================================================


# ============================================================
# Paths
# ============================================================

BACKEND_DIR = Path(
    r"C:\DeepShield\backend"
)

DATASET_DIR = (
    BACKEND_DIR
    / "ml"
    / "dataset"
)

LA_ZIP = (
    DATASET_DIR
    / "LA.zip"
)

PROTOCOL_FILE_IN_ZIP = (
    "LA/ASVspoof2019_LA_cm_protocols/"
    "ASVspoof2019.LA.cm.train.trn.txt"
)

REAL_AUDIO_DIR = Path(
    r"C:\DeepShield\exam_test_audio"
)


# ============================================================
# Real-world genuine files
# ============================================================

REAL_GENUINE_FILES = [
    "sample-12s.wav",
    "voice-sample-44100hz-16bit.flac",
    "voice-sample.mp3",
    "voice-sample.ogg",
    "Recording_test.wav",
]


# ============================================================
# Sampling configuration
# ============================================================

RANDOM_SEED = 42

# Number of Genuine ASVspoof samples to analyze.
# Enough for a representative diagnostic while keeping
# the script reasonably fast.
NUM_ASVSPOOF_SAMPLES = 300


# ============================================================
# V3 preprocessing configuration
# ============================================================

TARGET_SAMPLE_RATE = 16000
TARGET_DURATION = 4
TARGET_SAMPLES = (
    TARGET_SAMPLE_RATE
    * TARGET_DURATION
)

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128


# ============================================================
# Output
# ============================================================

OUTPUT_FILE = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3"
    / "evaluation"
    / "v3_asvspoof_vs_real_genuine.txt"
)


# ============================================================
# Helpers
# ============================================================

def separator(char="-", length=110):
    print(char * length)


def percentile(values, p):
    return float(
        np.percentile(
            np.asarray(values),
            p,
        )
    )


# ============================================================
# Audio standardization
# ============================================================

def standardize_audio(
    audio,
    sample_rate,
):
    """
    Convert audio to the exact V3 input format:

        16 kHz
        mono
        exactly 4 seconds
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError(
            "Audio contains no samples."
        )

    # Resample if necessary
    if sample_rate != TARGET_SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

    # Exactly 4 seconds
    if len(audio) > TARGET_SAMPLES:

        audio = audio[
            :TARGET_SAMPLES
        ]

    elif len(audio) < TARGET_SAMPLES:

        padding = (
            TARGET_SAMPLES
            - len(audio)
        )

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant",
        )

    return audio


# ============================================================
# Audio statistics
# ============================================================

def get_audio_stats(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    mean = float(
        np.mean(audio)
    )

    std = float(
        np.std(audio)
    )

    threshold_40 = (
        10 ** (-40 / 20)
    )

    threshold_50 = (
        10 ** (-50 / 20)
    )

    abs_audio = np.abs(audio)

    silence_40 = float(
        np.mean(
            abs_audio < threshold_40
        )
    )

    silence_50 = float(
        np.mean(
            abs_audio < threshold_50
        )
    )

    return {
        "rms": rms,
        "peak": peak,
        "mean": mean,
        "std": std,
        "silence_40": silence_40,
        "silence_50": silence_50,
    }


# ============================================================
# Raw Mel statistics
# ============================================================

def get_mel_stats(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    return {
        "mean": float(
            np.mean(mel_db)
        ),

        "std": float(
            np.std(mel_db)
        ),

        "min": float(
            np.min(mel_db)
        ),

        "max": float(
            np.max(mel_db)
        ),

        "p05": percentile(
            mel_db,
            5,
        ),

        "p25": percentile(
            mel_db,
            25,
        ),

        "p50": percentile(
            mel_db,
            50,
        ),

        "p75": percentile(
            mel_db,
            75,
        ),

        "p95": percentile(
            mel_db,
            95,
        ),
    }


# ============================================================
# Combined statistics
# ============================================================

def analyze_audio(
    audio,
    sample_rate,
):

    standardized = standardize_audio(
        audio,
        sample_rate,
    )

    audio_stats = get_audio_stats(
        standardized
    )

    mel_stats = get_mel_stats(
        standardized
    )

    return {
        **audio_stats,
        **{
            f"mel_{key}": value
            for key, value in mel_stats.items()
        },
    }


# ============================================================
# Read ASVspoof protocol
# ============================================================

def read_asvspoof_genuine_ids(
    archive,
):

    if PROTOCOL_FILE_IN_ZIP not in archive.namelist():

        raise FileNotFoundError(
            "ASVspoof training protocol not found "
            "inside LA.zip:\n"
            f"{PROTOCOL_FILE_IN_ZIP}"
        )

    with archive.open(
        PROTOCOL_FILE_IN_ZIP,
        "r",
    ) as f:

        text = (
            f.read()
            .decode(
                "utf-8",
                errors="replace",
            )
        )

    genuine_ids = []

    all_rows = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        speaker_id = parts[0]
        audio_id = parts[1]
        label = parts[-1]

        all_rows.append({
            "speaker_id": speaker_id,
            "audio_id": audio_id,
            "label": label,
            "parts": parts,
        })

        # ASVspoof LA protocol:
        # bonafide/genuine label is "bonafide"
        if label.lower() == "bonafide":

            genuine_ids.append({
                "speaker_id": speaker_id,
                "audio_id": audio_id,
            })

    return genuine_ids, all_rows


# ============================================================
# Locate FLAC inside ZIP
# ============================================================

def find_flac_member(
    archive_names,
    audio_id,
):

    target_filename = (
        f"{audio_id}.flac"
    ).lower()

    # Preferred ASVspoof LA training location
    preferred = (
        "LA/ASVspoof2019_LA_train/flac/"
        f"{audio_id}.flac"
    )

    if preferred in archive_names:
        return preferred

    # Fallback: search by exact filename
    for name in archive_names:

        if name.lower().endswith(
            "/" + target_filename
        ):
            return name

    return None


# ============================================================
# Main
# ============================================================

print()
print("=" * 110)
print(
    "DeepShield V3 - ASVspoof Genuine vs Real-World Genuine"
)
print("=" * 110)

print()
print("ASVspoof archive:")
print(LA_ZIP)

print()
print("Protocol:")
print(PROTOCOL_FILE_IN_ZIP)

print()
print("Real-world audio:")
print(REAL_AUDIO_DIR)


# ============================================================
# Verify files
# ============================================================

if not LA_ZIP.exists():

    raise FileNotFoundError(
        f"LA.zip not found:\n{LA_ZIP}"
    )

if not REAL_AUDIO_DIR.exists():

    raise FileNotFoundError(
        f"Real audio directory not found:\n"
        f"{REAL_AUDIO_DIR}"
    )


# ============================================================
# Open archive
# ============================================================

print()
print("=" * 110)
print("OPENING ASVSPOOF LA.ZIP")
print("=" * 110)

with zipfile.ZipFile(
    LA_ZIP,
    "r",
) as archive:

    archive_names = set(
        archive.namelist()
    )

    print(
        "Archive members:",
        len(archive_names),
    )

    # --------------------------------------------------------
    # Read protocol
    # --------------------------------------------------------

    print()
    print(
        "Reading Genuine entries..."
    )

    genuine_ids, all_rows = (
        read_asvspoof_genuine_ids(
            archive
        )
    )

    print(
        "Protocol rows:",
        len(all_rows),
    )

    print(
        "Genuine entries:",
        len(genuine_ids),
    )

    # --------------------------------------------------------
    # Deterministic sample
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    shuffled = list(
        genuine_ids
    )

    rng.shuffle(
        shuffled
    )

    selected = shuffled[
        :min(
            NUM_ASVSPOOF_SAMPLES,
            len(shuffled),
        )
    ]

    print(
        "Genuine samples selected:",
        len(selected),
    )

    # --------------------------------------------------------
    # Analyze ASVspoof samples
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "ANALYZING ASVSPOOF GENUINE AUDIO"
    )
    print("=" * 110)

    asvspoof_results = []

    missing_flac = 0
    failed_audio = 0

    for index, item in enumerate(
        selected,
        start=1,
    ):

        audio_id = item[
            "audio_id"
        ]

        speaker_id = item[
            "speaker_id"
        ]

        member = find_flac_member(
            archive_names,
            audio_id,
        )

        if member is None:

            missing_flac += 1
            continue

        try:

            audio_bytes = archive.read(
                member
            )

            audio, sr = librosa.load(
                io.BytesIO(
                    audio_bytes
                ),
                sr=None,
                mono=True,
            )

            stats = analyze_audio(
                audio,
                sr,
            )

            stats[
                "audio_id"
            ] = audio_id

            stats[
                "speaker_id"
            ] = speaker_id

            asvspoof_results.append(
                stats
            )

        except Exception as exc:

            failed_audio += 1

            print(
                f"Failed {audio_id}: {exc}"
            )

        if index % 50 == 0:

            print(
                f"Processed "
                f"{index}/{len(selected)}"
            )


# ============================================================
# ASVspoof aggregate
# ============================================================

print()
print("=" * 110)
print(
    "ASVSPOOF GENUINE RESULTS"
)
print("=" * 110)

print()
print(
    "Successful:",
    len(asvspoof_results),
)

print(
    "Missing FLAC:",
    missing_flac,
)

print(
    "Failed:",
    failed_audio,
)


if len(asvspoof_results) == 0:

    raise RuntimeError(
        "No ASVspoof genuine files could be analyzed."
    )


# ============================================================
# Aggregate function
# ============================================================

STAT_KEYS = [
    "rms",
    "peak",
    "silence_40",
    "silence_50",
    "mel_mean",
    "mel_std",
    "mel_min",
    "mel_max",
    "mel_p05",
    "mel_p25",
    "mel_p50",
    "mel_p75",
    "mel_p95",
]


def aggregate_results(
    results,
):

    output = {}

    for key in STAT_KEYS:

        values = np.asarray(
            [
                result[key]
                for result in results
            ],
            dtype=np.float64,
        )

        output[key] = {
            "mean": float(
                np.mean(values)
            ),

            "std": float(
                np.std(values)
            ),

            "min": float(
                np.min(values)
            ),

            "max": float(
                np.max(values)
            ),

            "p05": float(
                np.percentile(
                    values,
                    5,
                )
            ),

            "p50": float(
                np.percentile(
                    values,
                    50,
                )
            ),

            "p95": float(
                np.percentile(
                    values,
                    95,
                )
            ),
        }

    return output


asvspoof_aggregate = (
    aggregate_results(
        asvspoof_results
    )
)


# ============================================================
# Print ASVspoof aggregate
# ============================================================

print()

for key in STAT_KEYS:

    values = (
        asvspoof_aggregate[key]
    )

    print(
        f"{key:<20} "
        f"mean={values['mean']:.6f}  "
        f"std={values['std']:.6f}  "
        f"p05={values['p05']:.6f}  "
        f"p50={values['p50']:.6f}  "
        f"p95={values['p95']:.6f}"
    )


# ============================================================
# Analyze real-world genuine audio
# ============================================================

print()
print("=" * 110)
print(
    "ANALYZING REAL-WORLD GENUINE AUDIO"
)
print("=" * 110)

real_results = []

for filename in REAL_GENUINE_FILES:

    print()
    separator()

    print(
        "FILE:",
        filename,
    )

    file_path = (
        REAL_AUDIO_DIR
        / filename
    )

    if not file_path.exists():

        print(
            "STATUS: FILE NOT FOUND"
        )

        continue

    try:

        audio, sr = librosa.load(
            str(file_path),
            sr=None,
            mono=True,
        )

        stats = analyze_audio(
            audio,
            sr,
        )

        stats[
            "filename"
        ] = filename

        real_results.append(
            stats
        )

        print(
            "Original SR        :",
            sr,
            "Hz",
        )

        print(
            "Original Duration  :",
            f"{len(audio) / sr:.4f}",
            "sec",
        )

        print(
            "RMS                :",
            f"{stats['rms']:.6f}",
        )

        print(
            "Peak               :",
            f"{stats['peak']:.6f}",
        )

        print(
            "Silence < -40 dB  :",
            f"{stats['silence_40'] * 100:.2f}%",
        )

        print(
            "Silence < -50 dB  :",
            f"{stats['silence_50'] * 100:.2f}%",
        )

        print()
        print(
            "Raw Mel mean       :",
            f"{stats['mel_mean']:.6f}",
        )

        print(
            "Raw Mel std        :",
            f"{stats['mel_std']:.6f}",
        )

        print(
            "Raw Mel P05        :",
            f"{stats['mel_p05']:.6f}",
        )

        print(
            "Raw Mel P25        :",
            f"{stats['mel_p25']:.6f}",
        )

        print(
            "Raw Mel P50        :",
            f"{stats['mel_p50']:.6f}",
        )

        print(
            "Raw Mel P75        :",
            f"{stats['mel_p75']:.6f}",
        )

        print(
            "Raw Mel P95        :",
            f"{stats['mel_p95']:.6f}",
        )

        print(
            "STATUS: PASS"
        )

    except Exception as exc:

        print(
            "STATUS: ERROR"
        )

        print(
            "ERROR:",
            repr(exc),
        )


if len(real_results) == 0:

    raise RuntimeError(
        "No real-world genuine files could be analyzed."
    )


real_aggregate = (
    aggregate_results(
        real_results
    )
)


# ============================================================
# Direct comparison
# ============================================================

print()
print("=" * 110)
print(
    "ASVSPOOF GENUINE vs REAL-WORLD GENUINE"
)
print("=" * 110)

print()
print(
    "ASVspoof values are aggregate statistics across "
    f"{len(asvspoof_results)} Genuine samples."
)

print(
    "Real-world values are individual recordings."
)

print()

print(
    f"{'Statistic':25} "
    f"{'ASV mean':>12} "
    f"{'ASV p05':>12} "
    f"{'ASV p50':>12} "
    f"{'ASV p95':>12}"
)

separator("-", 80)

comparison_keys = [
    ("RMS", "rms"),
    ("Peak", "peak"),
    ("Silence < -40 dB", "silence_40"),
    ("Silence < -50 dB", "silence_50"),
    ("Raw Mel mean", "mel_mean"),
    ("Raw Mel std", "mel_std"),
    ("Raw Mel P05", "mel_p05"),
    ("Raw Mel P25", "mel_p25"),
    ("Raw Mel P50", "mel_p50"),
    ("Raw Mel P75", "mel_p75"),
    ("Raw Mel P95", "mel_p95"),
]

for display_name, key in comparison_keys:

    values = (
        asvspoof_aggregate[key]
    )

    print(
        f"{display_name:25} "
        f"{values['mean']:12.4f} "
        f"{values['p05']:12.4f} "
        f"{values['p50']:12.4f} "
        f"{values['p95']:12.4f}"
    )


# ============================================================
# Real-world comparison table
# ============================================================

print()
print("=" * 110)
print(
    "REAL-WORLD FILES AGAINST ASVSPOOF DISTRIBUTION"
)
print("=" * 110)

print()

print(
    f"{'File':38} "
    f"{'RMS':>9} "
    f"{'Mel Mean':>11} "
    f"{'Mel P50':>11} "
    f"{'Silence40':>11}"
)

separator("-", 90)

for result in real_results:

    print(
        f"{result['filename'][:38]:38} "
        f"{result['rms']:9.4f} "
        f"{result['mel_mean']:11.2f} "
        f"{result['mel_p50']:11.2f} "
        f"{result['silence_40'] * 100:10.2f}%"
    )


# ============================================================
# Automatic overlap indicators
# ============================================================

print()
print("=" * 110)
print(
    "DISTRIBUTION OVERLAP INDICATORS"
)
print("=" * 110)

print()
print(
    "These are descriptive indicators only."
)

print(
    "They do NOT determine whether an audio file is genuine "
    "or synthetic."
)


def inside_range(
    value,
    aggregate,
):

    return (
        aggregate["p05"]
        <= value
        <= aggregate["p95"]
    )


for result in real_results:

    print()
    print(
        result["filename"]
    )

    checks = [
        (
            "RMS",
            result["rms"],
            asvspoof_aggregate["rms"],
        ),

        (
            "Peak",
            result["peak"],
            asvspoof_aggregate["peak"],
        ),

        (
            "Silence < -40 dB",
            result["silence_40"],
            asvspoof_aggregate[
                "silence_40"
            ],
        ),

        (
            "Raw Mel mean",
            result["mel_mean"],
            asvspoof_aggregate[
                "mel_mean"
            ],
        ),

        (
            "Raw Mel P50",
            result["mel_p50"],
            asvspoof_aggregate[
                "mel_p50"
            ],
        ),
    ]

    for name, value, aggregate in checks:

        status = (
            "INSIDE ASV RANGE"
            if inside_range(
                value,
                aggregate,
            )
            else "OUTSIDE ASV RANGE"
        )

        print(
            f"{name:<24}: "
            f"{value:10.4f}  "
            f"{status}"
        )


# ============================================================
# Save report
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "DeepShield V3 - ASVspoof Genuine vs Real-World Genuine\n"
    )

    report.write(
        "=" * 100
        + "\n\n"
    )

    report.write(
        f"ASVspoof samples analyzed: "
        f"{len(asvspoof_results)}\n"
    )

    report.write(
        f"Missing FLAC: {missing_flac}\n"
    )

    report.write(
        f"Failed audio: {failed_audio}\n\n"
    )

    report.write(
        "ASVSPOOF AGGREGATE\n"
    )

    report.write(
        "-" * 80
        + "\n"
    )

    for key in STAT_KEYS:

        values = (
            asvspoof_aggregate[key]
        )

        report.write(
            f"{key}: "
            f"mean={values['mean']}, "
            f"std={values['std']}, "
            f"p05={values['p05']}, "
            f"p50={values['p50']}, "
            f"p95={values['p95']}\n"
        )

    report.write(
        "\nREAL-WORLD FILES\n"
    )

    report.write(
        "-" * 80
        + "\n"
    )

    for result in real_results:

        report.write(
            "\n"
            + result["filename"]
            + "\n"
        )

        for key, value in result.items():

            if key != "filename":

                report.write(
                    f"{key}: {value}\n"
                )


# ============================================================
# Complete
# ============================================================

print()
print("=" * 110)
print(
    "REPORT SAVED"
)
print("=" * 110)

print(
    OUTPUT_FILE
)

print()
print(
    "ASVSPOOF vs REAL-WORLD GENUINE COMPARISON COMPLETE"
)

print("=" * 110)