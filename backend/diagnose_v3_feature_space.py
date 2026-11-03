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

REAL_AUDIO_DIR = Path(
    r"C:\DeepShield\exam_test_audio"
)

OUTPUT_CSV = (
    V3_ROOT
    / "evaluation"
    / "v3_feature_space_diagnostic.csv"
)

SEED = 42

# Number of correctly classified ASVspoof
# Genuine samples to compare.
NUM_ASVSPOOF_GENUINE = 50

# ============================================================
# V3 AUDIO CONFIG
# ============================================================

SAMPLE_RATE = 16000
DURATION = 4.0
NUM_SAMPLES = int(
    SAMPLE_RATE * DURATION
)

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

EXPECTED_FEATURE_SHAPE = (
    128,
    251,
)

CLASS_NAMES = {
    0: "Genuine",
    1: "TTS",
    2: "Voice_Conversion",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL IMPORT
# ============================================================

sys.path.insert(
    0,
    str(V3_ROOT),
)

from model.v3_model import DeepShieldV3


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print(
    "DeepShield V3 — Feature Space / Spectral Diagnostic"
)
print("=" * 80)

print(
    f"Device     : {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU        : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    f"Checkpoint : {CHECKPOINT_PATH}"
)

print(
    f"Manifest   : {MANIFEST_PATH}"
)

print(
    f"Real audio : {REAL_AUDIO_DIR}"
)

print("=" * 80)


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

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = (
        checkpoint["model_state_dict"]
    )

else:

    state_dict = checkpoint

model.load_state_dict(
    state_dict
)

model = model.to(
    DEVICE
)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# AUDIO DECODER
# ============================================================

def load_audio_file(
    path: Path,
):

    audio, sr = librosa.load(
        str(path),
        sr=None,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:

        raise ValueError(
            "Empty audio"
        )

    return audio, int(sr)


# ============================================================
# ZIP AUDIO LOADER
# ============================================================

def load_zip_audio(
    zf,
    member_path,
):

    audio_bytes = zf.read(
        member_path
    )

    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=None,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:

        raise ValueError(
            "Empty audio"
        )

    return audio, int(sr)


# ============================================================
# RESAMPLE
# ============================================================

def resample_audio(
    audio,
    sr,
):

    if sr == SAMPLE_RATE:

        return audio.astype(
            np.float32
        )

    return librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=SAMPLE_RATE,
    ).astype(
        np.float32
    )


# ============================================================
# FIXED 4 SECOND AUDIO
# ============================================================

def prepare_audio(
    audio,
):

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

        audio = audio[
            :NUM_SAMPLES
        ]

    return audio.astype(
        np.float32
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    audio,
):

    audio = prepare_audio(
        audio
    )

    # --------------------------------------------------------
    # Mel spectrogram
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
    # V3 normalization
    # --------------------------------------------------------

    mel_mean = float(
        mel_db.mean()
    )

    mel_std = float(
        mel_db.std()
    )

    if mel_std < 1e-8:

        raise ValueError(
            "Mel standard deviation too small"
        )

    normalized = (
        mel_db - mel_mean
    ) / mel_std

    normalized = np.asarray(
        normalized,
        dtype=np.float32,
    )

    if normalized.shape != EXPECTED_FEATURE_SHAPE:

        raise ValueError(
            "Unexpected feature shape: "
            f"{normalized.shape}"
        )

    return (
        normalized,
        mel_db,
    )


# ============================================================
# SPECTRAL STATISTICS
# ============================================================

def calculate_statistics(
    audio,
    mel_db,
):

    audio = prepare_audio(
        audio
    )

    # --------------------------------------------------------
    # Time-domain
    # --------------------------------------------------------

    rms = float(
        np.sqrt(
            np.mean(
                audio ** 2
            )
        )
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    zero_crossing = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                audio,
                frame_length=1024,
                hop_length=256,
            )
        )
    )

    # --------------------------------------------------------
    # Spectral statistics
    # --------------------------------------------------------

    centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=audio,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )
        )
    )

    bandwidth = float(
        np.mean(
            librosa.feature.spectral_bandwidth(
                y=audio,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )
        )
    )

    rolloff = float(
        np.mean(
            librosa.feature.spectral_rolloff(
                y=audio,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                roll_percent=0.85,
            )
        )
    )

    # --------------------------------------------------------
    # Silence
    # --------------------------------------------------------

    if peak > 1e-8:

        threshold = (
            peak
            * (10 ** (-40 / 20))
        )

        silence_40 = float(
            np.mean(
                np.abs(audio)
                < threshold
            )
        )

    else:

        silence_40 = 1.0

    # --------------------------------------------------------
    # Mel statistics
    # --------------------------------------------------------

    mel_mean = float(
        mel_db.mean()
    )

    mel_std = float(
        mel_db.std()
    )

    mel_p05 = float(
        np.percentile(
            mel_db,
            5,
        )
    )

    mel_p25 = float(
        np.percentile(
            mel_db,
            25,
        )
    )

    mel_p50 = float(
        np.percentile(
            mel_db,
            50,
        )
    )

    mel_p75 = float(
        np.percentile(
            mel_db,
            75,
        )
    )

    mel_p95 = float(
        np.percentile(
            mel_db,
            95,
        )
    )

    # --------------------------------------------------------
    # Mel frequency-band energy
    #
    # 128 Mel bands are divided into:
    #   low
    #   mid
    #   high
    # --------------------------------------------------------

    mel_linear = librosa.db_to_power(
        mel_db
    )

    band_count = mel_linear.shape[0]

    low_end = band_count // 3
    high_start = (
        band_count * 2 // 3
    )

    low_energy = float(
        np.mean(
            mel_linear[
                :low_end
            ]
        )
    )

    mid_energy = float(
        np.mean(
            mel_linear[
                low_end:high_start
            ]
        )
    )

    high_energy = float(
        np.mean(
            mel_linear[
                high_start:
            ]
        )
    )

    total_energy = (
        low_energy
        + mid_energy
        + high_energy
    )

    if total_energy > 0:

        low_ratio = (
            low_energy
            / total_energy
        )

        mid_ratio = (
            mid_energy
            / total_energy
        )

        high_ratio = (
            high_energy
            / total_energy
        )

    else:

        low_ratio = 0.0
        mid_ratio = 0.0
        high_ratio = 0.0

    return {
        "rms": rms,
        "peak": peak,
        "silence_40": silence_40,
        "zero_crossing": zero_crossing,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "mel_p05": mel_p05,
        "mel_p25": mel_p25,
        "mel_p50": mel_p50,
        "mel_p75": mel_p75,
        "mel_p95": mel_p95,
        "low_ratio": float(
            low_ratio
        ),
        "mid_ratio": float(
            mid_ratio
        ),
        "high_ratio": float(
            high_ratio
        ),
    }


# ============================================================
# INTERNAL EMBEDDING
# ============================================================

@torch.inference_mode()
def get_internal_embedding(
    feature,
):

    x = torch.from_numpy(
        feature
    )

    x = x.unsqueeze(0)
    x = x.unsqueeze(0)

    x = x.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Reproduce the model forward pass up to classifier.
    #
    # This is exactly the representation immediately before
    # the V3 classifier.
    # --------------------------------------------------------

    x = model.stem(
        x
    )

    x = model.multi_scale(
        x
    )

    x = model.stage1(
        x
    )

    x = model.stage2(
        x
    )

    x = model.stage3(
        x
    )

    x = model.stage4(
        x
    )

    avg = model.avg_pool(
        x
    )

    max_value = model.max_pool(
        x
    )

    avg = torch.flatten(
        avg,
        start_dim=1,
    )

    max_value = torch.flatten(
        max_value,
        start_dim=1,
    )

    embedding = torch.cat(
        [
            avg,
            max_value,
        ],
        dim=1,
    )

    # --------------------------------------------------------
    # Classifier output
    # --------------------------------------------------------

    logits = model.classifier(
        embedding
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    prediction = int(
        torch.argmax(
            probabilities
        ).item()
    )

    embedding = (
        embedding[0]
        .detach()
        .cpu()
        .numpy()
    )

    probabilities = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    return (
        embedding,
        prediction,
        probabilities,
    )


# ============================================================
# LOAD V3 TEST MANIFEST
# ============================================================

def read_test_manifest():

    rows = []

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            if (
                row["split"]
                .strip()
                .lower()
                != "test"
            ):
                continue

            label_id = int(
                row["label_id"]
            )

            if label_id != 0:
                continue

            rows.append(
                row
            )

    return rows


# ============================================================
# MAIN DATA COLLECTION
# ============================================================

records = []

rng = np.random.default_rng(
    SEED
)

# ------------------------------------------------------------
# ASVSPOOF GENUINE
# ------------------------------------------------------------

print(
    "\nSelecting ASVspoof held-out Genuine samples..."
)

genuine_rows = read_test_manifest()

print(
    f"Available held-out Genuine: "
    f"{len(genuine_rows)}"
)

if len(genuine_rows) < NUM_ASVSPOOF_GENUINE:

    raise RuntimeError(
        "Not enough Genuine test samples."
    )

selected_indices = rng.choice(
    len(genuine_rows),
    size=NUM_ASVSPOOF_GENUINE,
    replace=False,
)

selected_rows = [
    genuine_rows[i]
    for i in selected_indices
]


print(
    f"Selected {len(selected_rows)} "
    f"ASVspoof Genuine samples."
)


# ------------------------------------------------------------
# OPEN ZIP
# ------------------------------------------------------------

print(
    "\nOpening ASVspoof archive..."
)

with zipfile.ZipFile(
    ZIP_PATH,
    "r",
) as zf:

    archive_members = set(
        zf.namelist()
    )

    for index, row in enumerate(
        selected_rows,
        start=1,
    ):

        try:

            audio_path = row[
                "audio_path"
            ]

            if (
                audio_path
                not in archive_members
            ):

                print(
                    f"ASVspoof [{index:03d}] "
                    f"MISSING "
                    f"{row['audio_id']}"
                )

                continue

            audio, sr = load_zip_audio(
                zf,
                audio_path,
            )

            audio = resample_audio(
                audio,
                sr,
            )

            feature, mel_db = (
                extract_features(
                    audio
                )
            )

            embedding, prediction, probs = (
                get_internal_embedding(
                    feature
                )
            )

            # We only want correctly classified
            # Genuine samples for the clean benchmark
            # representation.

            if prediction != 0:

                print(
                    f"ASVspoof [{index:03d}] "
                    f"skipped — predicted "
                    f"{CLASS_NAMES[prediction]}"
                )

                continue

            statistics = (
                calculate_statistics(
                    audio,
                    mel_db,
                )
            )

            records.append(
                {
                    "source": "ASVspoof_Genuine",
                    "name": row[
                        "audio_id"
                    ],
                    "speaker": row[
                        "speaker_id"
                    ],
                    "expected": "Genuine",
                    "predicted": "Genuine",
                    **statistics,
                    "_embedding": embedding,
                }
            )

        except Exception as e:

            print(
                f"ASVspoof [{index:03d}] "
                f"FAILED: {e}"
            )


# ============================================================
# REAL-WORLD GENUINE FILES
# ============================================================

REAL_FILES = [
    "sample-12s.wav",
    "voice-sample-44100hz-16bit.flac",
    "voice-sample.mp3",
    "voice-sample.ogg",
    "Recording_test.wav",
]


print(
    "\nProcessing real-world Genuine recordings..."
)


for filename in REAL_FILES:

    path = (
        REAL_AUDIO_DIR
        / filename
    )

    if not path.exists():

        print(
            f"Real [{filename}] "
            f"SKIPPED — file not found"
        )

        continue

    try:

        audio, sr = load_audio_file(
            path
        )

        audio = resample_audio(
            audio,
            sr,
        )

        feature, mel_db = (
            extract_features(
                audio
            )
        )

        embedding, prediction, probs = (
            get_internal_embedding(
                feature
            )
        )

        statistics = (
            calculate_statistics(
                audio,
                mel_db,
            )
        )

        records.append(
            {
                "source": "Real_Genuine",
                "name": filename,
                "speaker": "real_world",
                "expected": "Genuine",
                "predicted": CLASS_NAMES[
                    prediction
                ],
                **statistics,
                "_embedding": embedding,
            }
        )

        print(
            f"Real [{filename}] -> "
            f"{CLASS_NAMES[prediction]} | "
            f"G={probs[0] * 100:.2f}% "
            f"TTS={probs[1] * 100:.2f}% "
            f"VC={probs[2] * 100:.2f}%"
        )

    except Exception as e:

        print(
            f"Real [{filename}] "
            f"FAILED: {e}"
        )


# ============================================================
# EMBEDDING DISTANCE ANALYSIS
# ============================================================

benchmark_records = [
    r
    for r in records
    if r["source"]
    == "ASVspoof_Genuine"
]

real_records = [
    r
    for r in records
    if r["source"]
    == "Real_Genuine"
]


print(
    "\n" + "=" * 80
)
print(
    "FEATURE SPACE ANALYSIS"
)
print(
    "=" * 80
)

print(
    f"Correct ASVspoof Genuine samples: "
    f"{len(benchmark_records)}"
)

print(
    f"Real-world Genuine samples: "
    f"{len(real_records)}"
)


if benchmark_records:

    benchmark_embeddings = np.stack(
        [
            r["_embedding"]
            for r in benchmark_records
        ]
    )

    benchmark_mean = (
        benchmark_embeddings.mean(
            axis=0
        )
    )

    benchmark_std = (
        benchmark_embeddings.std(
            axis=0
        )
    )

    # --------------------------------------------------------
    # Distances of real samples from ASVspoof Genuine center
    # --------------------------------------------------------

    print(
        "\nDistance from ASVspoof Genuine "
        "embedding center:"
    )

    real_distances = []

    for r in real_records:

        embedding = r[
            "_embedding"
        ]

        euclidean = float(
            np.linalg.norm(
                embedding
                - benchmark_mean
            )
        )

        benchmark_z = (
            (
                embedding
                - benchmark_mean
            )
            / (
                benchmark_std
                + 1e-8
            )
        )

        mean_abs_z = float(
            np.mean(
                np.abs(
                    benchmark_z
                )
            )
        )

        r[
            "embedding_euclidean_distance"
        ] = euclidean

        r[
            "embedding_mean_abs_z"
        ] = mean_abs_z

        real_distances.append(
            euclidean
        )

        print(
            f"  {r['name']:40s} | "
            f"Euclidean={euclidean:10.3f} | "
            f"MeanAbsZ={mean_abs_z:7.3f} | "
            f"Prediction={r['predicted']}"
        )

    # --------------------------------------------------------
    # Benchmark-to-benchmark distances
    # --------------------------------------------------------

    benchmark_distances = []

    for i in range(
        len(benchmark_embeddings)
    ):

        for j in range(
            i + 1,
            len(benchmark_embeddings),
        ):

            distance = float(
                np.linalg.norm(
                    benchmark_embeddings[i]
                    - benchmark_embeddings[j]
                )
            )

            benchmark_distances.append(
                distance
            )

    if benchmark_distances:

        print(
            "\nASVspoof Genuine "
            "benchmark-to-benchmark distance:"
        )

        print(
            f"  Mean : "
            f"{np.mean(benchmark_distances):.3f}"
        )

        print(
            f"  P50  : "
            f"{np.percentile(benchmark_distances, 50):.3f}"
        )

        print(
            f"  P95  : "
            f"{np.percentile(benchmark_distances, 95):.3f}"
        )

    # --------------------------------------------------------
    # Compare real distances with benchmark distances
    # --------------------------------------------------------

    if benchmark_distances:

        p95 = np.percentile(
            benchmark_distances,
            95,
        )

        print(
            "\nReal-vs-benchmark distance check:"
        )

        for r in real_records:

            distance = r[
                "embedding_euclidean_distance"
            ]

            ratio = (
                distance / p95
                if p95 > 0
                else 0
            )

            print(
                f"  {r['name']:40s} | "
                f"distance={distance:10.3f} | "
                f"benchmark P95={p95:10.3f} | "
                f"ratio={ratio:6.2f}x"
            )


# ============================================================
# STATISTICAL GROUP COMPARISON
# ============================================================

STAT_KEYS = [
    "rms",
    "peak",
    "silence_40",
    "zero_crossing",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "mel_mean",
    "mel_std",
    "mel_p05",
    "mel_p25",
    "mel_p50",
    "mel_p75",
    "mel_p95",
    "low_ratio",
    "mid_ratio",
    "high_ratio",
]


print(
    "\n" + "=" * 80
)
print(
    "SPECTRAL STATISTICS COMPARISON"
)
print(
    "=" * 80
)

if benchmark_records:

    for key in STAT_KEYS:

        benchmark_values = np.array(
            [
                r[key]
                for r in benchmark_records
            ],
            dtype=np.float64,
        )

        real_values = np.array(
            [
                r[key]
                for r in real_records
            ],
            dtype=np.float64,
        )

        benchmark_mean_value = (
            np.mean(
                benchmark_values
            )
        )

        benchmark_p05 = (
            np.percentile(
                benchmark_values,
                5,
            )
        )

        benchmark_p95 = (
            np.percentile(
                benchmark_values,
                95,
            )
        )

        real_mean_value = (
            np.mean(
                real_values
            )
            if len(real_values)
            else float("nan")
        )

        print(
            f"\n{key}"
        )

        print(
            f"  ASVspoof Genuine "
            f"mean={benchmark_mean_value:.6f} "
            f"P05={benchmark_p05:.6f} "
            f"P95={benchmark_p95:.6f}"
        )

        print(
            f"  Real Genuine     "
            f"mean={real_mean_value:.6f}"
        )

        # Count real samples outside benchmark P05-P95.

        outside = np.sum(
            (
                real_values
                < benchmark_p05
            )
            |
            (
                real_values
                > benchmark_p95
            )
        )

        print(
            f"  Real outside "
            f"ASVspoof P05-P95: "
            f"{outside}/{len(real_values)}"
        )


# ============================================================
# SAVE CSV
# ============================================================

print(
    "\nSaving detailed results..."
)

OUTPUT_PATH = OUTPUT_CSV

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fieldnames = [
    "source",
    "name",
    "speaker",
    "expected",
    "predicted",
    *STAT_KEYS,
    "embedding_euclidean_distance",
    "embedding_mean_abs_z",
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
        extrasaction="ignore",
    )

    writer.writeheader()

    for record in records:

        writer.writerow(
            record
        )


print(
    f"\nResults saved to:\n"
    f"{OUTPUT_PATH}"
)

print(
    "\n" + "=" * 80
)
print(
    "FEATURE SPACE DIAGNOSTIC COMPLETE"
)
print(
    "=" * 80
)