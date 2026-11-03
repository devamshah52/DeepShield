import sys
from pathlib import Path

import subprocess
import numpy as np
import pandas as pd
import librosa
import torch


# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from ml.training.model_pool import DeepShieldCNNPool


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000
TARGET_SAMPLES = SAMPLE_RATE * 4

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

NUM_SAMPLES_PER_GROUP = 20

MODEL_PATH = (
    BACKEND_DIR
    / "ml"
    / "models"
    / "deepshield_cnn_pool_best.pth"
)

OUTPUT_FILE = (
    BACKEND_DIR
    / "ml"
    / "evaluation"
    / "centered_inference_diagnostic.csv"
)

CLASS_NAMES = [
    "Genuine",
    "TTS",
    "Voice Conversion",
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load original audio
#
# IMPORTANT:
# Do NOT use audio_processor.load_audio()
# because that already forces 4 seconds.
# ============================================================

def load_original_audio(file_path):

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(Path(file_path).resolve()),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    audio = np.frombuffer(
        result.stdout,
        dtype=np.float32,
    ).copy()

    if audio.size == 0:
        raise ValueError(
            f"No readable audio: {file_path}"
        )

    return audio


# ============================================================
# Current DeepShield preprocessing
# ============================================================

def current_crop(audio):

    if len(audio) >= TARGET_SAMPLES:

        return audio[
            :TARGET_SAMPLES
        ]

    padding = (
        TARGET_SAMPLES
        - len(audio)
    )

    return np.pad(
        audio,
        (0, padding),
        mode="constant",
    )


# ============================================================
# Speech-centered preprocessing
# ============================================================

def speech_centered_crop(audio):

    if len(audio) <= TARGET_SAMPLES:

        padding_total = (
            TARGET_SAMPLES
            - len(audio)
        )

        left_padding = (
            padding_total // 2
        )

        right_padding = (
            padding_total
            - left_padding
        )

        return np.pad(
            audio,
            (
                left_padding,
                right_padding,
            ),
            mode="constant",
        )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=256,
    )[0]

    threshold = max(
        np.percentile(
            rms,
            30,
        ),
        rms.max() * 0.10,
    )

    active = np.where(
        rms >= threshold
    )[0]

    if len(active) == 0:

        start = (
            len(audio)
            - TARGET_SAMPLES
        ) // 2

        start = max(
            0,
            start,
        )

        return audio[
            start:start + TARGET_SAMPLES
        ]

    first_frame = int(
        active[0]
    )

    last_frame = int(
        active[-1]
    )

    speech_start = (
        first_frame
        * HOP_LENGTH
    )

    speech_end = min(
        len(audio),
        last_frame
        * HOP_LENGTH
        + 1024,
    )

    speech_center = (
        speech_start
        + speech_end
    ) // 2

    start = (
        speech_center
        - TARGET_SAMPLES // 2
    )

    start = max(
        0,
        min(
            start,
            len(audio)
            - TARGET_SAMPLES,
        ),
    )

    return audio[
        start:start + TARGET_SAMPLES
    ]


# ============================================================
# Mel spectrogram
# ============================================================

def extract_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    mean = mel_db.mean()
    std = mel_db.std()

    if std > 0:

        mel_db = (
            mel_db - mean
        ) / std

    else:

        mel_db = (
            mel_db - mean
        )

    return mel_db.astype(
        np.float32
    )


# ============================================================
# Model loading
# ============================================================

print()
print("=" * 90)
print("DeepShield Current vs Speech-Centered Inference Diagnostic")
print("=" * 90)
print()

print(
    "Device:",
    DEVICE,
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

print(
    "Model:",
    MODEL_PATH,
)


model = DeepShieldCNNPool(
    num_classes=3
).to(DEVICE)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# Prediction helper
# ============================================================

def predict_mel(mel):

    features = torch.from_numpy(
        mel
    ).float()

    features = (
        features
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = model(
            features
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

    values = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    predicted_index = int(
        np.argmax(values)
    )

    return (
        CLASS_NAMES[
            predicted_index
        ],
        values,
    )


# ============================================================
# Load manifests
# ============================================================

train = pd.read_csv(
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "train_manifest.csv"
)

evaluation = pd.read_csv(
    BACKEND_DIR
    / "ml"
    / "dataset"
    / "eval_manifest.csv"
)


# ============================================================
# Select samples
# ============================================================

groups = [
    (
        "Genuine",
        train[
            train["attack_id"] == "-"
        ],
    ),

    (
        "A05",
        train[
            train["attack_id"] == "A05"
        ],
    ),

    (
        "A06",
        train[
            train["attack_id"] == "A06"
        ],
    ),

    (
        "A17",
        evaluation[
            evaluation["attack_id"] == "A17"
        ],
    ),

    (
        "A18",
        evaluation[
            evaluation["attack_id"] == "A18"
        ],
    ),

    (
        "A19",
        evaluation[
            evaluation["attack_id"] == "A19"
        ],
    ),
]


# ============================================================
# Run diagnostic
# ============================================================

results = []

total = (
    len(groups)
    * NUM_SAMPLES_PER_GROUP
)

processed = 0

print()
print("=" * 90)
print(
    f"Running inference on "
    f"{total} samples"
)
print("=" * 90)
print()


for group_name, dataframe in groups:

    selected = dataframe.head(
        NUM_SAMPLES_PER_GROUP
    )

    print(
        f"Processing {group_name}: "
        f"{len(selected)} samples"
    )

    for _, row in selected.iterrows():

        audio_id = row[
            "audio_id"
        ]

        audio_path = row[
            "audio_path"
        ]

        audio = load_original_audio(
            audio_path
        )

        original_duration = (
            len(audio)
            / SAMPLE_RATE
        )

        # ----------------------------------------------------
        # Current preprocessing
        # ----------------------------------------------------

        current_audio = current_crop(
            audio
        )

        current_mel = extract_mel(
            current_audio
        )

        current_prediction, current_probs = (
            predict_mel(
                current_mel
            )
        )

        # ----------------------------------------------------
        # Speech-centered preprocessing
        # ----------------------------------------------------

        centered_audio = (
            speech_centered_crop(
                audio
            )
        )

        centered_mel = extract_mel(
            centered_audio
        )

        centered_prediction, centered_probs = (
            predict_mel(
                centered_mel
            )
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append(
            {
                "attack_id": group_name,
                "audio_id": audio_id,
                "original_label": row["label"],
                "original_duration_sec":
                    round(
                        original_duration,
                        3,
                    ),

                "current_prediction":
                    current_prediction,

                "current_genuine":
                    round(
                        float(
                            current_probs[0]
                        ) * 100,
                        2,
                    ),

                "current_tts":
                    round(
                        float(
                            current_probs[1]
                        ) * 100,
                        2,
                    ),

                "current_vc":
                    round(
                        float(
                            current_probs[2]
                        ) * 100,
                        2,
                    ),

                "centered_prediction":
                    centered_prediction,

                "centered_genuine":
                    round(
                        float(
                            centered_probs[0]
                        ) * 100,
                        2,
                    ),

                "centered_tts":
                    round(
                        float(
                            centered_probs[1]
                        ) * 100,
                        2,
                    ),

                "centered_vc":
                    round(
                        float(
                            centered_probs[2]
                        ) * 100,
                        2,
                    ),
            }
        )

        processed += 1

        if (
            processed % 20 == 0
        ):

            print(
                f"Processed "
                f"{processed}/{total}"
            )


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 90)
print("INDIVIDUAL RESULTS")
print("=" * 90)
print()

display_columns = [
    "attack_id",
    "audio_id",
    "current_prediction",
    "current_vc",
    "centered_prediction",
    "centered_vc",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# Group summary
# ============================================================

print()
print("=" * 90)
print("GROUP SUMMARY")
print("=" * 90)
print()


for group_name in [
    "Genuine",
    "A05",
    "A06",
    "A17",
    "A18",
    "A19",
]:

    subset = results_df[
        results_df["attack_id"]
        == group_name
    ]

    print()
    print(
        f"{group_name}"
    )
    print(
        "-" * 60
    )

    print(
        "Current mean VC probability:",
        round(
            subset[
                "current_vc"
            ].mean(),
            2,
        ),
        "%",
    )

    print(
        "Centered mean VC probability:",
        round(
            subset[
                "centered_vc"
            ].mean(),
            2,
        ),
        "%",
    )

    print(
        "VC probability change:",
        round(
            (
                subset[
                    "centered_vc"
                ].mean()
                -
                subset[
                    "current_vc"
                ].mean()
            ),
            2,
        ),
        "percentage points",
    )

    current_vc_predictions = (
        (
            subset[
                "current_prediction"
            ]
            == "Voice Conversion"
        )
        .sum()
    )

    centered_vc_predictions = (
        (
            subset[
                "centered_prediction"
            ]
            == "Voice Conversion"
        )
        .sum()
    )

    print(
        "Current VC predictions:",
        f"{current_vc_predictions}/{len(subset)}",
    )

    print(
        "Centered VC predictions:",
        f"{centered_vc_predictions}/{len(subset)}",
    )


# ============================================================
# Completed
# ============================================================

print()
print("=" * 90)
print("DIAGNOSTIC COMPLETED")
print("=" * 90)
print()

print(
    "Results saved:"
)

print(
    OUTPUT_FILE
)

print()