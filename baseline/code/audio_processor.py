import subprocess
from pathlib import Path

import librosa
import numpy as np


# ============================================================
# Configuration
# ============================================================

TARGET_SAMPLE_RATE = 16000
TARGET_DURATION = 4
TARGET_SAMPLES = TARGET_SAMPLE_RATE * TARGET_DURATION

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
}


# ============================================================
# Validate audio format
# ============================================================

def validate_audio_format(file_path):
    """
    Validate that the supplied file has a supported
    audio extension.
    """

    extension = Path(file_path).suffix.lower()

    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            "Unsupported audio format. "
            "Supported formats: WAV, MP3, M4A, FLAC, OGG."
        )

    return True


# ============================================================
# Load audio using FFmpeg
# ============================================================

def load_audio(
    file_path,
    sample_rate=TARGET_SAMPLE_RATE,
):
    """
    Load audio using FFmpeg.

    The audio is converted to:
        - Mono
        - 16 kHz
        - float32 PCM

    The result is standardized to exactly 4 seconds.
    """

    file_path = str(Path(file_path).resolve())

    validate_audio_format(file_path)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        file_path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    except FileNotFoundError:

        raise RuntimeError(
            "FFmpeg was not found. "
            "Make sure FFmpeg is installed and available in PATH."
        )

    except subprocess.CalledProcessError as error:

        error_message = (
            error.stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()
        )

        raise RuntimeError(
            f"FFmpeg could not decode the audio file: "
            f"{error_message}"
        )

    audio = np.frombuffer(
        result.stdout,
        dtype=np.float32,
    ).copy()

    if audio.size == 0:

        raise ValueError(
            "The audio file contains no readable audio data."
        )

    # --------------------------------------------------------
    # Standardize to exactly 4 seconds
    # --------------------------------------------------------

    if len(audio) > TARGET_SAMPLES:

        audio = audio[:TARGET_SAMPLES]

    elif len(audio) < TARGET_SAMPLES:

        padding = TARGET_SAMPLES - len(audio)

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant",
        )

    return audio.astype(np.float32), sample_rate


# ============================================================
# Audio information
# ============================================================

def get_audio_info(file_path):

    audio, sr = load_audio(file_path)

    duration = len(audio) / sr

    return {
        "sample_rate": sr,
        "samples": len(audio),
        "duration": duration,
        "min_amplitude": float(np.min(audio)),
        "max_amplitude": float(np.max(audio)),
    }


# ============================================================
# Mel Spectrogram
# ============================================================

def extract_mel_spectrogram(file_path):

    audio, sample_rate = load_audio(
        file_path
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mean = np.mean(mel_db)
    std = np.std(mel_db)

    if std > 0:

        mel_db = (
            mel_db - mean
        ) / std

    else:

        mel_db = mel_db - mean

    return mel_db.astype(
        np.float32
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print(
        "DeepShield Audio Processor"
    )

    print(
        "Supported formats:"
    )

    for extension in sorted(
        SUPPORTED_AUDIO_EXTENSIONS
    ):

        print(
            f"  {extension}"
        )