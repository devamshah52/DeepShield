from pathlib import Path

from app.services.predict import predict_audio


# ============================================================
# Genuine audio files
# ============================================================

FOLDER = Path(
    r"C:\DeepShield\exam_test_audio"
)

FILES = [
    "sample-12s.wav",
    "voice-sample-44100hz-16bit.flac",
    "voice-sample-96kbps.m4a",
    "voice-sample.mp3",
    "voice-sample.ogg",
]


# ============================================================
# Test
# ============================================================

print()
print("=" * 110)
print("DEEPSHIELD V3 - GENUINE AUDIO TEST")
print("=" * 110)

print()

for filename in FILES:

    file_path = FOLDER / filename

    print("-" * 110)
    print(f"File: {filename}")

    if not file_path.exists():

        print("STATUS: FILE NOT FOUND")
        continue

    try:

        result = predict_audio(
            str(file_path)
        )

        probabilities = result[
            "probabilities"
        ]

        print(
            f"Prediction        : "
            f"{result['prediction']}"
        )

        print(
            f"Confidence        : "
            f"{result['confidence']:.2f}%"
        )

        print(
            f"Genuine           : "
            f"{probabilities['Genuine']:.2f}%"
        )

        print(
            f"TTS               : "
            f"{probabilities['TTS']:.2f}%"
        )

        print(
            f"Voice Conversion  : "
            f"{probabilities['Voice Conversion']:.2f}%"
        )

        print(
            f"Feature Shape     : "
            f"{result['feature_shape']}"
        )

        print(
            f"Device            : "
            f"{result['device']}"
        )

        print("STATUS            : PASS")

    except Exception as error:

        print(
            f"STATUS            : ERROR"
        )

        print(
            f"ERROR             : {error}"
        )


print()
print("=" * 110)
print("GENUINE AUDIO TEST COMPLETE")
print("=" * 110)