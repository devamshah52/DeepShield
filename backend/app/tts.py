from pathlib import Path
from gtts import gTTS


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = (
    Path(r"C:\DeepShield")
    / "tts_audio"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Generate TTS Audio
# ============================================================

text = input("Enter the text for TTS: ").strip()

if not text:
    print("No text entered.")
    raise SystemExit


# ------------------------------------------------------------
# Find next available filename
# ------------------------------------------------------------

number = 1

while True:

    output_path = (
        OUTPUT_DIR
        / f"tts_{number:02d}.mp3"
    )

    if not output_path.exists():
        break

    number += 1


# ============================================================
# Generate
# ============================================================

tts = gTTS(
    text=text,
    lang="en",
)

tts.save(
    str(output_path)
)


# ============================================================
# Result
# ============================================================

print()
print("Audio saved successfully.")
print("File:", output_path)