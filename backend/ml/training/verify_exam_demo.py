import sys
from pathlib import Path


# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BACKEND_DIR)
    )


from app.services.predict import (
    predict_audio,
)


# ============================================================
# Configuration
# ============================================================

ROOT = (
    BACKEND_DIR.parent
    / "exam_demo_audio"
)


# ============================================================
# Main
# ============================================================

print()
print("=" * 100)
print("DeepShield VERIFIED EXAM DEMO TEST")
print("=" * 100)
print()

print(
    "Demo directory:",
    ROOT
)

print()


# ============================================================
# Verify directory
# ============================================================

if not ROOT.exists():

    raise FileNotFoundError(
        f"Exam demo directory not found:\n"
        f"{ROOT}"
    )


# ============================================================
# Find files
# ============================================================

files = sorted(
    [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
    ]
)


print(
    "Files found:",
    len(files)
)

print()


# ============================================================
# Test every file
# ============================================================

results = []


for file_path in files:

    result = predict_audio(
        str(file_path)
    )

    prediction = result[
        "prediction"
    ]

    confidence = result[
        "confidence"
    ]

    actual_folder = (
        file_path.parent.name
    )

    results.append(
        {
            "actual": actual_folder,
            "file": file_path.name,
            "prediction": prediction,
            "confidence": confidence,
        }
    )

    print(
        f"{actual_folder:<20} "
        f"{file_path.name:<25} "
        f"=> {prediction:<20} "
        f"{confidence:.2f}%"
    )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print()


for class_name in [
    "Genuine",
    "TTS",
    "Voice Conversion",
]:

    class_results = [
        result
        for result in results
        if result["actual"] == class_name
    ]

    correct = sum(
        1
        for result in class_results
        if result["prediction"] == class_name
    )

    total = len(
        class_results
    )

    print(
        f"{class_name:<20} "
        f"{correct}/{total} correct"
    )


print()
print("=" * 100)
print("VERIFICATION COMPLETED")
print("=" * 100)