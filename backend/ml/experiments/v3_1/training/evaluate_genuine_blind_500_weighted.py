"""
DeepShield V3.2
Independent Genuine Blind-500 Evaluation

Purpose:
Evaluate the V3.2 weighted model on 500 Genuine recordings
from speakers completely excluded from V3.2 training.

This evaluation:
- Does NOT train the model
- Does NOT modify any dataset
- Does NOT modify production
- Uses the weighted V3.2 best checkpoint
- Uses pre-extracted V3-compatible features
"""

from pathlib import Path
import sys
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

FEATURE_DIR = (
    PROJECT_ROOT
    / "v3_1_dataset_inspection"
    / "genuine_blind_500_features"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "ml"
    / "experiments"
    / "v3_1"
    / "checkpoints"
    / "deepshield_v3_2_weighted_best.pth"
)

# Add backend to Python path
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# MODEL IMPORT
# ============================================================

from ml.experiments.v3.model.v3_model import DeepShieldV3


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = {
    0: "Genuine",
    1: "TTS",
    2: "Voice_Conversion",
}

EXPECTED_SHAPE = (128, 251)


# ============================================================
# CHECK PATHS
# ============================================================

print("=" * 75)
print("DeepShield V3.2 - Independent Genuine Blind 500 Evaluation")
print("=" * 75)

print()
print("Feature directory:")
print(FEATURE_DIR)

print()
print("Model checkpoint:")
print(MODEL_PATH)

print()

if not FEATURE_DIR.exists():
    raise FileNotFoundError(
        f"Feature directory does not exist:\n{FEATURE_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint does not exist:\n{MODEL_PATH}"
    )


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 75)
print("LOADING MODEL")
print("=" * 75)

model = DeepShieldV3(
    num_classes=3,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

# Support normal state_dict checkpoint format
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model.to(device)
model.eval()

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Model loaded successfully.")
print("Parameters:", f"{total_params:,}")
print("Checkpoint:", MODEL_PATH.name)
print()


# ============================================================
# FIND FEATURES
# ============================================================

feature_files = sorted(FEATURE_DIR.glob("*.npy"))

print("=" * 75)
print("FEATURE VALIDATION")
print("=" * 75)

print("Feature files found:", len(feature_files))

if len(feature_files) != 500:
    raise RuntimeError(
        f"Expected exactly 500 feature files, found {len(feature_files)}"
    )

print("Expected shape:", EXPECTED_SHAPE)
print()


# ============================================================
# VALIDATE ALL FEATURES
# ============================================================

invalid_files = []

for feature_path in feature_files:

    try:
        feature = np.load(feature_path)

        if feature.shape != EXPECTED_SHAPE:
            invalid_files.append(
                (
                    feature_path.name,
                    f"wrong shape {feature.shape}"
                )
            )
            continue

        if feature.dtype != np.float32:
            invalid_files.append(
                (
                    feature_path.name,
                    f"wrong dtype {feature.dtype}"
                )
            )
            continue

        if not np.isfinite(feature).all():
            invalid_files.append(
                (
                    feature_path.name,
                    "contains non-finite values"
                )
            )

    except Exception as e:
        invalid_files.append(
            (
                feature_path.name,
                str(e)
            )
        )

if invalid_files:

    print("INVALID FEATURES FOUND:")

    for filename, reason in invalid_files[:20]:
        print(f"  {filename}: {reason}")

    raise RuntimeError(
        f"{len(invalid_files)} invalid feature files found."
    )

print("All 500 feature files are valid.")
print()


# ============================================================
# EVALUATION
# ============================================================

print("=" * 75)
print("EVALUATING 500 INDEPENDENT GENUINE SAMPLES")
print("=" * 75)

print()
print("Ground truth for every sample: Genuine")
print()

prediction_counts = {
    0: 0,
    1: 0,
    2: 0,
}

confidences = []

results = []

correct = 0

with torch.no_grad():

    for index, feature_path in enumerate(feature_files, start=1):

        feature = np.load(feature_path)

        # numpy:
        # (128, 251)
        #
        # torch:
        # (1, 1, 128, 251)

        tensor = torch.from_numpy(feature).float()

        tensor = tensor.unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(device)

        logits = model(tensor)

        probabilities = torch.softmax(logits, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        predicted_class = int(prediction.item())
        confidence_value = float(confidence.item() * 100.0)

        prediction_counts[predicted_class] += 1
        confidences.append(confidence_value)

        if predicted_class == 0:
            correct += 1

        results.append(
            (
                feature_path.name,
                predicted_class,
                confidence_value
            )
        )

        print(
            f"[{index:03d}/500] "
            f"{feature_path.name} -> "
            f"{CLASS_NAMES[predicted_class]} "
            f"({confidence_value:.2f}%)"
        )


# ============================================================
# SUMMARY
# ============================================================

accuracy = (correct / 500) * 100.0

confidence_array = np.array(
    confidences,
    dtype=np.float64
)

mean_confidence = float(confidence_array.mean())
median_confidence = float(np.median(confidence_array))
minimum_confidence = float(confidence_array.min())
maximum_confidence = float(confidence_array.max())


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 75)
print("FINAL RESULTS")
print("=" * 75)

print()

print("Total samples          :", 500)
print("Ground truth           : Genuine")

print()

print(
    f"Genuine predictions    : "
    f"{prediction_counts[0]} "
    f"({prediction_counts[0] / 500 * 100:.2f}%)"
)

print(
    f"TTS predictions        : "
    f"{prediction_counts[1]} "
    f"({prediction_counts[1] / 500 * 100:.2f}%)"
)

print(
    f"Voice Conversion       : "
    f"{prediction_counts[2]} "
    f"({prediction_counts[2] / 500 * 100:.2f}%)"
)

print()

print(
    f"Genuine detection rate : "
    f"{accuracy:.2f}%"
)

print()

print("Confidence statistics:")
print(f"  Mean   : {mean_confidence:.2f}%")
print(f"  Median : {median_confidence:.2f}%")
print(f"  Min    : {minimum_confidence:.2f}%")
print(f"  Max    : {maximum_confidence:.2f}%")

print()


# ============================================================
# MISCLASSIFIED SAMPLES
# ============================================================

misclassified = [
    result
    for result in results
    if result[1] != 0
]

print("=" * 75)
print("MISCLASSIFIED GENUINE SAMPLES")
print("=" * 75)

print()

if not misclassified:

    print("None.")
    print("All 500 Genuine samples were classified as Genuine.")

else:

    for filename, predicted_class, confidence in misclassified:

        print(
            f"{filename} -> "
            f"{CLASS_NAMES[predicted_class]} "
            f"({confidence:.2f}%)"
        )

print()


# ============================================================
# STATUS
# ============================================================

print("=" * 75)

if accuracy == 100.0:

    print("STATUS: PASS")
    print("500/500 independent Genuine samples classified correctly.")

else:

    print("STATUS: COMPLETED")
    print(
        f"{prediction_counts[0]}/500 independent Genuine samples "
        f"classified correctly."
    )

print("=" * 75)