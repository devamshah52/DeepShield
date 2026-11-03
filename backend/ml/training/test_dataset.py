import pandas as pd

from audio_dataset import (
    DeepShieldAudioDataset,
)


# ============================================================
# Load training manifest
# ============================================================

manifest_path = (
    r".\ml\dataset\train_manifest.csv"
)

dataframe = pd.read_csv(
    manifest_path
)


# ============================================================
# Create dataset
# ============================================================

dataset = DeepShieldAudioDataset(
    dataframe
)


# ============================================================
# Dataset information
# ============================================================

print()
print("=" * 60)
print("DeepShield PyTorch Dataset Test")
print("=" * 60)

print(
    "Dataset size:",
    len(dataset)
)


# ============================================================
# Load first sample
# ============================================================

features, label = dataset[0]


print(
    "Feature shape:",
    features.shape
)

print(
    "Feature dtype:",
    features.dtype
)

print(
    "Label:",
    label.item()
)

print(
    "Feature minimum:",
    features.min().item()
)

print(
    "Feature maximum:",
    features.max().item()
)

print("=" * 60)