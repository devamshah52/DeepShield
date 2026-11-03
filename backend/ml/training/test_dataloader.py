import torch
import pandas as pd

from torch.utils.data import DataLoader

from audio_dataset import (
    DeepShieldAudioDataset,
)


# ============================================================
# Configuration
# ============================================================

MANIFEST_PATH = (
    r".\ml\dataset\train_manifest.csv"
)

BATCH_SIZE = 8


# ============================================================
# Load manifest
# ============================================================

dataframe = pd.read_csv(
    MANIFEST_PATH
)


# ============================================================
# Create Dataset
# ============================================================

dataset = DeepShieldAudioDataset(
    dataframe
)


# ============================================================
# Create DataLoader
# ============================================================

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


# ============================================================
# Select device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Test one batch
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DeepShield DataLoader Test")
    print("=" * 60)

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Dataset size:",
        len(dataset)
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    # --------------------------------------------------------
    # Get one batch
    # --------------------------------------------------------

    features, labels = next(
        iter(dataloader)
    )

    print(
        "CPU feature shape:",
        features.shape
    )

    print(
        "CPU label shape:",
        labels.shape
    )

    # --------------------------------------------------------
    # Move batch to GPU
    # --------------------------------------------------------

    features = features.to(
        device
    )

    labels = labels.to(
        device
    )

    print(
        "Device feature shape:",
        features.shape
    )

    print(
        "Device:",
        features.device
    )

    print(
        "Labels:",
        labels
    )

    print("=" * 60)