import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Add backend directory to Python path
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# DeepShield Cached Audio Dataset
# ============================================================

class DeepShieldAudioDataset(Dataset):

    def __init__(
        self,
        dataframe,
        split,
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.split = split

        # ----------------------------------------------------
        # Feature cache directory
        # ----------------------------------------------------

        self.cache_dir = (
            BACKEND_DIR
            / "ml"
            / "features"
            / "cache"
            / split
        )

        # ----------------------------------------------------
        # Verify cache exists
        # ----------------------------------------------------

        if not self.cache_dir.exists():

            raise FileNotFoundError(
                f"Feature cache not found:\n"
                f"{self.cache_dir}\n\n"
                f"Run cache_features.py first."
            )

    # ========================================================
    # Dataset length
    # ========================================================

    def __len__(self):

        return len(
            self.dataframe
        )

    # ========================================================
    # Get one sample
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        row = self.dataframe.iloc[index]

        # ----------------------------------------------------
        # Audio ID
        # ----------------------------------------------------

        audio_id = str(
            row["audio_id"]
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = int(
            row["label_id"]
        )

        # ----------------------------------------------------
        # Cached feature path
        # ----------------------------------------------------

        feature_path = (
            self.cache_dir
            / f"{audio_id}.npy"
        )

        # ----------------------------------------------------
        # Verify feature exists
        # ----------------------------------------------------

        if not feature_path.exists():

            raise FileNotFoundError(
                f"Cached feature not found:\n"
                f"{feature_path}"
            )

        # ----------------------------------------------------
        # Load cached Mel spectrogram
        # ----------------------------------------------------

        mel = np.load(
            feature_path
        )

        # ----------------------------------------------------
        # Safety check
        #
        # Expected:
        # [128, 251]
        # ----------------------------------------------------

        if mel.shape != (
            128,
            251,
        ):

            raise ValueError(
                f"Invalid feature shape for "
                f"{audio_id}: {mel.shape}"
            )

        # ----------------------------------------------------
        # NumPy → PyTorch Tensor
        # ----------------------------------------------------

        mel_tensor = torch.from_numpy(
            mel
        ).float()

        # ----------------------------------------------------
        # Add channel dimension
        #
        # Before:
        # [128, 251]
        #
        # After:
        # [1, 128, 251]
        # ----------------------------------------------------

        mel_tensor = mel_tensor.unsqueeze(
            0
        )

        # ----------------------------------------------------
        # Label tensor
        # ----------------------------------------------------

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return (
            mel_tensor,
            label_tensor
        )