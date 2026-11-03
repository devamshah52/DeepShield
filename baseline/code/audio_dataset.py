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
        # Normal feature cache directory
        #
        # For example:
        #
        # split="train"
        #   -> ml/features/cache/train
        #
        # split="dev"
        #   -> ml/features/cache/dev
        #
        # split="eval"
        #   -> ml/features/cache/eval
        # ----------------------------------------------------

        self.cache_dir = (
            BACKEND_DIR
            / "ml"
            / "features"
            / "cache"
            / split
        )

        # ----------------------------------------------------
        # VC augmented training cache
        #
        # Augmented samples have IDs beginning with:
        #
        # VC_AUG_
        #
        # Their features are stored separately.
        # ----------------------------------------------------

        self.augmented_cache_dir = (
            BACKEND_DIR
            / "ml"
            / "features"
            / "cache"
            / "train_vc_augmented"
        )

        # ----------------------------------------------------
        # Verify normal cache
        # ----------------------------------------------------

        if not self.cache_dir.exists():

            raise FileNotFoundError(
                f"Feature cache not found:\n"
                f"{self.cache_dir}\n\n"
                f"Run cache_features.py first."
            )

        # ----------------------------------------------------
        # Verify augmented cache when needed
        #
        # We only require it if this dataset actually contains
        # VC_AUG_ samples.
        # ----------------------------------------------------

        has_augmented_samples = (
            self.dataframe["audio_id"]
            .astype(str)
            .str.startswith("VC_AUG_")
            .any()
        )

        if has_augmented_samples:

            if not self.augmented_cache_dir.exists():

                raise FileNotFoundError(
                    f"VC augmented feature cache not found:\n"
                    f"{self.augmented_cache_dir}\n\n"
                    f"Run create_vc_augmented_cache.py first."
                )

    # ========================================================
    # Dataset length
    # ========================================================

    def __len__(self):

        return len(
            self.dataframe
        )

    # ========================================================
    # Resolve feature path
    # ========================================================

    def _get_feature_path(
        self,
        audio_id,
    ):

        audio_id = str(
            audio_id
        )

        # ----------------------------------------------------
        # VC augmented samples
        #
        # These always use:
        #
        # train_vc_augmented/
        # ----------------------------------------------------

        if audio_id.startswith(
            "VC_AUG_"
        ):

            return (
                self.augmented_cache_dir
                / f"{audio_id}.npy"
            )

        # ----------------------------------------------------
        # All normal samples
        #
        # Use the cache corresponding to the split.
        # ----------------------------------------------------

        return (
            self.cache_dir
            / f"{audio_id}.npy"
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
        # Resolve cached feature path
        # ----------------------------------------------------

        feature_path = (
            self._get_feature_path(
                audio_id
            )
        )

        # ----------------------------------------------------
        # Verify feature exists
        # ----------------------------------------------------

        if not feature_path.exists():

            raise FileNotFoundError(
                f"Cached feature not found:\n"
                f"{feature_path}\n\n"
                f"Audio ID: {audio_id}\n"
                f"Split: {self.split}"
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
        #
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
        # NumPy -> PyTorch Tensor
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

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return (
            mel_tensor,
            label_tensor
        )