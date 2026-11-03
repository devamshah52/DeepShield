"""
DeepShield V3.1 - Pre-Training Smoke Test

This test does NOT perform full training.

It verifies:
    - V3.1 train dataset loading
    - V3.1 dev dataset loading
    - feature tensor shape
    - DeepShieldV3 model creation
    - CUDA availability
    - forward pass
    - loss calculation
    - one backward pass
    - AMP compatibility
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# Paths
# ============================================================

V31_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

V3_ROOT = (
    V31_ROOT.parent
    / "v3"
)

if str(V31_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(V31_ROOT),
    )

if str(V3_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(V3_ROOT),
    )


# ============================================================
# Imports
# ============================================================

from dataset.v3_1_dataset import (
    DeepShieldV31Dataset,
)

from model.v3_model import (
    DeepShieldV3,
    count_parameters,
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32

NUM_CLASSES = 3

EXPECTED_TRAIN_SAMPLES = 7224

EXPECTED_DEV_SAMPLES = 1158

EXPECTED_FEATURE_SHAPE = (
    1,
    128,
    251,
)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DeepShield V3.1 - Pre-Training Smoke Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print()
    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "WARNING: CUDA is not available."
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("1. Loading datasets")
    print("=" * 70)

    train_dataset = DeepShieldV31Dataset(
        "train"
    )

    dev_dataset = DeepShieldV31Dataset(
        "dev"
    )

    print()
    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Dev samples: "
        f"{len(dev_dataset)}"
    )

    if len(train_dataset) != EXPECTED_TRAIN_SAMPLES:

        raise RuntimeError(
            "Unexpected V3.1 train dataset size. "
            f"Expected {EXPECTED_TRAIN_SAMPLES}, "
            f"received {len(train_dataset)}."
        )

    if len(dev_dataset) != EXPECTED_DEV_SAMPLES:

        raise RuntimeError(
            "Unexpected V3 dev dataset size. "
            f"Expected {EXPECTED_DEV_SAMPLES}, "
            f"received {len(dev_dataset)}."
        )

    print(
        "Dataset sizes: PASS"
    )

    # --------------------------------------------------------
    # Inspect samples
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("2. Checking feature tensors")
    print("=" * 70)

    train_features, train_label = (
        train_dataset[0]
    )

    dev_features, dev_label = (
        dev_dataset[0]
    )

    print()
    print(
        f"Train feature shape: "
        f"{tuple(train_features.shape)}"
    )

    print(
        f"Train feature dtype: "
        f"{train_features.dtype}"
    )

    print(
        f"Train label: "
        f"{train_label}"
    )

    print()
    print(
        f"Dev feature shape: "
        f"{tuple(dev_features.shape)}"
    )

    print(
        f"Dev feature dtype: "
        f"{dev_features.dtype}"
    )

    print(
        f"Dev label: "
        f"{dev_label}"
    )

    if (
        tuple(train_features.shape)
        != EXPECTED_FEATURE_SHAPE
    ):

        raise RuntimeError(
            "Unexpected train feature shape."
        )

    if (
        tuple(dev_features.shape)
        != EXPECTED_FEATURE_SHAPE
    ):

        raise RuntimeError(
            "Unexpected dev feature shape."
        )

    if train_features.dtype != torch.float32:

        raise RuntimeError(
            "Train features must be float32."
        )

    if dev_features.dtype != torch.float32:

        raise RuntimeError(
            "Dev features must be float32."
        )

    print()
    print(
        "Feature validation: PASS"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3. Creating DataLoader")
    print("=" * 70)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    print()
    print(
        f"Train batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Dev batches: "
        f"{len(dev_loader)}"
    )

    # --------------------------------------------------------
    # Get one batch
    # --------------------------------------------------------

    train_batch_features, train_batch_labels = (
        next(
            iter(train_loader)
        )
    )

    print()
    print(
        f"Batch feature shape: "
        f"{tuple(train_batch_features.shape)}"
    )

    print(
        f"Batch label shape: "
        f"{tuple(train_batch_labels.shape)}"
    )

    expected_batch_shape = (
        BATCH_SIZE,
        1,
        128,
        251,
    )

    if (
        tuple(train_batch_features.shape)
        != expected_batch_shape
    ):

        raise RuntimeError(
            "Unexpected training batch shape. "
            f"Expected {expected_batch_shape}, "
            f"received "
            f"{tuple(train_batch_features.shape)}."
        )

    print()
    print(
        "DataLoader: PASS"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("4. Creating DeepShieldV3")
    print("=" * 70)

    model = DeepShieldV3(
        num_classes=NUM_CLASSES
    )

    print()
    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    print(
        "Initialization: RANDOM V3 WEIGHTS"
    )

    model = model.to(
        device
    )

    print()
    print(
        "Model creation: PASS"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.0001,
    )

    # --------------------------------------------------------
    # One forward pass
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("5. Testing forward pass")
    print("=" * 70)

    features = train_batch_features.to(
        device,
        non_blocking=True,
    )

    labels = train_batch_labels.to(
        device,
        non_blocking=True,
    )

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    if device.type == "cuda":

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):

            outputs = model(
                features
            )

            loss = criterion(
                outputs,
                labels,
            )

    else:

        outputs = model(
            features
        )

        loss = criterion(
            outputs,
            labels,
        )

    print()
    print(
        f"Input shape: "
        f"{tuple(features.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(outputs.shape)}"
    )

    print(
        f"Output dtype: "
        f"{outputs.dtype}"
    )

    print(
        f"Loss: "
        f"{loss.item():.6f}"
    )

    expected_output_shape = (
        BATCH_SIZE,
        NUM_CLASSES,
    )

    if (
        tuple(outputs.shape)
        != expected_output_shape
    ):

        raise RuntimeError(
            "Unexpected model output shape. "
            f"Expected {expected_output_shape}, "
            f"received {tuple(outputs.shape)}."
        )

    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(
            "Loss is NaN or infinite."
        )

    print()
    print(
        "Forward pass: PASS"
    )

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("6. Testing backward pass")
    print("=" * 70)

    loss.backward()

    gradient_count = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            if not torch.isfinite(
                parameter.grad
            ).all():

                raise RuntimeError(
                    "Non-finite gradient detected."
                )

            gradient_count += 1

    print()
    print(
        f"Parameters with gradients: "
        f"{gradient_count}"
    )

    if gradient_count == 0:

        raise RuntimeError(
            "No gradients were generated."
        )

    print()
    print(
        "Backward pass: PASS"
    )

    # --------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("7. Testing optimizer step")
    print("=" * 70)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        5.0,
    )

    optimizer.step()

    print()
    print(
        "Optimizer step: PASS"
    )

    # --------------------------------------------------------
    # AMP test
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("8. Testing AMP")
    print("=" * 70)

    if device.type == "cuda":

        scaler = torch.amp.GradScaler(
            "cuda"
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):

            amp_outputs = model(
                features
            )

            amp_loss = criterion(
                amp_outputs,
                labels,
            )

        scaler.scale(
            amp_loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        print()
        print(
            f"AMP loss: "
            f"{amp_loss.item():.6f}"
        )

        if not torch.isfinite(
            amp_loss
        ):

            raise RuntimeError(
                "AMP loss is NaN or infinite."
            )

        print()
        print(
            "AMP: PASS"
        )

    else:

        print()
        print(
            "AMP: SKIPPED because CUDA is unavailable."
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V3.1 PRE-TRAINING SMOKE TEST: PASS")
    print("=" * 70)

    print()
    print(
        "The V3.1 pipeline is ready for full training."
    )

    print()
    print(
        "No full training was performed."
    )

    print(
        "No V3 checkpoints were modified."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()