"""
DeepShield V3.2 - Pre-Training Smoke Test

Checks:
    1. CUDA availability
    2. V3.2 train/dev dataset loading
    3. DataLoader batch shapes
    4. DeepShieldV3 forward pass
    5. Loss calculation
    6. Backward pass
    7. Optimizer step
    8. AMP forward/backward
    9. Explicit gTTS batch test

IMPORTANT:
    This script does NOT perform real training.
    No checkpoint is saved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"C:\DeepShield")

BACKEND_DIR = (
    PROJECT_ROOT
    / "backend"
)

V3_ROOT = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3"
)

V31_ROOT = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3_1"
)


# ============================================================
# Python paths
# ============================================================

if str(BACKEND_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(BACKEND_DIR)
    )

if str(V3_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(V3_ROOT)
    )


# ============================================================
# Imports
# ============================================================

from ml.experiments.v3.model.v3_model import (
    DeepShieldV3,
)

from ml.experiments.v3_1.dataset.v3_2_dataset import (
    DeepShieldV32Dataset,
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32

LEARNING_RATE = 0.0005

EXPECTED_PARAMETERS = 4_492_939

EXPECTED_BATCH_SHAPE = (
    32,
    1,
    128,
    251,
)

NUM_CLASSES = 3


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "DeepShield V3.2 - Pre-Training Smoke Test"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    print()
    print("Device check...")

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "V3.2 training requires CUDA."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"  Device: {device}"
    )

    print(
        f"  GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"  CUDA version: "
        f"{torch.version.cuda}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("Loading V3.2 datasets...")

    train_dataset = (
        DeepShieldV32Dataset(
            "train"
        )
    )

    dev_dataset = (
        DeepShieldV32Dataset(
            "dev"
        )
    )

    print()
    print(
        f"  Train samples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"  Dev samples: "
        f"{len(dev_dataset):,}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print()
    print("Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"  Train batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"  Dev batches: "
        f"{len(dev_loader):,}"
    )

    # --------------------------------------------------------
    # Get normal training batch
    # --------------------------------------------------------

    print()
    print("Testing training batch...")

    train_features, train_labels = next(
        iter(train_loader)
    )

    print(
        f"  Features: "
        f"{tuple(train_features.shape)}"
    )

    print(
        f"  Labels: "
        f"{tuple(train_labels.shape)}"
    )

    if tuple(train_features.shape) != (
        EXPECTED_BATCH_SHAPE
    ):

        raise RuntimeError(
            "Unexpected training batch shape: "
            f"{tuple(train_features.shape)}"
        )

    if train_labels.dtype != torch.long:

        raise RuntimeError(
            "Training labels are not torch.long."
        )

    print(
        "  Batch shape: PASS"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Creating DeepShieldV3 model...")

    model = DeepShieldV3(
        num_classes=NUM_CLASSES
    )

    model = model.to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"  Trainable parameters: "
        f"{parameter_count:,}"
    )

    if parameter_count != EXPECTED_PARAMETERS:

        raise RuntimeError(
            "Unexpected model parameter count.\n"
            f"Expected: {EXPECTED_PARAMETERS:,}\n"
            f"Found:    {parameter_count:,}"
        )

    print(
        "  Model architecture: PASS"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    # --------------------------------------------------------
    # Move batch to GPU
    # --------------------------------------------------------

    train_features = (
        train_features
        .to(
            device,
            non_blocking=True,
        )
    )

    train_labels = (
        train_labels
        .to(
            device,
            non_blocking=True,
        )
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print()
    print("Testing forward pass...")

    optimizer.zero_grad(
        set_to_none=True
    )

    outputs = model(
        train_features
    )

    print(
        f"  Output shape: "
        f"{tuple(outputs.shape)}"
    )

    expected_output_shape = (
        BATCH_SIZE,
        NUM_CLASSES,
    )

    if tuple(outputs.shape) != (
        expected_output_shape
    ):

        raise RuntimeError(
            "Unexpected model output shape: "
            f"{tuple(outputs.shape)}"
        )

    print(
        "  Forward pass: PASS"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = criterion(
        outputs,
        train_labels,
    )

    print(
        f"  Loss: "
        f"{loss.item():.6f}"
    )

    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(
            "Loss is not finite."
        )

    print(
        "  Loss calculation: PASS"
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    print()
    print("Testing backward pass...")

    loss.backward()

    gradient_found = False

    for parameter in model.parameters():

        if (
            parameter.grad is not None
        ):

            if not torch.isfinite(
                parameter.grad
            ).all():

                raise RuntimeError(
                    "Non-finite gradient detected."
                )

            gradient_found = True

    if not gradient_found:

        raise RuntimeError(
            "No gradients were produced."
        )

    print(
        "  Backward pass: PASS"
    )

    # --------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------

    print()
    print("Testing optimizer step...")

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0,
    )

    optimizer.step()

    print(
        "  Optimizer step: PASS"
    )

    # --------------------------------------------------------
    # AMP test
    # --------------------------------------------------------

    print()
    print("Testing CUDA AMP...")

    amp_model = DeepShieldV3(
        num_classes=NUM_CLASSES
    ).to(
        device
    )

    amp_optimizer = AdamW(
        amp_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    amp_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    amp_optimizer.zero_grad(
        set_to_none=True
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    ):

        amp_outputs = amp_model(
            train_features
        )

        amp_loss = criterion(
            amp_outputs,
            train_labels,
        )

    print(
        f"  AMP loss: "
        f"{amp_loss.item():.6f}"
    )

    if not torch.isfinite(
        amp_loss
    ):

        raise RuntimeError(
            "AMP loss is not finite."
        )

    amp_scaler.scale(
        amp_loss
    ).backward()

    amp_scaler.unscale_(
        amp_optimizer
    )

    torch.nn.utils.clip_grad_norm_(
        amp_model.parameters(),
        max_norm=5.0,
    )

    amp_scaler.step(
        amp_optimizer
    )

    amp_scaler.update()

    print(
        "  AMP: PASS"
    )

    # --------------------------------------------------------
    # gTTS batch test
    # --------------------------------------------------------

    print()
    print("Testing gTTS batch...")

    gtts_indices = [
        index
        for index, source
        in enumerate(
            train_dataset.manifest[
                "source"
            ].tolist()
        )
        if source == "gtts"
    ]

    if not gtts_indices:

        raise RuntimeError(
            "No gTTS samples found."
        )

    # Take up to one complete batch
    # of gTTS samples.

    gtts_batch_indices = (
        gtts_indices[
            :BATCH_SIZE
        ]
    )

    gtts_features = []
    gtts_labels = []

    for index in gtts_batch_indices:

        feature, label = (
            train_dataset[index]
        )

        gtts_features.append(
            feature
        )

        gtts_labels.append(
            label
        )

    gtts_features = torch.stack(
        gtts_features
    ).to(
        device,
        non_blocking=True,
    )

    gtts_labels = torch.stack(
        gtts_labels
    ).to(
        device,
        non_blocking=True,
    )

    print(
        f"  gTTS batch shape: "
        f"{tuple(gtts_features.shape)}"
    )

    print(
        f"  gTTS labels: "
        f"{sorted(set(gtts_labels.tolist()))}"
    )

    if not torch.all(
        gtts_labels == 1
    ):

        raise RuntimeError(
            "gTTS batch contains "
            "a non-TTS label."
        )

    gtts_outputs = amp_model(
        gtts_features
    )

    print(
        f"  gTTS output shape: "
        f"{tuple(gtts_outputs.shape)}"
    )

    if tuple(gtts_outputs.shape) != (
        len(gtts_batch_indices),
        NUM_CLASSES,
    ):

        raise RuntimeError(
            "Unexpected gTTS output shape."
        )

    print(
        "  gTTS model forward: PASS"
    )

    # --------------------------------------------------------
    # Dev batch test
    # --------------------------------------------------------

    print()
    print("Testing development batch...")

    dev_features, dev_labels = next(
        iter(dev_loader)
    )

    print(
        f"  Dev features: "
        f"{tuple(dev_features.shape)}"
    )

    print(
        f"  Dev labels: "
        f"{tuple(dev_labels.shape)}"
    )

    dev_features = (
        dev_features
        .to(
            device,
            non_blocking=True,
        )
    )

    with torch.no_grad():

        dev_outputs = amp_model(
            dev_features
        )

    print(
        f"  Dev output shape: "
        f"{tuple(dev_outputs.shape)}"
    )

    if dev_outputs.shape[1] != (
        NUM_CLASSES
    ):

        raise RuntimeError(
            "Unexpected dev output shape."
        )

    print(
        "  Dev forward pass: PASS"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model
    del amp_model

    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "V3.2 PRE-TRAINING SMOKE TEST: PASS"
    )

    print("=" * 70)

    print()
    print(
        "Train samples: 9,024"
    )

    print(
        "Dev samples: 1,158"
    )

    print(
        "Model parameters: 4,492,939"
    )

    print(
        "Input shape: (32, 1, 128, 251)"
    )

    print(
        "Output shape: (32, 3)"
    )

    print(
        "AMP: PASS"
    )

    print(
        "gTTS batch: PASS"
    )

    print(
        "Test set: NOT USED"
    )


if __name__ == "__main__":

    main()