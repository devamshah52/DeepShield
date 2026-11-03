"""
DeepShield V3 - CNN Model

Input:
    (batch, 1, 128, 251)

Output:
    3 classes

Classes:
    0 = Genuine
    1 = TTS
    2 = Voice_Conversion

This is a completely new V3 architecture.
It does NOT load V1/V2 weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 3

CLASS_NAMES = {
    0: "Genuine",
    1: "TTS",
    2: "Voice_Conversion",
}


# ============================================================
# ConvNormAct
# ============================================================

class ConvNormAct(nn.Module):
    """
    Convolution + BatchNorm + SiLU.

    This is the basic building block used throughout V3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ):

        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


# ============================================================
# SE Block
# ============================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Allows the network to learn which feature channels
    are important for distinguishing Genuine, TTS and VC.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
    ):

        super().__init__()

        reduced_channels = max(
            channels // reduction,
            4,
        )

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                reduced_channels,
                kernel_size=1,
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                reduced_channels,
                channels,
                kernel_size=1,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        scale = self.pool(x)

        scale = self.fc(
            scale
        )

        return x * scale


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(nn.Module):
    """
    Two-convolution residual block with SE attention.

    If input and output channels differ, a projection
    is used for the residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ):

        super().__init__()

        self.conv1 = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),
        )

        self.se = SEBlock(
            out_channels
        )

        self.dropout = nn.Dropout2d(
            dropout
        ) if dropout > 0 else nn.Identity()

        if (
            in_channels != out_channels
            or stride != 1
        ):

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),

                nn.BatchNorm2d(
                    out_channels
                ),
            )

        else:

            self.shortcut = nn.Identity()

        self.activation = nn.SiLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.shortcut(
            x
        )

        out = self.conv1(
            x
        )

        out = self.conv2(
            out
        )

        out = self.se(
            out
        )

        out = self.dropout(
            out
        )

        out = out + residual

        out = self.activation(
            out
        )

        return out


# ============================================================
# Multi-Scale Block
# ============================================================

class MultiScaleBlock(nn.Module):
    """
    Multi-scale convolution block.

    Uses parallel 3x3 and 5x5 convolutions so the network
    can learn both local and wider spectro-temporal patterns.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):

        super().__init__()

        branch_channels = (
            out_channels // 2
        )

        self.branch3 = nn.Sequential(

            ConvNormAct(
                in_channels,
                branch_channels,
                kernel_size=3,
            ),

            ConvNormAct(
                branch_channels,
                branch_channels,
                kernel_size=3,
            ),
        )

        self.branch5 = nn.Sequential(

            ConvNormAct(
                in_channels,
                branch_channels,
                kernel_size=5,
            ),

            ConvNormAct(
                branch_channels,
                branch_channels,
                kernel_size=5,
            ),
        )

        self.fuse = ConvNormAct(
            out_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

        self.se = SEBlock(
            out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        branch3 = self.branch3(
            x
        )

        branch5 = self.branch5(
            x
        )

        out = torch.cat(
            [
                branch3,
                branch5,
            ],
            dim=1,
        )

        out = self.fuse(
            out
        )

        out = self.se(
            out
        )

        return out


# ============================================================
# V3 Model
# ============================================================

class DeepShieldV3(nn.Module):
    """
    DeepShield V3 CNN.

    Architecture:

        Input
          |
        Stem
          |
        Multi-scale block
          |
        Residual Stage 1
          |
        Downsample
          |
        Residual Stage 2
          |
        Downsample
          |
        Residual Stage 3
          |
        Downsample
          |
        Residual Stage 4
          |
        Global Average + Max Pool
          |
        Feature Fusion
          |
        Classifier
          |
        3 classes
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
    ):

        super().__init__()

        self.num_classes = num_classes

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        self.stem = nn.Sequential(

            ConvNormAct(
                1,
                32,
                kernel_size=3,
            ),

            ConvNormAct(
                32,
                32,
                kernel_size=3,
            ),
        )

        # ----------------------------------------------------
        # Multi-scale front-end
        # ----------------------------------------------------

        self.multi_scale = MultiScaleBlock(
            32,
            64,
        )

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        self.stage1 = nn.Sequential(

            ResidualBlock(
                64,
                64,
                stride=1,
                dropout=0.05,
            ),

            ResidualBlock(
                64,
                64,
                stride=1,
                dropout=0.05,
            ),
        )

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        self.stage2 = nn.Sequential(

            ResidualBlock(
                64,
                128,
                stride=2,
                dropout=0.08,
            ),

            ResidualBlock(
                128,
                128,
                stride=1,
                dropout=0.08,
            ),
        )

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        self.stage3 = nn.Sequential(

            ResidualBlock(
                128,
                192,
                stride=2,
                dropout=0.10,
            ),

            ResidualBlock(
                192,
                192,
                stride=1,
                dropout=0.10,
            ),
        )

        # ----------------------------------------------------
        # Stage 4
        # ----------------------------------------------------

        self.stage4 = nn.Sequential(

            ResidualBlock(
                192,
                256,
                stride=2,
                dropout=0.12,
            ),

            ResidualBlock(
                256,
                256,
                stride=1,
                dropout=0.12,
            ),
        )

        # ----------------------------------------------------
        # Global pooling
        # ----------------------------------------------------

        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.max_pool = (
            nn.AdaptiveMaxPool2d(1)
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                256,
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Dropout(
                0.35
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                128,
                num_classes,
            ),
        )

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------

        self._initialize_weights()

    # ========================================================
    # Weight initialization
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d,
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

            elif isinstance(
                module,
                nn.BatchNorm1d,
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

            elif isinstance(
                module,
                nn.Linear,
            ):

                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "DeepShieldV3 expects a 4D tensor "
                "(batch, channels, mel, time). "
                f"Received shape: {tuple(x.shape)}"
            )

        if x.shape[1] != 1:

            raise ValueError(
                "DeepShieldV3 expects 1 input channel. "
                f"Received: {x.shape[1]}"
            )

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        x = self.stem(
            x
        )

        x = self.multi_scale(
            x
        )

        x = self.stage1(
            x
        )

        x = self.stage2(
            x
        )

        x = self.stage3(
            x
        )

        x = self.stage4(
            x
        )

        # ----------------------------------------------------
        # Dual global pooling
        # ----------------------------------------------------

        avg = self.avg_pool(
            x
        )

        max_value = self.max_pool(
            x
        )

        avg = torch.flatten(
            avg,
            start_dim=1,
        )

        max_value = torch.flatten(
            max_value,
            start_dim=1,
        )

        x = torch.cat(
            [
                avg,
                max_value,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        x = self.classifier(
            x
        )

        return x


# ============================================================
# Model information
# ============================================================

def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def print_model_info(
    model: nn.Module,
):

    total_parameters = (
        count_parameters(model)
    )

    print()
    print("=" * 70)
    print("DeepShield V3 Model")
    print("=" * 70)

    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Number of classes: "
        f"{model.num_classes}"
    )

    print()
    print("Classes:")

    for label_id in range(
        NUM_CLASSES
    ):

        print(
            f"  {label_id}: "
            f"{CLASS_NAMES[label_id]}"
        )

    print("=" * 70)


# ============================================================
# Model test
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DeepShield V3 - Model Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = DeepShieldV3(
        num_classes=NUM_CLASSES
    )

    print_model_info(
        model
    )

    # --------------------------------------------------------
    # Confirm model starts from scratch
    # --------------------------------------------------------

    print()
    print(
        "Initialization: RANDOM V3 WEIGHTS"
    )

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    test_input = torch.randn(
        2,
        1,
        128,
        251,
    )

    model = model.to(
        device
    )

    test_input = test_input.to(
        device
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        output = model(
            test_input
        )

    print()
    print(
        f"Input shape : "
        f"{tuple(test_input.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        f"Output dtype: "
        f"{output.dtype}"
    )

    # --------------------------------------------------------
    # Output validation
    # --------------------------------------------------------

    expected_shape = (
        2,
        NUM_CLASSES,
    )

    if (
        tuple(output.shape)
        != expected_shape
    ):

        raise RuntimeError(
            "Unexpected model output shape: "
            f"{tuple(output.shape)}. "
            f"Expected {expected_shape}."
        )

    if not torch.isfinite(
        output
    ).all():

        raise RuntimeError(
            "Model output contains "
            "NaN or infinite values."
        )

    # --------------------------------------------------------
    # Softmax test
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=1,
    )

    print()
    print(
        "Probability sums:"
    )

    print(
        probabilities.sum(
            dim=1
        ).detach().cpu()
    )

    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(
            probabilities.shape[0],
            device=device,
        ),
        atol=1e-5,
    ):

        raise RuntimeError(
            "Softmax probabilities "
            "do not sum to 1."
        )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    print()
    print(
        f"Total trainable parameters: "
        f"{count_parameters(model):,}"
    )

    print()
    print(
        "V3 MODEL TEST: PASS"
    )


if __name__ == "__main__":

    main()