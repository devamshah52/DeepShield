"""
DeepShield V2-A
Multi-Scale Residual CNN + SE Attention

Input:
    [B, 1, 128, 251]

Classes:
    0 = Genuine
    1 = TTS
    2 = Voice Conversion

This model is an experimental V2 model.
It does NOT replace the existing DeepShield V1 model.
"""

import torch
import torch.nn as nn


# ============================================================
# SE ATTENTION BLOCK
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape

        scale = self.pool(x).view(batch, channels)
        scale = self.fc(scale).view(batch, channels, 1, 1)

        return x * scale


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = (
            nn.Dropout2d(dropout)
            if dropout > 0
            else nn.Identity()
        )

        self.se = SEBlock(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out = out + identity
        out = self.relu(out)

        return out


# ============================================================
# MULTI-SCALE FEATURE BLOCK
# ============================================================

class MultiScaleBlock(nn.Module):
    """
    Extracts features using both 3x3 and 5x5 receptive fields.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        branch_channels = out_channels // 2

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x3 = self.branch3(x)
        x5 = self.branch5(x)

        out = torch.cat([x3, x5], dim=1)

        return self.fusion(out)


# ============================================================
# DEEPSHIELD V2-A
# ============================================================

class DeepShieldV2(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Multi-scale feature extraction
        # ----------------------------------------------------

        self.multi_scale = MultiScaleBlock(
            in_channels=32,
            out_channels=64
        )

        self.pool1 = nn.MaxPool2d(2)

        # ----------------------------------------------------
        # Residual stage 1
        # ----------------------------------------------------

        self.stage1 = nn.Sequential(
            ResidualBlock(
                64,
                64,
                dropout=0.10
            ),
            ResidualBlock(
                64,
                64,
                dropout=0.10
            )
        )

        self.pool2 = nn.MaxPool2d(2)

        # ----------------------------------------------------
        # Residual stage 2
        # ----------------------------------------------------

        self.stage2 = nn.Sequential(
            ResidualBlock(
                64,
                128,
                stride=1,
                dropout=0.15
            ),
            ResidualBlock(
                128,
                128,
                dropout=0.15
            )
        )

        self.pool3 = nn.MaxPool2d(2)

        # ----------------------------------------------------
        # Residual stage 3
        # ----------------------------------------------------

        self.stage3 = nn.Sequential(
            ResidualBlock(
                128,
                256,
                stride=1,
                dropout=0.20
            ),
            ResidualBlock(
                256,
                256,
                dropout=0.20
            )
        )

        # ----------------------------------------------------
        # Dual global pooling
        # ----------------------------------------------------

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # [B, 1, 128, 251]
        x = self.stem(x)

        # Multi-scale features
        x = self.multi_scale(x)
        x = self.pool1(x)

        # Residual stages
        x = self.stage1(x)
        x = self.pool2(x)

        x = self.stage2(x)
        x = self.pool3(x)

        x = self.stage3(x)

        # Average pooling
        avg = self.avg_pool(x)
        avg = torch.flatten(avg, 1)

        # Maximum pooling
        maximum = self.max_pool(x)
        maximum = torch.flatten(maximum, 1)

        # Combine both representations
        x = torch.cat([avg, maximum], dim=1)

        # Classification
        x = self.classifier(x)

        return x


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("DeepShield V2-A Model Test")
    print("=" * 70)

    model = DeepShieldV2(num_classes=3).to(device)

    # Test input matching DeepShield's real feature shape
    test_input = torch.randn(
        2,
        1,
        128,
        251,
        device=device
    )

    with torch.no_grad():
        output = model(test_input)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Device          : {device}")
    print(f"Input shape      : {tuple(test_input.shape)}")
    print(f"Output shape     : {tuple(output.shape)}")
    print(f"Total parameters : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")

    print()
    print("Class mapping:")
    print("  0 = Genuine")
    print("  1 = TTS")
    print("  2 = Voice Conversion")

    print()
    print("Model test completed successfully.")
    print("=" * 70)