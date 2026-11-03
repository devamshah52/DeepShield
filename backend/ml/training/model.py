import torch
import torch.nn as nn


class DeepShieldCNN(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        # ====================================================
        # Convolutional Block 1
        # ====================================================

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(
                p=0.20
            ),

            # =================================================
            # Convolutional Block 2
            # =================================================

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(
                p=0.20
            ),

            # =================================================
            # Convolutional Block 3
            # =================================================

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(
                p=0.30
            ),

            # =================================================
            # Convolutional Block 4
            # =================================================

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        # ====================================================
        # Classifier
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.40
            ),

            nn.Linear(
                128,
                num_classes
            ),
        )

    # ========================================================
    # Forward pass
    # ========================================================

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# Model test
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print("DeepShield CNN Model Test")
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

    model = DeepShieldCNN(
        num_classes=3
    ).to(device)

    # --------------------------------------------------------
    # Create test input
    #
    # Shape:
    # [batch, channel, mel, time]
    # --------------------------------------------------------

    test_input = torch.randn(
        1,
        1,
        128,
        251,
        device=device,
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(
        test_input
    )

    print(
        "Input shape:",
        test_input.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    print(
        "Output:",
        output
    )

    print("=" * 60)