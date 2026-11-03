import torch
import torch.nn as nn


class DeepShieldCNNPool(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1, 32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),

            # Block 2
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),

            # Block 3
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.30),

            # Block 4
            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Two complementary pooling operations
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.max_pool = nn.AdaptiveMaxPool2d(
            (1, 1)
        )

        # 256 average + 256 maximum = 512
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                512,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.40
            ),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        avg = self.avg_pool(x)
        maximum = self.max_pool(x)

        x = torch.cat(
            [
                avg,
                maximum,
            ],
            dim=1,
        )

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("DeepShield Dual-Pooling CNN Test")
    print("=" * 70)

    print(
        "Device:",
        device,
    )

    model = DeepShieldCNNPool(
        num_classes=3
    ).to(device)

    test_input = torch.randn(
        2,
        1,
        128,
        251,
        device=device,
    )

    output = model(
        test_input
    )

    print(
        "Input shape:",
        test_input.shape,
    )

    print(
        "Output shape:",
        output.shape,
    )

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )

    print("=" * 70)