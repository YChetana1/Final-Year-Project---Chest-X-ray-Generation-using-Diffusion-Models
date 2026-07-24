import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.output_layer = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        pooled1 = self.pool1(enc1)

        enc2 = self.encoder2(pooled1)
        pooled2 = self.pool2(enc2)

        bottleneck = self.bottleneck(pooled2)

        up1 = self.up1(bottleneck)
        merged1 = torch.cat([up1, enc2], dim=1)
        dec1 = self.decoder1(merged1)

        up2 = self.up2(dec1)
        merged2 = torch.cat([up2, enc1], dim=1)
        dec2 = self.decoder2(merged2)

        output = self.output_layer(dec2)

        return output