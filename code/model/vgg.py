import sys
from typing import List, Optional

import torch.nn as nn

try:
    sys.path.insert(0, '../')
finally:
    pass


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv(x))
        return x


class VGGBase(nn.Module):

    def __init__(self, n_blocks: List[int], n_channels: List[int],
                 img_channels: int = 3):
        super().__init__()
        blocks = []

        pre_channels = img_channels
        for i, channels in enumerate(n_channels):
            blocks.append(VGGBlock(
                pre_channels, channels
            ))
            pre_channels = channels

            for _ in range(n_blocks[i] - 1):
                blocks.append(VGGBlock(
                    channels, channels
                ))
            blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        n_channels = [64, 128, 256, 512, 512]
        n_blocks = [2, 2, 3, 3, 3]

        self.backbone = VGGBase(
            n_blocks, n_channels
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


class VGG19(nn.Module):

    def __init__(self):
        super().__init__()
        n_channels = [64, 128, 256, 512, 512]
        n_blocks = [2, 2, 4, 4, 4]

        self.backbone = VGGBase(
            n_blocks, n_channels
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


class CancerVGGNet(nn.Module):

    def __init__(self, output_dim: int, n_blocks: List[int],
                 n_channels: List[int], img_channels: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.backbone = VGGBase(
            n_blocks, n_channels, img_channels
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(4096)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.LazyLinear(4096)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.LazyLinear(output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.dropout1(self.act1(self.fc1(x)))
        x = self.dropout2(self.act2(self.fc2(x)))
        logit = self.fc3(x)
        return logit
