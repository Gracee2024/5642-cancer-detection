from typing import List, Optional
import torch
import torch.nn as nn
import sys

from math import floor

try:
    sys.path.insert(0, '../')
finally:
    pass


class Transition(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.act(self.bn(x)))
        x = self.max_pool(x)
        return x


class DenseBlock(nn.Module):

    def __init__(self, in_channels: int,
                 growth_rate: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=1)

    def forward(self, x):
        output = self.conv(self.act(self.bn(x)))
        output = torch.cat((x, output), 1)
        return output


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels: int,
                 growth_rate: int):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        output = self.conv1(self.act1(self.bn1(x)))
        output = self.conv2(self.act2(self.bn2(output)))
        output = torch.cat((x, output), 1)
        return output


class DenseBase(nn.Module):

    def __init__(self, n_blocks: List[int],
                 growth_rate: int,
                 theta: float = 1.0, bottlenecks: Optional[List[int]] = None):
        super().__init__()

        in_channels = 2 * growth_rate
        blocks = []
        for i in range(4):
            if bottlenecks is not None:
                blocks.extend(self._make_dense(in_channels, growth_rate, n_blocks[i], bottlenecks[i]))
            else:
                blocks.extend(self._make_dense(in_channels, growth_rate, n_blocks[i]))
            in_channels += n_blocks[i] * growth_rate
            out_channels = int(floor(in_channels * theta))
            if i != 3:
                blocks.append(
                    Transition(in_channels, out_channels)
                )
            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

    def _make_dense(self, in_channels: int, growth_rate: int, n_block: int, bottleneck: Optional[int] = None):
        blocks = []
        for i in range(n_block):
            if bottleneck is not None:
                blocks.append(
                    BottleneckBlock(in_channels, growth_rate)
                )
            else:
                blocks.append(
                    DenseBlock(in_channels, growth_rate)
                )
            in_channels += growth_rate
        return blocks

    def forward(self, x):
        x = self.blocks(x)
        return x


class DenseNet121(nn.Module):

    def __init__(self, growth_rate: int = 32, theta: float = 1.0):
        super().__init__()
        n_blocks = [6, 12, 24, 16]
        self.backbone = DenseBase(n_blocks, growth_rate, theta)

    def forward(self, x):
        return self.backbone(x)


class DenseNet169(nn.Module):

    def __init__(self, growth_rate: int = 32, theta: float = 1.0):
        super().__init__()
        n_blocks = [6, 12, 32, 32]
        self.backbone = DenseBase(n_blocks, growth_rate, theta)

    def forward(self, x):
        return self.backbone(x)


class DenseNet201(nn.Module):

    def __init__(self, growth_rate: int = 32, theta: float = 1.0):
        super().__init__()
        n_blocks = [6, 12, 48, 32]
        self.backbone = DenseBase(n_blocks, growth_rate, theta)

    def forward(self, x):
        return self.backbone(x)


class DenseNet264(nn.Module):

    def __init__(self, growth_rate: int = 32, theta: float = 1.0):
        super().__init__()
        n_blocks = [6, 12, 64, 64]
        self.backbone = DenseBase(n_blocks, growth_rate, theta)

    def forward(self, x):
        return self.backbone(x)


class CancerDenseNet(nn.Module):

    def __init__(self, output_dim: int,
                 n_blocks: List[int],
                 growth_rate: int,
                 theta: float,
                 bottlenecks: List[int] = None,
                 img_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(img_channels, 2*growth_rate, kernel_size=7, padding=3, stride=2)
        self.bn = nn.BatchNorm2d(2*growth_rate)
        self.act = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.backbone = DenseBase(n_blocks, growth_rate, theta, bottlenecks)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(output_dim)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.max_pool(x)
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        logit = self.fc(x)
        return logit
