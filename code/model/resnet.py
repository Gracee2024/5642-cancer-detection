from typing import List, Optional

import torch
import torch.nn as nn
import sys

try:
    sys.path.insert(0, '../')
finally:
    pass


class ShortcutProjection(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + shortcut)


class BottleneckResidualBlock(nn.Module):

    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

        self.act3 = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.act3(x + shortcut)


class ResNetBase(nn.Module):
    def __init__(self, n_blocks: List[int], n_channels: List[int],
                 bottlenecks: Optional[List[int]] = None):
        super().__init__()

        blocks = []
        prev_channels = n_channels[0]

        """
        n_blocks = [2, 2, 2, 2]
        n_channels = [64, 128, 256, 512]
        """
        for i, channels in enumerate(n_channels):
            stride = 1 if len(blocks) == 0 else 2

            if bottlenecks is not None:
                blocks.append(BottleneckResidualBlock(
                    prev_channels, bottlenecks[i], channels, stride
                ))
            else:
                blocks.append(ResidualBlock(
                    prev_channels, channels, stride
                ))

            prev_channels = channels
            for _ in range(n_blocks[i] - 1):
                stride = 1
                if bottlenecks is not None:
                    blocks.append(BottleneckResidualBlock(
                        prev_channels, bottlenecks[i], channels, stride
                    ))

                else:
                    blocks.append(ResidualBlock(channels, channels, stride=stride))

        self.blocks = nn.Sequential(*blocks)

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x = self.bn(self.conv(x))
        # x = self.max_pool(x)
        x = self.blocks(x)
        # x = self.avg_pool(x).squeeze()
        return x
    
    
class ResNet18(nn.Module):
    
    def __init__(self):
        super().__init__()
        n_blocks = [2, 2, 2, 2]
        n_channels = [64, 128, 256, 512]

        self.backbone = ResNetBase(
            n_blocks, n_channels,
        )

    def forward(self, x):
        return self.backbone(x)


class ResNet34(nn.Module):

    def __init__(self):
        super().__init__()
        n_blocks = [3, 4, 6, 3]
        n_channels = [64, 128, 256, 512]

        self.backbone = ResNetBase(
            n_blocks, n_channels,
        )

    def forward(self, x):
        return self.backbone(x)


class ResNet50(nn.Module):
    
    def __init__(self):
        super().__init__()

        n_blocks = [3, 4, 6, 3]
        n_channels = [256, 512, 1024, 2048]
        bottlenecks = [64, 128, 256, 512]

        self.backbone = ResNetBase(
            n_blocks, n_channels, bottlenecks,
        )

    def forward(self, x):
        return self.backbone(x)


class ResNet101(nn.Module):

    def __init__(self):
        super().__init__()

        n_blocks = [3, 4, 23, 3]
        n_channels = [256, 512, 1024, 2048]
        bottlenecks = [64, 128, 256, 512]

        self.backbone = ResNetBase(
            n_blocks, n_channels, bottlenecks,
        )

    def forward(self, x):
        return self.backbone(x)


class ResNet152(nn.Module):

    def __init__(self):
        super().__init__()

        n_blocks = [3, 8, 36, 3]
        n_channels = [256, 512, 1024, 2048]
        bottlenecks = [64, 128, 256, 512]

        self.backbone = ResNetBase(
            n_blocks, n_channels, bottlenecks,
        )

    def forward(self, x):
        return self.backbone(x)


class CancerResNet(nn.Module):
    
    def __init__(self, output_dim: int, n_blocks: List[int], n_channels: List[int],
                 img_channels: int = 3, first_kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(img_channels, n_channels[0], kernel_size=first_kernel_size,
                              stride=2, padding=first_kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_channels[0])
        self.act = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.backbone = self.get_resnet_layer(n_blocks, n_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=n_channels[-1], out_features=output_dim)

    def get_resnet_layer(self, n_blocks: List[int], n_channels: List[int]):
        return ResNetBase(
            n_blocks, n_channels,
        )

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.act(x)
        x = self.max_pool(x)
        x = self.backbone(x)
        x = self.flatten(self.avg_pool(x))
        logit = self.fc(x)
        return logit



