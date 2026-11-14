"""VoxelCNN: a small 3D convolutional network (VoxNet-style).

Same idea as the MNIST CNN, but every Conv2d becomes a Conv3d and the
filters slide through width, height AND depth.

    1x32x32x32 -> conv5 stride2 -> 32x16^3 -> conv3 -> pool -> 64x8^3
               -> conv3 -> pool -> 128x4^3 -> dense 256 -> dense 8 (one per fashion type)
"""
from __future__ import annotations

import torch
from torch import nn


class VoxelCNN(nn.Module):
    def __init__(self, n_classes: int = 8, grid: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        side = grid // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * side ** 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B,1,G,G,G)
        return self.classifier(self.features(x))          # logits (B, n_classes)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch can't see an NVIDIA GPU. Use --device cpu.")
    return torch.device(name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
