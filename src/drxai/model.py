"""Backbone, output head and loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_model(backbone: str, head: str, num_classes: int, pretrained: bool = True,
                dropout: float = 0.2) -> nn.Module:
    """CORN needs K-1 outputs (one per threshold), plain softmax needs K."""
    n_out = num_classes - 1 if head == "corn" else num_classes
    if backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, n_out))
    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.classifier[1].in_features, n_out))
    elif backbone == "tiny":   # test-only stand-in so the smoke test runs on a CPU in seconds
        net = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4, padding=1), nn.ReLU(),
                            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, n_out))
    else:
        raise ValueError(f"unknown backbone {backbone!r}")
    return net


def gradcam_layer(net: nn.Module, backbone: str) -> nn.Module:
    """Last convolutional block: the deepest layer that still has spatial resolution."""
    if backbone == "resnet50":
        return net.layer4
    if backbone == "efficientnet_b0":
        return net.features[-1]
    if backbone == "tiny":
        return net[2]
    raise ValueError(backbone)


def corn_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CORN loss (Shi, Cao & Raschka, 2023).

    Task k learns P(y > k | y > k-1), so it is trained only on examples with y >= k.
    Conditioning on the earlier thresholds is what guarantees a monotone survival function.
    """
    total = logits.new_zeros(())
    n = 0
    for k in range(logits.shape[1]):
        mask = y >= k
        if mask.any():
            target = (y[mask] > k).float()
            total = total + F.binary_cross_entropy_with_logits(logits[mask, k], target, reduction="sum")
            n += int(mask.sum())
    return total / max(n, 1)


def loss_fn(head: str):
    return corn_loss if head == "corn" else F.cross_entropy


def corn_expected_grade(logits: torch.Tensor) -> torch.Tensor:
    """E[y] = sum_k P(y > k): a single differentiable severity score, used as the Grad-CAM target."""
    return torch.cumprod(torch.sigmoid(logits), dim=1).sum(dim=1)
