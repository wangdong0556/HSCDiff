
#you can run this with diff 
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(indices: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Standard sinusoidal embedding for frame or diffusion indices."""
    if indices.ndim == 0:
        indices = indices[None]
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=indices.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = indices.float().unsqueeze(-1) * frequencies
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def human36m_adjacency(num_joints: int = 17) -> torch.Tensor:
    """Return a normalized Human3.6M-style skeletal adjacency matrix."""
    if num_joints != 17:
        return torch.eye(num_joints)
    edges = (
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16),
    )
    adjacency = torch.eye(num_joints)
    for left, right in edges:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(dim=-1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = max(dim, int(round(dim * ratio)))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalConvBlock(nn.Module):
    """Residual temporal block modulated by the diffusion-step embedding."""

    def __init__(self, channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.time_projection = nn.Linear(time_dim, channels * 2)
        self.depthwise = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        batch, frames, joints, channels = x.shape
        residual = x
        x = self.norm(x)
        scale, shift = self.time_projection(time_embedding).chunk(2, dim=-1)
        x = x * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
        x = x.permute(0, 2, 3, 1).reshape(batch * joints, channels, frames)
        x = self.pointwise(F.gelu(self.depthwise(x)))
        x = x.reshape(batch, joints, channels, frames).permute(0, 3, 1, 2)
        return residual + self.dropout(x)


class TemporalDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv1d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, joints, channels = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch * joints, channels, frames)
        x = self.projection(x)
        out_frames = x.shape[-1]
        return x.reshape(batch, joints, -1, out_frames).permute(0, 3, 1, 2)


class TemporalUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv1d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        batch, frames, joints, channels = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch * joints, channels, frames)
        x = F.interpolate(x, size=target_frames, mode="linear", align_corners=False)
        x = self.projection(x)
        return x.reshape(batch, joints, -1, target_frames).permute(0, 3, 1, 2)


class TemporalCrossAttention(nn.Module):
    """Cross-attention over the joint temporal-token axis.

    The class name is retained for API stability, but both temporal and joint
    dimensions are flattened as specified by the manuscript. With
    ``need_weights=False``, current PyTorch versions dispatch to a
    memory-efficient scaled-dot-product attention kernel when available.
    """

    def __init__(self, query_dim: int, condition_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.condition_projection = nn.Linear(condition_dim, query_dim)
        self.query_norm = nn.LayerNorm(query_dim)
        self.condition_norm = nn.LayerNorm(query_dim)
        self.attention = nn.MultiheadAttention(query_dim, heads, dropout=dropout, batch_first=True)

    def forward(self, query: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, frames, joints, channels = query.shape
        condition = self.condition_projection(condition)
        q = self.query_norm(query).reshape(batch, frames * joints, channels)
        kv = self.condition_norm(condition).reshape(batch, condition.shape[1] * joints, channels)
        attended, _ = self.attention(q, kv, kv, need_weights=False)
        attended = attended.reshape(batch, frames, joints, channels)
        return query + attended
