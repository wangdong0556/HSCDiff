#MIE process
from __future__ import annotations

import math

import torch
from torch import nn

from .layers import FeedForward, human36m_adjacency, sinusoidal_embedding


class SpatiotemporalAttentionBlock(nn.Module):
    """Factorized motion/structure attention followed by adaptive fusion."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_ratio: float,
        dropout: float,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(dim)
        self.spatial_norm = nn.LayerNorm(dim)
        self.temporal_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.spatial_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.spatial_adjacency_bias = nn.Parameter(adjacency.clone())
        self.fusion_gate = nn.Linear(dim * 2, dim)
        self.fusion_projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_ratio, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, joints, dim = tokens.shape

        temporal = self.temporal_norm(tokens).permute(0, 2, 1, 3)
        temporal = temporal.reshape(batch * joints, frames, dim)
        temporal, _ = self.temporal_attention(temporal, temporal, temporal, need_weights=False)
        temporal = temporal.reshape(batch, joints, frames, dim).permute(0, 2, 1, 3)

        spatial = self.spatial_norm(tokens).reshape(batch * frames, joints, dim)
        spatial, _ = self.spatial_attention(
            spatial,
            spatial,
            spatial,
            attn_mask=self.spatial_adjacency_bias,
            need_weights=False,
        )
        spatial = spatial.reshape(batch, frames, joints, dim)

        gate = torch.sigmoid(self.fusion_gate(torch.cat((temporal, spatial), dim=-1)))
        fused = gate * temporal + (1.0 - gate) * spatial
        tokens = tokens + self.dropout(self.fusion_projection(fused))
        return tokens + self.ffn(self.ffn_norm(tokens))


class MotionStructureInteractionEncoder(nn.Module):
    """MIE from the manuscript.

    The 2D observation is embedded once. Temporal attention follows each joint
    through the video, spatial attention models within-frame skeletal
    relations, and a learned gate combines both sources in every STAB.
    """

    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 2,
        feature_dim: int = 256,
        layers: int = 6,
        heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        self.input_projection = nn.Linear(input_dim, feature_dim)
        self.spatial_embedding = nn.Parameter(torch.empty(1, num_joints, feature_dim))
        self.topology_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        adjacency = human36m_adjacency(num_joints)
        self.register_buffer("adjacency", adjacency, persistent=True)
        self.blocks = nn.ModuleList(
            SpatiotemporalAttentionBlock(
                feature_dim, heads, ffn_ratio, dropout, adjacency
            )
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        repeated_topology = adjacency.repeat(1, math.ceil(feature_dim / num_joints))[
            :, :feature_dim
        ]
        with torch.no_grad():
            self.spatial_embedding.copy_(0.02 * repeated_topology.unsqueeze(0))
            self.spatial_embedding.add_(0.005 * torch.randn_like(self.spatial_embedding))

    def _position_encoding(self, frames: int, device: torch.device) -> torch.Tensor:
        temporal = sinusoidal_embedding(torch.arange(frames, device=device), self.feature_dim)
        temporal = temporal[None, :, None, :]
        topology = torch.einsum("ij,bjd->bid", self.adjacency, self.spatial_embedding)
        spatial = self.spatial_embedding + self.topology_projection(topology)
        return temporal + spatial[:, None, :, :]

    def forward(self, pose_2d: torch.Tensor) -> torch.Tensor:
        if pose_2d.ndim != 4:
            raise ValueError("MIE expects [batch, frames, joints, 2]")
        if pose_2d.shape[2] != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, received {pose_2d.shape[2]}")
        tokens = self.input_projection(pose_2d)
        tokens = tokens + self._position_encoding(tokens.shape[1], tokens.device)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)
