
#2026
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .mie import MotionStructureInteractionEncoder
from .prd import HierarchicalCondition, ProgressiveRefinementDecoder


class HSCD(nn.Module):
    """Composed HSCD network with explicit condition caching."""

    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 2,
        output_dim: int = 3,
        feature_dim: int = 256,
        mie_layers: int = 6,
        attention_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        decoder_channels: Sequence[int] = (64, 96, 128, 160),
        decoder_blocks_per_scale: int = 1,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.output_dim = output_dim
        self.mie = MotionStructureInteractionEncoder(
            num_joints=num_joints,
            input_dim=input_dim,
            feature_dim=feature_dim,
            layers=mie_layers,
            heads=attention_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
        )
        self.prd = ProgressiveRefinementDecoder(
            pose_dim=output_dim,
            condition_dim=feature_dim,
            channels=decoder_channels,
            attention_heads=attention_heads,
            blocks_per_scale=decoder_blocks_per_scale,
            dropout=dropout,
        )

    def encode(self, pose_2d: torch.Tensor) -> HierarchicalCondition:
        """Compute and project observation evidence once for reverse diffusion."""
        return self.prd.prepare_condition(self.mie(pose_2d))

    def denoise(
        self,
        noisy_pose: torch.Tensor,
        timesteps: torch.Tensor,
        condition: HierarchicalCondition,
    ) -> torch.Tensor:
        return self.prd(noisy_pose, timesteps, condition)

    def forward(
        self,
        pose_2d: torch.Tensor,
        noisy_pose: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.encode(pose_2d)
        return self.denoise(noisy_pose, timesteps, condition)
