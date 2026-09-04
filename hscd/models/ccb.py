from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

  
class CrossScaleConditioningBridge(nn.Module):
    """Scale-align and state-adapt MIE evidence for one PRD level."""

    def __init__(self, condition_dim: int, state_dim: int) -> None:
        super().__init__()
        self.condition_projection = nn.Linear(condition_dim, state_dim)
        self.state_gate = nn.Linear(state_dim, state_dim)
        self.condition_gate = nn.Linear(state_dim, state_dim)
        self.gate_bias = nn.Parameter(torch.zeros(state_dim))

    @staticmethod
    def pool_condition(condition: torch.Tensor, target_frames: int) -> torch.Tensor:
        batch, frames, joints, channels = condition.shape
        pooled = condition.permute(0, 2, 3, 1).reshape(batch * joints, channels, frames)
        pooled = F.adaptive_avg_pool1d(pooled, target_frames)
        return pooled.reshape(batch, joints, channels, target_frames).permute(0, 3, 1, 2)

    def project(self, condition: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Build the scale-specific observation feature once per input video."""
        aligned = self.pool_condition(condition, target_frames)
        return self.condition_projection(aligned)

    def fuse(self, state: torch.Tensor, aligned: torch.Tensor) -> torch.Tensor:
        """Adapt a cached condition to the current reverse-step state."""
        if state.shape != aligned.shape:
            raise ValueError("CCB state and projected condition must share shape")
        gate = torch.sigmoid(
            self.state_gate(state) + self.condition_gate(aligned) + self.gate_bias
        )
        return gate * state + (1.0 - gate) * aligned

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if state.ndim != 4 or condition.ndim != 4:
            raise ValueError("CCB expects [batch, frames, joints, channels] tensors")
        return self.fuse(state, self.project(condition, state.shape[1]))
