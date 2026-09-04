

#The PRD implementation is currently being organized and will be released soon.

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import torch
from torch import nn

from .ccb import CrossScaleConditioningBridge
from .layers import (
    TemporalConvBlock,
    TemporalCrossAttention,
    TemporalDownsample,
    TemporalUpsample,
    sinusoidal_embedding,
)


@dataclass(frozen=True)
class HierarchicalCondition:
    """Full-resolution MIE output and its cached CCB scale projections."""

    full: torch.Tensor
    scales: tuple[torch.Tensor, ...]


class ProgressiveRefinementDecoder(nn.Module):
    """Multi-scale denoiser that realizes progressive diffusion refinement."""

