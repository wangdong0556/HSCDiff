#Get [B, T, J, 3] for human 3D
from __future__ import annotations

import torch


def mpjpe(predicted: torch.Tensor, target: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Mean per-joint position error over arbitrary leading dimensions."""
    if predicted.shape != target.shape or predicted.shape[-1] != 3:
        raise ValueError("predicted and target must share shape [..., 3]")
    return torch.linalg.vector_norm(predicted - target, dim=-1).mean() * scale


def _flatten_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.ndim == 3:
        return pose
    if pose.ndim == 4:
        return pose.flatten(0, 1)
    raise ValueError("Expected [B, J, 3] or [B, T, J, 3]")


def p_mpjpe(predicted: torch.Tensor, target: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Protocol #2 error after a similarity Procrustes alignment."""
    predicted = _flatten_pose(predicted)
    target = _flatten_pose(target)
    if predicted.shape != target.shape or predicted.shape[-1] != 3:
        raise ValueError("predicted and target must align as [N, J, 3]")

    mu_x = target.mean(dim=1, keepdim=True)
    mu_y = predicted.mean(dim=1, keepdim=True)
    x0 = target - mu_x
    y0 = predicted - mu_y
    norm_x = torch.linalg.vector_norm(x0.flatten(1), dim=1, keepdim=True).clamp_min(1e-8)
    norm_y = torch.linalg.vector_norm(y0.flatten(1), dim=1, keepdim=True).clamp_min(1e-8)
    x0 = x0 / norm_x.unsqueeze(-1)
    y0 = y0 / norm_y.unsqueeze(-1)

    h = x0.transpose(1, 2) @ y0
    u, singular_values, vh = torch.linalg.svd(h)
    v = vh.transpose(1, 2)
    rotation = v @ u.transpose(1, 2)
    det = torch.det(rotation)
    v[:, :, -1] *= torch.where(det < 0, -1.0, 1.0).unsqueeze(1)
    singular_values[:, -1] *= torch.where(det < 0, -1.0, 1.0)
    rotation = v @ u.transpose(1, 2)

    trace = singular_values.sum(dim=1, keepdim=True)
    aligned_scale = trace * norm_x / norm_y
    translation = mu_x - aligned_scale.unsqueeze(-1) * (mu_y @ rotation)
    aligned = aligned_scale.unsqueeze(-1) * (predicted @ rotation) + translation
    return mpjpe(aligned, target, scale=scale)


def pck(
    predicted: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 150.0,
    scale: float = 1000.0,
) -> torch.Tensor:
    """Percentage of joints whose Euclidean error is below ``threshold``."""
    errors = torch.linalg.vector_norm(predicted - target, dim=-1) * scale
    return (errors < threshold).float().mean() * 100.0


def pck_auc(
    predicted: torch.Tensor,
    target: torch.Tensor,
    max_threshold: float = 150.0,
    scale: float = 1000.0,
    samples: int = 31,
) -> torch.Tensor:
    """Normalized area under the PCK curve over [0, max_threshold]."""
    errors = torch.linalg.vector_norm(predicted - target, dim=-1) * scale
    thresholds = torch.linspace(0.0, max_threshold, samples, device=errors.device)
    curve = torch.stack([(errors < threshold).float().mean() for threshold in thresholds])
    return torch.trapezoid(curve, thresholds) / max_threshold * 100.0
