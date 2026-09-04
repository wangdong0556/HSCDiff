from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _extract(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (reference.ndim - 1)))


class GaussianPoseDiffusion(nn.Module):
    """Standard noise-prediction diffusion with respaced ancestral sampling."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        reverse_steps: int = 50, # 10 25 or 100
        clip_x0: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not 1 <= reverse_steps <= timesteps:
            raise ValueError("reverse_steps must be in [1, timesteps]")
        self.timesteps = int(timesteps)
        self.reverse_steps = int(reverse_steps)
        self.clip_x0 = clip_x0

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    def q_sample(
        self,
        clean_pose: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(clean_pose) if noise is None else noise
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, clean_pose) * clean_pose
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_pose) * noise
        )

    def predict_x0(
        self,
        noisy_pose: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = _extract(self.alphas_cumprod, timesteps, noisy_pose)
        clean = (noisy_pose - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        if self.clip_x0 is not None:
            clean = clean.clamp(-self.clip_x0, self.clip_x0)
        return clean

    def training_loss(
        self,
        model: nn.Module,
        pose_2d: torch.Tensor,
        clean_pose_3d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch = clean_pose_3d.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=clean_pose_3d.device)
        noise = torch.randn_like(clean_pose_3d)
        noisy_pose = self.q_sample(clean_pose_3d, timesteps, noise)
        condition = model.encode(pose_2d)
        predicted_noise = model.denoise(noisy_pose, timesteps, condition)
        loss = F.mse_loss(predicted_noise, noise)
        return {
            "loss": loss,
            "predicted_noise": predicted_noise,
            "target_noise": noise,
            "timesteps": timesteps,
        }

    def sampling_schedule(self, reverse_steps: Optional[int] = None) -> torch.Tensor:
        steps = self.reverse_steps if reverse_steps is None else int(reverse_steps)
        if not 1 <= steps <= self.timesteps:
            raise ValueError("reverse_steps must be in [1, timesteps]")
        schedule = torch.linspace(self.timesteps - 1, 0, steps=steps, device=self.betas.device)
        return schedule.round().long()

    def _respaced_posterior(
        self,
        noisy_pose: torch.Tensor,
        clean_prediction: torch.Tensor,
        timestep: int,
        previous_timestep: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_bar_t = self.alphas_cumprod[timestep]
        alpha_bar_previous = (
            torch.ones((), device=noisy_pose.device, dtype=noisy_pose.dtype)
            if previous_timestep < 0
            else self.alphas_cumprod[previous_timestep]
        )
        transition_alpha = alpha_bar_t / alpha_bar_previous
        transition_beta = 1.0 - transition_alpha
        denominator = (1.0 - alpha_bar_t).clamp_min(1e-12)
        variance = transition_beta * (1.0 - alpha_bar_previous) / denominator
        coefficient_x0 = alpha_bar_previous.sqrt() * transition_beta / denominator
        coefficient_xt = transition_alpha.sqrt() * (1.0 - alpha_bar_previous) / denominator
        mean = coefficient_x0 * clean_prediction + coefficient_xt * noisy_pose
        return mean, variance.clamp_min(0.0)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        pose_2d: torch.Tensor,
        reverse_steps: Optional[int] = None,
        return_trajectory: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """Run the Algorithm-1-style reverse process with one cached MIE pass."""
        condition = model.encode(pose_2d)
        batch, frames, joints, _ = pose_2d.shape
        shape = (batch, frames, joints, model.output_dim)
        state = torch.randn(shape, device=pose_2d.device, dtype=pose_2d.dtype, generator=generator)
        schedule = self.sampling_schedule(reverse_steps)
        trajectory = []

        for index, timestep_tensor in enumerate(schedule):
            timestep = int(timestep_tensor.item())
            previous_timestep = int(schedule[index + 1].item()) if index + 1 < len(schedule) else -1
            timestep_batch = torch.full(
                (batch,), timestep, device=pose_2d.device, dtype=torch.long
            )
            predicted_noise = model.denoise(state, timestep_batch, condition)
            clean_prediction = self.predict_x0(state, timestep_batch, predicted_noise)
            clean_prediction = clean_prediction - clean_prediction[:, :, :1, :]

            if previous_timestep < 0:
                state = clean_prediction
            else:
                mean, variance = self._respaced_posterior(
                    state, clean_prediction, timestep, previous_timestep
                )
                noise = torch.randn(
                    state.shape,
                    device=state.device,
                    dtype=state.dtype,
                    generator=generator,
                )
                state = mean + variance.sqrt() * noise
            if return_trajectory:
                trajectory.append(state.clone())

        if return_trajectory:
            return state, torch.stack(trajectory, dim=1)
        return state

