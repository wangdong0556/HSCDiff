import torch

from hscd.diffusion import GaussianPoseDiffusion
from hscd.models import HSCD


def tiny_model() -> HSCD:
    return HSCD(
        num_joints=17,
        feature_dim=32,
        mie_layers=2,
        attention_heads=4,
        ffn_ratio=2.0,
        decoder_channels=(32, 48),
    )


def test_hscd_noise_prediction_shape() -> None:
    model = tiny_model()
    pose_2d = torch.randn(2, 9, 17, 2)
    noisy_pose = torch.randn(2, 9, 17, 3)
    timesteps = torch.tensor([1, 7])
    condition = model.encode(pose_2d)
    prediction = model.denoise(noisy_pose, timesteps, condition)
    assert condition.full.shape == (2, 9, 17, 32)
    assert len(condition.scales) == 2
    assert condition.scales[0].shape == (2, 9, 17, 32)
    assert condition.scales[1].shape == (2, 5, 17, 48)
    assert prediction.shape == noisy_pose.shape


def test_training_loss_and_respaced_sample() -> None:
    model = tiny_model()
    diffusion = GaussianPoseDiffusion(timesteps=20, reverse_steps=4)
    pose_2d = torch.randn(2, 9, 17, 2)
    pose_3d = torch.randn(2, 9, 17, 3)
    pose_3d = pose_3d - pose_3d[:, :, :1]
    result = diffusion.training_loss(model, pose_2d, pose_3d)
    assert result["loss"].ndim == 0
    prediction, trajectory = diffusion.sample(model, pose_2d, return_trajectory=True)
    assert prediction.shape == pose_3d.shape
    assert trajectory.shape == (2, 4, 9, 17, 3)
    assert torch.allclose(prediction[:, :, 0], torch.zeros_like(prediction[:, :, 0]), atol=1e-5)


def test_mie_is_evaluated_once_during_sampling() -> None:
    model = tiny_model()
    diffusion = GaussianPoseDiffusion(timesteps=8, reverse_steps=3)
    pose_2d = torch.randn(1, 7, 17, 2)
    calls = 0
    original = model.mie.forward

    def counted_forward(value):
        nonlocal calls
        calls += 1
        return original(value)

    model.mie.forward = counted_forward
    diffusion.sample(model, pose_2d)
    assert calls == 1
