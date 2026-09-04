import torch

from hscd.metrics import mpjpe, p_mpjpe, pck, pck_auc


def test_metrics_are_zero_for_identical_poses() -> None:
    pose = torch.randn(2, 5, 17, 3)
    assert torch.allclose(mpjpe(pose, pose), torch.tensor(0.0))
    assert p_mpjpe(pose, pose).item() < 1e-5


def test_procrustes_removes_similarity_transform() -> None:
    target = torch.randn(2, 5, 17, 3)
    angle = torch.tensor(0.4)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0],
         [torch.sin(angle), torch.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    predicted = 1.7 * (target @ rotation) + torch.tensor([0.4, -0.2, 0.7]) ####
    assert p_mpjpe(predicted, target).item() < 1e-4


def test_pck_and_auc_bounds() -> None:
    target = torch.zeros(1, 1, 2, 3)
    predicted = target.clone()
    assert pck(predicted, target).item() == 100.0
    value = pck_auc(predicted, target).item()
    assert 95.0 <= value <= 100.0
