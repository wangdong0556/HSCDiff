from pathlib import Path

import numpy as np

from hscd.data import PoseSequenceDataset


def test_pose_archive_window_and_root(tmp_path: Path) -> None:
    pose_2d = np.zeros((2, 11, 17, 2), dtype=np.float32)
    pose_3d = np.ones((2, 11, 17, 3), dtype=np.float32)
    pose_3d[:, :, 1:, 0] += 2.0
    path = tmp_path / "poses.npz"
    np.savez(path, poses_2d=pose_2d, poses_3d=pose_3d)
    dataset = PoseSequenceDataset(path, window=9, train=False, root_relative=True)
    sample = dataset[0]
    assert sample["pose_2d"].shape == (9, 17, 2)
    assert sample["pose_3d"].shape == (9, 17, 3)
    assert np.allclose(sample["pose_3d"][:, 0].numpy(), 0.0)

