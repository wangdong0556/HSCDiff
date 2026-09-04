# PoseSequenceDataset
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class PoseSequenceDataset(Dataset):
    """Windowed 2D/3D pose pairs stored in a compact NPZ archive.

    Required arrays are ``poses_2d`` with shape [N, T, J, 2] and ``poses_3d``
    with shape [N, T, J, 3]. The loader never guesses coordinate units or
    camera normalization.
    """

    def __init__(
        self,
        path: str | Path,
        window: int,
        train: bool,
        root_relative: bool = True,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.window = int(window)
        self.train = bool(train)
        archive = np.load(self.path, allow_pickle=False)
        if "poses_2d" not in archive or "poses_3d" not in archive:
            raise KeyError(f"{self.path} must contain poses_2d and poses_3d")
        self.poses_2d = np.asarray(archive["poses_2d"], dtype=np.float32)
        self.poses_3d = np.asarray(archive["poses_3d"], dtype=np.float32)
        self._validate(root_relative)

    def _validate(self, root_relative: bool) -> None:
        if self.poses_2d.ndim != 4 or self.poses_3d.ndim != 4:
            raise ValueError("Pose arrays must have shape [N, T, J, C]")
        if self.poses_2d.shape[:3] != self.poses_3d.shape[:3]:
            raise ValueError("2D and 3D arrays must align over N, T, and J")
        if self.poses_2d.shape[-1] != 2 or self.poses_3d.shape[-1] != 3:
            raise ValueError("Expected 2D inputs and 3D targets")
        if self.poses_2d.shape[1] < self.window:
            raise ValueError(
                f"Sequences contain {self.poses_2d.shape[1]} frames, shorter than window={self.window}"
            )
        if not np.isfinite(self.poses_2d).all() or not np.isfinite(self.poses_3d).all():
            raise ValueError("Pose arrays contain NaN or infinity")
        if root_relative:
            self.poses_3d = self.poses_3d - self.poses_3d[:, :, :1, :]

    def __len__(self) -> int:
        return self.poses_2d.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        total = self.poses_2d.shape[1]
        if total == self.window:
            start = 0
        elif self.train:
            start = int(np.random.randint(0, total - self.window + 1))
        else:
            start = (total - self.window) // 2
        stop = start + self.window
        return {
            "pose_2d": torch.from_numpy(self.poses_2d[index, start:stop].copy()),
            "pose_3d": torch.from_numpy(self.poses_3d[index, start:stop].copy()),
        }


def save_pose_archive(path: str | Path, poses_2d: np.ndarray, poses_3d: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        poses_2d=np.asarray(poses_2d, dtype=np.float32),
        poses_3d=np.asarray(poses_3d, dtype=np.float32),
    )

