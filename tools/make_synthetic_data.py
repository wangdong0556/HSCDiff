from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small HSCD pipeline-check dataset")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--frames", type=int, default=27)
    parser.add_argument("--train-sequences", type=int, default=8)
    parser.add_argument("--valid-sequences", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_sequences(rng: np.random.Generator, count: int, frames: int) -> tuple[np.ndarray, np.ndarray]:
    joints = 17
    base = rng.normal(0.0, 0.15, size=(count, 1, joints, 3)).astype(np.float32)
    motion = rng.normal(0.0, 0.008, size=(count, frames, joints, 3)).astype(np.float32)
    pose_3d = base + np.cumsum(motion, axis=1)
    pose_3d -= pose_3d[:, :, :1, :]
    depth = pose_3d[..., 2:3] + 3.0
    pose_2d = pose_3d[..., :2] / depth
    pose_2d += rng.normal(0.0, 0.002, size=pose_2d.shape).astype(np.float32)
    return pose_2d.astype(np.float32), pose_3d.astype(np.float32)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for split, count in (("train", args.train_sequences), ("valid", args.valid_sequences)):
        pose_2d, pose_3d = make_sequences(rng, count, args.frames)
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, poses_2d=pose_2d, poses_3d=pose_3d)
        print(f"saved {path}: 2D={pose_2d.shape}, 3D={pose_3d.shape}")


if __name__ == "__main__":
    main()

