from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert D3DP/VideoPose3D-style Human3.6M archives to HSCD NPZ windows"
    )
    parser.add_argument("--d3dp-root", required=True, help="Extracted D3DP-main directory")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--keypoints", default="cpn_ft_h36m_dbb")
    parser.add_argument("--frames", type=int, default=243)
    parser.add_argument("--stride", type=int, default=243)
    parser.add_argument("--max-sequences", type=int, default=None, help="Debug-only limit per split")
    return parser.parse_args()


def windows(sequence_2d: np.ndarray, sequence_3d: np.ndarray, frames: int, stride: int):
    length = min(len(sequence_2d), len(sequence_3d))
    for start in range(0, length - frames + 1, stride):
        stop = start + frames
        yield sequence_2d[start:stop], sequence_3d[start:stop]


def main() -> None:
    args = parse_args()
    root = Path(args.d3dp_root).resolve()
    data_dir = root / "data"
    if not (root / "common" / "h36m_dataset.py").exists():
        raise FileNotFoundError("--d3dp-root must point to an extracted D3DP-main directory")
    sys.path.insert(0, str(root))

    # Imported from the user-supplied MIT-licensed D3DP/VideoPose3D data stack.
    from common.camera import normalize_screen_coordinates, world_to_camera
    from common.h36m_dataset import Human36mDataset

    dataset = Human36mDataset(str(data_dir / "data_3d_h36m.npz"))
    keypoint_path = data_dir / f"data_2d_h36m_{args.keypoints}.npz"
    keypoints = np.load(keypoint_path, allow_pickle=True)["positions_2d"].item()

    def collect(subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
        output_2d = []
        output_3d = []
        for subject in subjects:
            for action, animation in dataset[subject].items():
                camera_poses = []
                for camera in animation["cameras"]:
                    pose = world_to_camera(
                        animation["positions"],
                        R=camera["orientation"],
                        t=camera["translation"],
                    ).astype(np.float32)
                    pose = pose - pose[:, :1, :]
                    camera_poses.append(pose)

                for camera_index, pose_3d in enumerate(camera_poses):
                    pose_2d = np.asarray(keypoints[subject][action][camera_index], dtype=np.float32)
                    camera = dataset.cameras()[subject][camera_index]
                    pose_2d = normalize_screen_coordinates(
                        pose_2d[..., :2], w=camera["res_w"], h=camera["res_h"]
                    ).astype(np.float32)
                    for clip_2d, clip_3d in windows(
                        pose_2d, pose_3d, args.frames, args.stride
                    ):
                        output_2d.append(clip_2d)
                        output_3d.append(clip_3d)
                        if args.max_sequences and len(output_2d) >= args.max_sequences:
                            return np.stack(output_2d), np.stack(output_3d)
        if not output_2d:
            raise RuntimeError("No complete windows were produced")
        return np.stack(output_2d), np.stack(output_3d)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "h36m_train.npz": ["S1", "S5", "S6", "S7", "S8"],
        "h36m_valid.npz": ["S9", "S11"],
    }
    for filename, subjects in splits.items():
        pose_2d, pose_3d = collect(subjects)
        destination = output_dir / filename
        np.savez_compressed(destination, poses_2d=pose_2d, poses_3d=pose_3d)
        print(f"saved {destination}: 2D={pose_2d.shape}, 3D={pose_3d.shape}")


if __name__ == "__main__":
    main()

