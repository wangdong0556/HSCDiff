from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hscd.config import build_diffusion, build_model, load_config
from hscd.data import PoseSequenceDataset
from hscd.metrics import mpjpe, p_mpjpe, pck, pck_auc
from train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an HSCD checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None, help="Override the validation NPZ")
    parser.add_argument("--device", default=None)
    parser.add_argument("--reverse-steps", type=int, default=None)
    parser.add_argument("--save-predictions", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config["training"].get("device", "auto"))
    model = build_model(config).to(device)
    diffusion = build_diffusion(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    data_config = config["data"]
    dataset = PoseSequenceDataset(
        args.data or data_config["valid_file"],
        data_config["input_frames"],
        train=False,
        root_relative=data_config.get("root_relative", True),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_config.get("num_workers", 0),
    )
    predictions = []
    targets = []
    for batch in tqdm(loader, desc="HSCD reverse sampling"):
        pose_2d = batch["pose_2d"].to(device)
        prediction = diffusion.sample(model, pose_2d, reverse_steps=args.reverse_steps)
        predictions.append(prediction.cpu())
        targets.append(batch["pose_3d"])

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    metric_scale = float(data_config.get("metric_scale", 1.0))
    metrics = {name.upper() for name in data_config.get("metrics", ["MPJPE", "P-MPJPE"])}
    if "MPJPE" in metrics:
        print(f"MPJPE: {mpjpe(prediction, target, metric_scale).item():.4f} mm")
    if "P-MPJPE" in metrics:
        print(f"P-MPJPE: {p_mpjpe(prediction, target, metric_scale).item():.4f} mm")
    if "PCK" in metrics:
        threshold = float(data_config.get("pck_threshold_mm", 150.0))
        print(f"PCK@{threshold:g}mm: {pck(prediction, target, threshold, metric_scale).item():.4f}%")
    if "AUC" in metrics:
        maximum = float(data_config.get("auc_max_threshold_mm", 150.0))
        print(f"AUC(0-{maximum:g}mm): {pck_auc(prediction, target, maximum, metric_scale).item():.4f}%")
    if args.save_predictions:
        output = Path(args.save_predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, predictions_3d=prediction.numpy(), targets_3d=target.numpy())
        print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
