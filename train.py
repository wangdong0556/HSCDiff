#You can train 

##The PRD implementation is currently being organized and will be released soon.

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hscd.config import build_diffusion, build_model, load_config
from hscd.data import PoseSequenceDataset
from hscd.metrics import mpjpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HSCD")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume")
    parser.add_argument("--device", default=None, help="Override cpu/cuda/auto")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--dry-run", action="store_true", help="Run one synthetic train/sample pass")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loss(model, diffusion, loader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        pose_2d = batch["pose_2d"].to(device)
        pose_3d = batch["pose_3d"].to(device)
        result = diffusion.training_loss(model, pose_2d, pose_3d)
        total += float(result["loss"].item()) * pose_2d.shape[0]
        count += pose_2d.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def evaluate_pose(model, diffusion, loader, device: torch.device, metric_scale: float) -> float:
    model.eval()
    weighted_error = 0.0
    count = 0
    for batch in tqdm(loader, desc="reverse evaluation", leave=False):
        pose_2d = batch["pose_2d"].to(device)
        target = batch["pose_3d"].to(device)
        prediction = diffusion.sample(model, pose_2d)
        error = mpjpe(prediction, target, scale=metric_scale)
        weighted_error += float(error.item()) * pose_2d.shape[0]
        count += pose_2d.shape[0]
    return weighted_error / max(count, 1)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_mpjpe: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_mpjpe": best_mpjpe,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["experiment"]["seed"]))
    requested_device = args.device or config["training"].get("device", "auto")
    device = resolve_device(requested_device)
    model = build_model(config).to(device)
    diffusion = build_diffusion(config).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} parameters={parameter_count:,}")

    if args.dry_run:
        batch = 2
        frames = min(int(config["data"]["input_frames"]), 27)
        joints = int(config["data"]["num_joints"])
        pose_2d = torch.randn(batch, frames, joints, 2, device=device)
        pose_3d = torch.randn(batch, frames, joints, 3, device=device)
        pose_3d = pose_3d - pose_3d[:, :, :1, :]
        result = diffusion.training_loss(model, pose_2d, pose_3d)
        prediction = diffusion.sample(model, pose_2d, reverse_steps=min(2, diffusion.timesteps))
        print(f"dry_run_loss={result['loss'].item():.6f} output_shape={tuple(prediction.shape)}")
        return

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(output_dir / "train.log", encoding="utf-8"), logging.StreamHandler()],
    )
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    data_config = config["data"]
    train_dataset = PoseSequenceDataset(
        data_config["train_file"], data_config["input_frames"], train=True,
        root_relative=data_config.get("root_relative", True),
    )
    valid_dataset = PoseSequenceDataset(
        data_config["valid_file"], data_config["input_frames"], train=False,
        root_relative=data_config.get("root_relative", True),
    )
    workers = int(data_config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay") or 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(config["training"]["lr_decay"])
    )
    start_epoch = 0
    best_mpjpe = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mpjpe = float(checkpoint.get("best_mpjpe", best_mpjpe))

    epochs = args.epochs or int(config["training"]["epochs"])
    eval_every = int(config["training"].get("eval_every", 1))
    save_every = int(config["training"].get("save_every", eval_every))
    metric_scale = float(data_config.get("metric_scale", 1.0))
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch in progress:
            pose_2d = batch["pose_2d"].to(device, non_blocking=True)
            pose_3d = batch["pose_3d"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            result = diffusion.training_loss(model, pose_2d, pose_3d)
            result["loss"].backward()
            optimizer.step()
            total_loss += float(result["loss"].item()) * pose_2d.shape[0]
            total_examples += pose_2d.shape[0]
            progress.set_postfix(loss=f"{result['loss'].item():.5f}")
        scheduler.step()
        train_loss = total_loss / max(total_examples, 1)
        valid_loss = evaluate_loss(model, diffusion, valid_loader, device)
        logging.info(
            "epoch=%d train_noise_mse=%.7f valid_noise_mse=%.7f lr=%.8g",
            epoch + 1, train_loss, valid_loss, scheduler.get_last_lr()[0],
        )

        if (epoch + 1) % eval_every == 0:
            pose_error = evaluate_pose(model, diffusion, valid_loader, device, metric_scale)
            logging.info("epoch=%d Protocol#1_MPJPE=%.4fmm", epoch + 1, pose_error)
            if pose_error < best_mpjpe:
                best_mpjpe = pose_error
                save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, best_mpjpe)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pt", model, optimizer, scheduler, epoch, best_mpjpe)
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, best_mpjpe)


if __name__ == "__main__":
    main()
