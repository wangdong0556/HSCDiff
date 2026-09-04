# you can choose different version

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration and validate the method-defining fields."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")

    required_sections = ("experiment", "data", "model", "diffusion", "training")
    missing = [key for key in required_sections if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

    data = config["data"]
    model = config["model"]
    diffusion = config["diffusion"]
    training = config["training"]

    if data["num_joints"] <= 0 or data["input_frames"] <= 0:
        raise ValueError("num_joints and input_frames must be positive")
    if model["feature_dim"] % model["attention_heads"] != 0:
        raise ValueError("feature_dim must be divisible by attention_heads")
    configured_head_dim = model.get("attention_head_dim")
    if configured_head_dim is not None:
        actual_head_dim = model["feature_dim"] // model["attention_heads"]
        if configured_head_dim != actual_head_dim:
            raise ValueError(
                f"attention_head_dim={configured_head_dim} but feature_dim/heads={actual_head_dim}"
            )
    channels = model["decoder_channels"]
    if len(channels) != model["temporal_scales"]:
        raise ValueError("decoder_channels must contain one width per temporal scale")
    if diffusion["beta_schedule"] != "linear":
        raise ValueError("This HSCD implementation currently supports the paper's linear beta schedule")
    if not 0.0 < diffusion["beta_start"] < diffusion["beta_end"] < 1.0:
        raise ValueError("Expected 0 < beta_start < beta_end < 1")
    if not 1 <= diffusion["reverse_steps"] <= diffusion["training_noise_levels"]:
        raise ValueError("reverse_steps must be in [1, training_noise_levels]")
    if training["optimizer"].lower() != "adamw":
        raise ValueError("The manuscript configuration uses AdamW")
    return config


def build_model(config: Dict[str, Any]):
    """Construct HSCD without introducing a circular import at module import time."""
    from .models import HSCD

    data = config["data"]
    model = config["model"]
    return HSCD(
        num_joints=data["num_joints"],
        input_dim=data.get("input_dim", 2),
        output_dim=data.get("output_dim", 3),
        feature_dim=model["feature_dim"],
        mie_layers=model["mie_layers"],
        attention_heads=model["attention_heads"],
        ffn_ratio=model.get("ffn_ratio", 2.0),
        dropout=model.get("dropout", 0.0),
        decoder_channels=tuple(model["decoder_channels"]),
        decoder_blocks_per_scale=model.get("decoder_blocks_per_scale", 1),
    )


def build_diffusion(config: Dict[str, Any]):
    from .diffusion import GaussianPoseDiffusion

    diffusion = config["diffusion"]
    return GaussianPoseDiffusion(
        timesteps=diffusion["training_noise_levels"],
        beta_start=diffusion["beta_start"],
        beta_end=diffusion["beta_end"],
        reverse_steps=diffusion["reverse_steps"],
        clip_x0=diffusion.get("clip_x0"),
    )


def with_overrides(config: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """Return a deep-copied configuration with dotted-key overrides."""
    result = deepcopy(config)
    for dotted_key, value in overrides.items():
        target = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return result
