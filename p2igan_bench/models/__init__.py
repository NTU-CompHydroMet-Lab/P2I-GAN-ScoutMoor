"""Model registry for the benchmark."""

from typing import Any, Dict

import torch.nn as nn

from .dk import DKGenerator
from .p2igan import P2IDiscriminator, P2IGenerator
from .p2iganwopos import P2IDiscriminator as P2IDiscriminatorWoPos
from .p2iganwopos import P2IGenerator as P2IGeneratorWoPos
from .stdk import STDKGenerator


def _model_name(cfg: Dict[str, Any]) -> str:
    try:
        return cfg["model"]["name"].lower()
    except KeyError as exc:
        raise KeyError("Config requires model.name") from exc


def _sample_length(cfg: Dict[str, Any]) -> int:
    return cfg.get("data", {}).get("train", {}).get("sample_length", 16)


def build_generator(cfg: Dict[str, Any]) -> nn.Module:
    model_name = _model_name(cfg)
    if model_name == "p2igan":
        return P2IGenerator(cfg)
    if model_name == "p2iganwopos":
        return P2IGeneratorWoPos(cfg)
    if model_name == "dk":
        return DKGenerator(cfg, length=_sample_length(cfg))
    if model_name == "stdk":
        return STDKGenerator(cfg, length=_sample_length(cfg))
    raise ValueError(f"Unknown model.name: {model_name}")


def build_discriminator(cfg: Dict[str, Any]) -> nn.Module:
    model_name = _model_name(cfg)
    seq_channels = cfg["model"].get("in_channels", 1) * _sample_length(cfg)
    if model_name == "p2igan":
        return P2IDiscriminator(in_channels=seq_channels)
    if model_name == "p2iganwopos":
        return P2IDiscriminatorWoPos(in_channels=seq_channels)
    raise ValueError(f"No discriminator for model.name: {model_name}")


__all__ = [
    "build_generator",
    "build_discriminator",
    "P2IGenerator",
    "P2IDiscriminator",
    "P2IGeneratorWoPos",
    "P2IDiscriminatorWoPos",
    "DKGenerator",
    "STDKGenerator",
]
