"""Typed schema for the JSON training and inference configs.

Defaults here match the fallbacks in the code that consumes the configs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Tag


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    name: Literal["p2igan", "p2iganwopos", "dk", "stdk"]
    in_channels: int = 1
    out_channels: int = 1


class MaskConfig(_Strict):
    type: Literal["stis"] = "stis"
    file: str


class EventSplit(_Strict):
    """Legacy events: a directory of HDF5 files or a Zarr group, masked by a fixed gauge grid."""

    kind: Literal["events"] = "events"
    data_root: str
    w: int | None = None
    h: int | None = None
    sample_length: int | None = None
    mask: MaskConfig | None = None
    shuffle: bool = False


class WindowSelection(_Strict):
    """Keep a window if any frame has ``wet_fraction`` of pixels at or above ``wet_raw`` and one pixel at or
    above ``peak_raw``. Thresholds are in the store's raw units (mm/h * 32)."""

    wet_raw: int = 2
    wet_fraction: float = 0.3
    peak_raw: int = 65


class DomainSplit(_Strict):
    """Windows cut from the cropped NIMROD domain store, input taken at fixed gauge pixels."""

    kind: Literal["domain"]
    store: str
    start: str
    end: str
    gauges: str
    h: int
    w: int
    sample_length: int = 16
    stride: int = 1
    selection: WindowSelection = WindowSelection()
    shuffle: bool = False


def _split_kind(value: Any) -> str:
    if isinstance(value, dict):
        return value.get("kind", "events")
    return getattr(value, "kind", "events")


Split = Annotated[Annotated[EventSplit, Tag("events")] | Annotated[DomainSplit, Tag("domain")], Discriminator(_split_kind)]


class DataConfig(_Strict):
    train: Split
    valid: Split | None = None
    test: Split | None = None


class LossConfig(_Strict):
    k1_weight: float = 0.0
    use_pm: bool = False
    pm_weight: float = 10.0
    use_gan: bool = False
    gan_loss: Literal["hinge", "nsgan", "lsgan"] = "hinge"
    adversarial_weight: float = 0.01
    target_real_label: float = 1.0
    target_fake_label: float = 0.0


class OptimizerConfig(_Strict):
    type: Literal["Adam"] = "Adam"
    lr: float
    beta1: float = 0.0
    beta2: float = 0.99


class TrainConfig(_Strict):
    optimizer: OptimizerConfig
    batch_size: int
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    log_step: int = 100
    iterations: int | None = None
    max_epochs: int | None = None
    niter: int = 1
    use_validation: bool = True
    cudnn_benchmark: bool = False
    amp: Literal["off", "bf16"] = "off"
    compile: bool = False


class VizConfig(_Strict):
    scale: Literal["gt_pred", "gt", "fixed"] = "gt_pred"
    vmin: float | None = None
    vmax: float | None = None


class Config(_Strict):
    seed: int = 42
    device: str | None = None
    save_dir: str = "weights"
    experiment_name: str | None = None
    run_name: str | None = None
    description: str = ""
    model: ModelConfig
    data: DataConfig
    loss: LossConfig = LossConfig()
    train: TrainConfig
    viz: VizConfig = VizConfig()


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check ``cfg`` against the schema and return it with only the keys it set."""
    return Config.model_validate(cfg).model_dump(exclude_unset=True)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    return validate_config(raw)
