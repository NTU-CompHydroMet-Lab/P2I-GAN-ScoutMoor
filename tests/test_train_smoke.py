"""Two-step training runs on a handful of HDF5 events, one per shipped config."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from p2igan_bench.models import build_generator

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "p2igan_bench" / "config"
H5_ROOT = ROOT / "datasets" / "nimrod_h5"
CONFIGS = ["p2igan_gan_baseline.json", "p2igan_ganwopos_baseline.json", "dk.json", "stdk.json"]


def _link_events(src: Path, dst: Path, n: int) -> Path:
    dst.mkdir()
    for f in sorted(src.glob("*.h5"))[:n]:
        (dst / f.name).symlink_to(f)
    return dst


def _smoke_config(config_name, tmp_path, monkeypatch):
    if not (H5_ROOT / "train").is_dir() or not (H5_ROOT / "test").is_dir():
        pytest.skip("missing datasets/nimrod_h5")
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")

    cfg = json.loads((CONFIG_DIR / config_name).read_text())
    cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg["save_dir"] = str(tmp_path / "weights")
    cfg["experiment_name"] = "smoke"
    cfg["log_to_wandb"] = True
    cfg["data"]["train"]["data_root"] = str(_link_events(H5_ROOT / "train", tmp_path / "train", 2))
    cfg["data"]["valid"]["data_root"] = str(_link_events(H5_ROOT / "test", tmp_path / "valid", 1))
    cfg["train"].update(batch_size=1, num_workers=0, persistent_workers=False, iterations=2)
    cfg["train"].pop("prefetch_factor", None)
    return cfg


def _train_module():
    spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    return train


@pytest.mark.slow
@pytest.mark.parametrize("config_name", CONFIGS)
def test_two_training_steps(config_name, tmp_path, monkeypatch):
    cfg = _smoke_config(config_name, tmp_path, monkeypatch)
    trainer = _train_module().Trainer(cfg)
    trainer.train()

    assert trainer.global_step == 2
    assert math.isfinite(trainer.last_rec_loss)
    state = torch.load(tmp_path / "weights" / "latest.pt", map_location="cpu", weights_only=True)
    build_generator(cfg).load_state_dict(state["generator"], strict=True)


@pytest.mark.slow
def test_resume_continues_from_latest(tmp_path, monkeypatch):
    cfg = _smoke_config("p2igan_gan_baseline.json", tmp_path, monkeypatch)
    train = _train_module()
    first = train.Trainer(cfg)
    first.train()
    saved = torch.load(tmp_path / "weights" / "latest.pt", map_location="cpu", weights_only=True)

    cfg["train"]["iterations"] = 4
    second = train.Trainer(cfg)
    assert first.wandb_run_id is not None
    assert (second.global_step, second.start_epoch, second.wandb_run_id) == (2, 2, first.wandb_run_id)
    for name, value in second.generator.state_dict().items():
        assert torch.equal(value.cpu(), saved["generator"][name]), name
    second.train()
    assert second.global_step == 4


@pytest.mark.slow
def test_step_based_checkpoints_and_lr_schedule(tmp_path, monkeypatch):
    cfg = _smoke_config("p2igan_gan_baseline.json", tmp_path, monkeypatch)
    cfg["data"]["train"]["data_root"] = str(_link_events(H5_ROOT / "train", tmp_path / "train3", 3))
    cfg["train"].update(iterations=3, val_every_steps=2)
    cfg["train"]["optimizer"].update(lr_schedule="cosine", lr_min=1e-5, warmup_steps=1)
    trainer = _train_module().Trainer(cfg)
    assert trainer._lr_factor(0) == 1.0  # end of the one-step warmup
    assert trainer._lr_factor(3) == pytest.approx(1e-5 / cfg["train"]["optimizer"]["lr"])
    trainer.train()
    state = torch.load(tmp_path / "weights" / "latest.pt", map_location="cpu", weights_only=True)
    assert trainer.global_step == 3 and state["global_step"] == 3  # final checkpoint off the 2-step grid
    assert (tmp_path / "weights" / "best.pt").exists()
    assert sorted(p.name for p in (tmp_path / "weights" / "artifacts").glob("val_*")) == [
        "val_step2_batch0_ex0.png",
        "val_step3_batch0_ex0.png",
    ]
