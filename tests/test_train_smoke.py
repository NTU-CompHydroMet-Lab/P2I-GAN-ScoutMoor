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
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    cfg = json.loads((CONFIG_DIR / config_name).read_text())
    cfg["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg["save_dir"] = str(tmp_path / "weights")
    cfg["experiment_name"] = "smoke"
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
    assert (second.global_step, second.start_epoch, second.mlflow_run_id) == (2, 2, first.mlflow_run_id)
    for name, value in second.generator.state_dict().items():
        assert torch.equal(value.cpu(), saved["generator"][name]), name
    second.train()
    assert second.global_step == 4
