"""Golden tests pinning the legacy pipeline to the released v9 / v2 results.

Paths are repo-relative; run pytest from the repository root.
"""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from p2igan_bench.models import build_generator

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "p2igan_bench" / "config"

CHECKPOINTS = {
    "p2igan_gan_baseline.json": "weights/p2igan_gan_v9/latest.pt",
    "p2igan_ganwopos_baseline.json": "weights/p2igan_gan_wopos_v9/latest.pt",
    "dk.json": "weights/dkv2/latest.pt",
    "stdk.json": "weights/stdkv2/latest.pt",
}


def _require(*paths: str) -> None:
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        pytest.skip(f"missing local data: {missing}")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def in_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.mark.parametrize("config_name", sorted(CHECKPOINTS))
def test_checkpoint_loads_strictly(config_name, in_root):
    ckpt_path = CHECKPOINTS[config_name]
    _require(ckpt_path)
    cfg = json.loads((CONFIG_DIR / config_name).read_text())
    generator = build_generator(cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    generator.load_state_dict(state["generator"], strict=True)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="golden outputs were produced on CUDA")
def test_v9_inference_matches_stored_output(tmp_path, in_root):
    ckpt = CHECKPOINTS["p2igan_gan_baseline.json"]
    reference = "datasets/infer/p2igan_gan_baseline_v3_pm_gauge.zarr"
    _require(ckpt, reference, "datasets/midas/midas_test.zarr", "datasets/masks/gauge_mask_128_train.txt")

    infer = _load_script("infer")
    out = tmp_path / "v9.zarr"
    infer.main(
        Namespace(
            config=CONFIG_DIR / "p2igan_gan_baseline.json",
            checkpoint=Path(ckpt),
            model_dir=None,
            data_root=None,
            output=out,
            passes=1,
            device="cuda:0",
            log_every=50,
            stride=16,
            overlap=12,
            output_scale=255.0,
            overwrite=True,
            log_level="WARNING",
        )
    )

    produced = zarr.open(str(out), mode="r")
    expected = zarr.open(reference, mode="r")
    assert sorted(produced.array_keys()) == sorted(expected.array_keys())
    for key in expected.array_keys():
        diff = np.abs(produced[key][:] - expected[key][:]).max()
        assert diff == 0.0, f"{key}: max abs diff {diff}"
