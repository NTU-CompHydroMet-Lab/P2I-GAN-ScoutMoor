"""Golden tests pinning the legacy pipeline to the released v9 / v2 results.

Paths are repo-relative; run pytest from the repository root.

The references in ``datasets/golden/`` were produced by tag ``baseline-v0`` on the
GPU, driver and torch version recorded in their attrs, with every model fed the
MIDAS gauge test set. Inference is only bit-exact on that same setup; on CPU the
P2I models are not comparable at all, because ``InputBlock`` runs its IDW in
float16 on CUDA and float32 on CPU.
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


GAUGE_TEST_CONFIG = "p2igan_gan_baseline.json"

# config, golden reference, author's stored output, (max, mean) allowed vs. author
GOLDEN = {
    "v9": ("p2igan_gan_baseline.json", "v9_midas_gauge", "p2igan_gan_baseline_v3_pm_gauge", (2.5, 0.02)),
    "wopos_v9": ("p2igan_ganwopos_baseline.json", "wopos_v9_midas_gauge", None, None),
    "dk": ("dk.json", "dk_midas_gauge", "dk_v3_pm_gauge", (0.0, 0.0)),
    "stdk": ("stdk.json", "stdk_midas_gauge", "stdk_v3_pm_gauge", (0.0, 0.0)),
}

# Largest difference to the golden references accepted from this environment.
INFERENCE_ATOL = 0.0


def _golden(name: str) -> str:
    return f"datasets/golden/{GOLDEN[name][1]}.zarr"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="golden outputs were produced on CUDA")
@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_inference_matches_golden(name, tmp_path, in_root):
    config_name = GOLDEN[name][0]
    ckpt = CHECKPOINTS[config_name]
    _require(ckpt, _golden(name), "datasets/midas/midas_test.zarr", "datasets/masks/gauge_mask_128_train.txt")

    cfg = json.loads((CONFIG_DIR / config_name).read_text())
    cfg["data"]["test"] = json.loads((CONFIG_DIR / GAUGE_TEST_CONFIG).read_text())["data"]["test"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))

    infer = _load_script("infer")
    out = tmp_path / "out.zarr"
    infer.main(
        Namespace(
            config=config_path,
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
    expected = zarr.open(_golden(name), mode="r")
    assert sorted(produced.array_keys()) == sorted(expected.array_keys())
    for key in expected.array_keys():
        diff = np.abs(produced[key][:] - expected[key][:]).max()
        assert diff <= INFERENCE_ATOL, f"{key}: max abs diff {diff}"


@pytest.mark.parametrize("name", sorted(n for n in GOLDEN if GOLDEN[n][2]))
def test_golden_close_to_author_output(name, in_root):
    author_path = f"datasets/infer/{GOLDEN[name][2]}.zarr"
    max_tol, mean_tol = GOLDEN[name][3]
    _require(_golden(name), author_path)
    golden = zarr.open(_golden(name), mode="r")
    author = zarr.open(author_path, mode="r")
    assert sorted(golden.array_keys()) == sorted(author.array_keys())
    for key in author.array_keys():
        diff = np.abs(golden[key][:] - author[key][:])
        assert diff.max() <= max_tol and diff.mean() <= mean_tol, f"{key}: max {diff.max()}, mean {diff.mean()}"
