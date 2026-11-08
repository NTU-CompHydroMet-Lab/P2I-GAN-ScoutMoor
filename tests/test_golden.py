"""Golden tests pinning the legacy pipeline to the released v9 / v2 results.

Paths are repo-relative; run pytest from the repository root.

The references in ``datasets/golden/`` were produced by tag ``baseline-v0`` on the
GPU, driver and torch version recorded in their attrs, with every model fed the
MIDAS gauge test set. The NIMROD radar run is compared with the author's stored
outputs directly. On other GPUs or torch versions the P2I models drift by up to
~2 on the 0-255 scale; torch 2.15 and CPU inference move them much further, see
``docs/reports/p2i-numerical-instability.md``.
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

from p2igan_bench.models import build_discriminator, build_generator

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


@pytest.mark.parametrize("config_name", ["p2igan_gan_baseline.json", "p2igan_ganwopos_baseline.json"])
def test_discriminator_loads_pre_parametrization_checkpoint(config_name, in_root):
    ckpt_path = CHECKPOINTS[config_name]
    _require(ckpt_path)
    cfg = json.loads((CONFIG_DIR / config_name).read_text())
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)["discriminator"]
    assert any(k.endswith("weight_orig") for k in state)
    build_discriminator(cfg).load_state_dict(state, strict=True)


GAUGE_TEST_CONFIG = "p2igan_gan_baseline.json"

# P2I outputs move by up to ~2 (0-255 scale) across GPUs and torch versions;
# DK and STDK have stayed bit-exact so far.
P2I_TOL = (2.5, 0.02)
EXACT = (0.0, 0.0)

# config, golden reference, author's stored output, (max, mean) allowed
GOLDEN = {
    "v9": ("p2igan_gan_baseline.json", "v9_midas_gauge", "p2igan_gan_baseline_v3_pm_gauge", P2I_TOL),
    "wopos_v9": ("p2igan_ganwopos_baseline.json", "wopos_v9_midas_gauge", "p2igan_baseline_v3_pm_gauge", P2I_TOL),
    "dk": ("dk.json", "dk_midas_gauge", "dk_v3_pm_gauge", EXACT),
    "stdk": ("stdk.json", "stdk_midas_gauge", "stdk_v3_pm_gauge", EXACT),
}


def _assert_close(produced, expected, tol) -> None:
    max_tol, mean_tol = tol
    assert sorted(produced.array_keys()) == sorted(expected.array_keys())
    for key in expected.array_keys():
        diff = np.abs(produced[key][:] - expected[key][:])
        assert diff.max() <= max_tol and diff.mean() <= mean_tol, f"{key}: max {diff.max()}, mean {diff.mean()}"


def _golden(name: str) -> str:
    return f"datasets/golden/{GOLDEN[name][1]}.zarr"


# The author's NIMROD evaluation: 20 radar events, gauges sampled from the 1024 mask.
NIMROD_TEST = {
    "data_root": "datasets/nimrod_h5/author_val",
    "w": 128,
    "h": 128,
    "mask": {"type": "stis", "file": "datasets/masks/gauge_mask_1024_train.txt"},
    "shuffle": False,
    "sample_length": None,
}
AUTHOR_NIMROD = {"v9": "p2igan_gan_baseline_v3_pm_nimrod", "wopos_v9": "p2igan_baseline_v3_pm_nimrod",
                 "dk": "dk_v3_pm_nimrod", "stdk": "stdk_v3_pm_nimrod"}


def _infer(name: str, test_block: dict, tmp_path: Path):
    config_name = GOLDEN[name][0]
    cfg = json.loads((CONFIG_DIR / config_name).read_text())
    cfg["data"]["test"] = test_block
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))

    infer = _load_script("infer")
    out = tmp_path / "out.zarr"
    infer.main(
        Namespace(
            config=config_path,
            checkpoint=Path(CHECKPOINTS[config_name]),
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
    return zarr.open(str(out), mode="r")


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="golden outputs were produced on CUDA")


@pytest.mark.slow
@needs_cuda
@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_inference_matches_golden(name, tmp_path, in_root):
    ckpt = CHECKPOINTS[GOLDEN[name][0]]
    _require(ckpt, _golden(name), "datasets/midas/midas_test.zarr", "datasets/masks/gauge_mask_128_train.txt")
    gauge_test = json.loads((CONFIG_DIR / GAUGE_TEST_CONFIG).read_text())["data"]["test"]
    _assert_close(_infer(name, gauge_test, tmp_path), zarr.open(_golden(name), mode="r"), GOLDEN[name][3])


@pytest.mark.slow
@needs_cuda
@pytest.mark.parametrize("name", sorted(AUTHOR_NIMROD))
def test_nimrod_inference_matches_author(name, tmp_path, in_root):
    author_path = f"datasets/infer/{AUTHOR_NIMROD[name]}.zarr"
    _require(CHECKPOINTS[GOLDEN[name][0]], author_path, NIMROD_TEST["data_root"], NIMROD_TEST["mask"]["file"])
    _assert_close(_infer(name, NIMROD_TEST, tmp_path), zarr.open(author_path, mode="r"), GOLDEN[name][3])


@pytest.mark.parametrize("name", sorted(n for n in GOLDEN if GOLDEN[n][2]))
def test_golden_close_to_author_output(name, in_root):
    author_path = f"datasets/infer/{GOLDEN[name][2]}.zarr"
    _require(_golden(name), author_path)
    _assert_close(zarr.open(_golden(name), mode="r"), zarr.open(author_path, mode="r"), GOLDEN[name][3])
