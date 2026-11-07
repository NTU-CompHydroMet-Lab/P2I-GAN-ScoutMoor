from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from p2igan_bench.config import Config, load_config, validate_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "p2igan_bench" / "config"
SHIPPED = sorted(CONFIG_DIR.glob("*.json"))


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_shipped_config_validates_unchanged(path):
    assert load_config(path) == json.loads(path.read_text())


@pytest.mark.parametrize(
    "where",
    [(), ("model",), ("loss",), ("train",), ("train", "optimizer"), ("data", "train"), ("data", "train", "mask")],
)
def test_unknown_key_is_rejected(where):
    cfg = json.loads((CONFIG_DIR / "p2igan_gan_baseline.json").read_text())
    node = cfg
    for key in where:
        node = node[key]
    node["hole_weight_typo"] = 1.0
    with pytest.raises(ValidationError, match="hole_weight_typo"):
        validate_config(cfg)


def test_domain_split_parses_with_defaults():
    cfg = json.loads((CONFIG_DIR / "p2igan_gan_baseline.json").read_text())
    cfg["data"]["train"] = {
        "kind": "domain",
        "store": "datasets/nimrod_domain.zarr",
        "start": "2020-01-01",
        "end": "2023-12-31",
        "gauges": "datasets/gauges_in_domain.csv",
        "h": 128,
        "w": 128,
    }
    split = Config.model_validate(cfg).data.train
    assert split.sample_length == 16
    assert (split.selection.wet_raw, split.selection.wet_fraction, split.selection.peak_raw) == (2, 0.3, 65)


def test_events_kind_is_the_default():
    cfg = json.loads((CONFIG_DIR / "p2igan_gan_baseline.json").read_text())
    assert Config.model_validate(cfg).data.train.kind == "events"
    bad = copy.deepcopy(cfg)
    bad["data"]["train"]["kind"] = "radar"
    with pytest.raises(ValidationError):
        validate_config(bad)
