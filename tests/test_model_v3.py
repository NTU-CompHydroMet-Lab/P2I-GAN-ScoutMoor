from __future__ import annotations

import json
from pathlib import Path

import torch

from p2igan_bench.models import build_discriminator, build_generator
from p2igan_bench.models.p2igan import P2IGeneratorV3

CONFIG = Path(__file__).resolve().parents[1] / "p2igan_bench" / "config" / "p2igan_gan_domain.json"


def _cfg():
    cfg = json.loads(CONFIG.read_text())
    cfg["model"]["name"] = "p2igan_v3"
    return cfg


def _batch():
    torch.manual_seed(0)
    obs = torch.rand(2, 16, 1, 128, 128)
    mask = torch.zeros(2, 16, 1, 128, 128)
    mask[..., ::8, ::8] = 1
    return obs * mask, mask


def test_v3_takes_three_channel_groups_and_keeps_zero_input_params():
    gen = build_generator(_cfg())
    assert isinstance(gen, P2IGeneratorV3)
    assert gen.Convsin[0].main[0].in_channels == 3 * 16  # IDW field + masked obs + mask
    assert sum(p.numel() for p in gen.input.parameters()) == 0
    build_discriminator(_cfg())


def test_v3_composites_observations_at_gauge_pixels():
    gen = build_generator(_cfg()).eval()
    masked, mask = _batch()
    with torch.no_grad():
        out = gen(masked, mask)
    gate = mask.bool()
    assert torch.equal(out[gate], masked[gate])  # gauge pixels copy the observation exactly
    assert out[~gate].abs().max() <= 1.0  # elsewhere the tanh output


def test_v3_no_gradient_to_network_at_gauge_pixels():
    gen = build_generator(_cfg())
    masked, mask = _batch()
    out = gen(masked, mask)
    # a loss living only on gauge pixels has nothing to train: those outputs are constant observations
    (out * mask).sum().backward()
    grads = [p.grad for p in gen.parameters() if p.grad is not None and p.requires_grad]
    assert all(torch.count_nonzero(g) == 0 for g in grads)


def test_v3_checkpoint_round_trips(tmp_path):
    cfg = _cfg()
    gen = build_generator(cfg)
    torch.save(gen.state_dict(), tmp_path / "g.pt")
    build_generator(cfg).load_state_dict(torch.load(tmp_path / "g.pt", weights_only=True), strict=True)
