from __future__ import annotations

import json
from pathlib import Path

import torch

from p2igan_bench.models import build_discriminator, build_generator
from p2igan_bench.models.p2igan import P2IGeneratorV2
from p2igan_bench.modules.layer import RobustInputBlock, _get_grid_points_matched, idw_3d_knn

CONFIG = Path(__file__).resolve().parents[1] / "p2igan_bench" / "config" / "p2igan_gan_domain.json"


def _cfg():
    cfg = json.loads(CONFIG.read_text())
    cfg["model"]["name"] = "p2igan_v2"
    return cfg


def test_robust_inputblock_is_plain_idw():
    torch.manual_seed(0)
    d, h, w = 16, 32, 32
    mask2d = torch.zeros(h, w)
    mask2d.view(-1)[torch.randperm(h * w)[:20]] = 1
    mask = mask2d.expand(d, h, w).contiguous()
    x = torch.rand(d, h, w) * mask

    out = RobustInputBlock()(x[None], mask[None])[0]

    tz, ty, tx = torch.nonzero(mask > 0, as_tuple=True)
    points = torch.stack([tx.float() / (w - 1), ty.float() / (h - 1), tz.float() / (d - 1)], -1)
    grid = _get_grid_points_matched(d, h, w, x.device, torch.float32)
    ref = idw_3d_knn(points, x[tz, ty, tx], (d, h, w), k=4, rho=2.0, tau=0.05, chunk=16384,
                     dtype=torch.float32, use_amp=False, grid_points=grid)
    assert torch.equal(out, ref)


def test_matched_grid_coord_equals_gauge_coord():
    d, h, w = 16, 32, 32
    grid = _get_grid_points_matched(d, h, w, torch.device("cpu"), torch.float32).view(d, h, w, 3)
    for z, y, x in [(0, 0, 0), (5, 17, 9), (15, 31, 31)]:
        gauge = torch.tensor([x / (w - 1), y / (h - 1), z / (d - 1)])
        assert torch.equal(grid[z, y, x], gauge)  # no float drift between the two coordinate paths


def test_v2_generator_matches_v1_except_inputblock():
    cfg = _cfg()
    gen = build_generator(cfg)
    assert isinstance(gen, P2IGeneratorV2)
    assert sum(p.numel() for p in gen.input.parameters()) == 0
    v1 = build_generator({**cfg, "model": {**cfg["model"], "name": "p2igan"}})

    def non_input(g):
        return sum(p.numel() for n, p in g.named_parameters() if not n.startswith("input."))

    assert non_input(gen) == non_input(v1)


def test_v2_forward_and_checkpoint_round_trip(tmp_path):
    cfg = _cfg()
    gen = build_generator(cfg)
    masked = torch.rand(1, 16, 1, 128, 128)
    masks = torch.zeros(1, 16, 1, 128, 128)
    masks[..., ::8, ::8] = 1
    out = gen(masked * masks, masks)
    assert out.shape == (1, 16, 1, 128, 128)
    torch.save(gen.state_dict(), tmp_path / "g.pt")
    build_generator(cfg).load_state_dict(torch.load(tmp_path / "g.pt", weights_only=True), strict=True)
    build_discriminator(cfg)
