from __future__ import annotations

import numpy as np
import pytest
import torch

from p2igan_bench.codec import P2I_CODEC, RainCodec
from p2igan_bench.metrics import RainScores


def test_codec_round_trip_and_floor():
    rain = np.array([0.0, 0.03, 0.5, 2.0, 30.0, 300.0])
    back = P2I_CODEC.decode(P2I_CODEC.encode(rain))
    assert back[0] == 0.0 and back[1] == 0.0  # at or below 0.036 mm/h encodes to 0
    assert np.allclose(back[2:], rain[2:], rtol=1e-9)
    t = torch.tensor(rain, dtype=torch.float32)
    assert torch.allclose(P2I_CODEC.decode(P2I_CODEC.encode(t))[2:], t[2:], rtol=1e-4)
    assert RainCodec.from_dict(P2I_CODEC.to_dict()) == P2I_CODEC


def _field(values):
    return P2I_CODEC.encode(torch.tensor(values, dtype=torch.float32))


def test_perfect_prediction_scores():
    torch.manual_seed(0)
    truth = _field(np.random.default_rng(0).gamma(0.5, 4.0, size=(2, 3, 32, 32)))
    mask = torch.zeros_like(truth)
    scores = RainScores(fss_window=5)
    scores.update(truth, truth, mask)
    out = scores.compute()
    assert out["mae_mmh"] == pytest.approx(0.0, abs=1e-6)
    for thr in (1, 5, 10):
        assert out[f"csi@{thr}mmh"] == 1.0 and out[f"far@{thr}mmh"] == 0.0
    assert out["fss@1mmh_5km"] == pytest.approx(1.0)


def test_contingency_counts_ignore_observed_pixels():
    truth = _field([[[0.0, 6.0, 6.0, 0.0]]])
    pred = _field([[[6.0, 6.0, 0.0, 0.0]]])
    mask = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    scores = RainScores(thresholds=(5.0,), fss_thresholds=(5.0,), fss_window=1)
    scores.update(pred, truth, mask)
    out = scores.compute()
    # unobserved pixels: pred yes/truth no, yes/yes, no/yes -> 1 hit, 1 false alarm, 1 miss
    assert out["pod@5mmh"] == pytest.approx(0.5)
    assert out["far@5mmh"] == pytest.approx(0.5)
    assert out["csi@5mmh"] == pytest.approx(1 / 3)
