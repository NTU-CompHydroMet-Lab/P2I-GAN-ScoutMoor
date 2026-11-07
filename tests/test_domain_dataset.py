from __future__ import annotations

import numpy as np
import torch

from p2igan_bench.data.domain_dataset import encode, select_windows


def _select(wet, peak, missing, length=4, stride=1):
    n = len(wet)
    return select_windows(np.asarray(wet, float), np.asarray(peak), np.asarray(missing, bool), length, stride,
                          0.3, 65, np.ones(n, bool)).tolist()


def test_window_needs_one_frame_passing_both_thresholds():
    wet = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    peak = [0, 10, 70, 0, 0, 0, 0, 0]  # only frame 2 passes both
    assert _select(wet, peak, [0] * 8) == [0, 1, 2]


def test_window_with_a_missing_frame_is_dropped():
    wet, peak = [0.5] * 8, [70] * 8
    missing = [0, 0, 0, 0, 1, 0, 0, 0]
    assert _select(wet, peak, missing) == [0]


def test_stride_skips_overlapping_windows():
    wet, peak = [0.5] * 10, [70] * 10
    assert _select(wet, peak, [0] * 10, stride=4) == [0, 4]


def test_encode_matches_p2i_scale():
    raw = torch.tensor([-1, 0, 1, 2, 32, 3200])  # -1 missing, 1/32 mm/h is below the 0.036 floor
    rain = np.array([0, 0, 0, 2 / 32, 1, 100])
    expected = np.clip(48 * np.log10(np.maximum(rain, 1e-6) / 0.036), 0, 255) / 255
    assert np.allclose(encode(raw).numpy(), expected, atol=1e-6)
