"""Rain scores accumulated over batches and computed in mm/h.

Point and categorical scores use only pixels outside the observation mask; FSS uses the full field.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from p2igan_bench.codec import P2I_CODEC, RainCodec


class RainScores:
    def __init__(
        self,
        codec: RainCodec = P2I_CODEC,
        thresholds: Sequence[float] = (1.0, 5.0, 10.0),
        fss_thresholds: Sequence[float] = (1.0, 5.0),
        fss_window: int = 11,
    ):
        self.codec = codec
        self.thresholds = tuple(thresholds)
        self.fss_thresholds = tuple(fss_thresholds)
        self.fss_window = fss_window
        self.reset()

    def reset(self) -> None:
        self.abs_sum = self.sq_sum = self.err_sum = self.count = 0.0
        self.table = {thr: [0.0, 0.0, 0.0] for thr in self.thresholds}  # hits, false alarms, misses
        self.fss_sums = {thr: [0.0, 0.0] for thr in self.fss_thresholds}  # sum (fe-ft)^2, sum fe^2 + ft^2

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor, obs_mask: torch.Tensor) -> None:
        """``pred``, ``target``, ``obs_mask``: same shape ``(..., H, W)`` on the model scale; mask is 1 at inputs."""
        p = self.codec.decode(pred.clamp(0.0, 1.0))
        t = self.codec.decode(target)
        off = obs_mask == 0
        pe, te = p[off], t[off]
        diff = pe - te
        self.abs_sum += float(diff.abs().sum())
        self.sq_sum += float((diff * diff).sum())
        self.err_sum += float(diff.sum())
        self.count += diff.numel()
        for thr in self.thresholds:
            yes_p, yes_t = pe >= thr, te >= thr
            row = self.table[thr]
            row[0] += float((yes_p & yes_t).sum())
            row[1] += float((yes_p & ~yes_t).sum())
            row[2] += float((~yes_p & yes_t).sum())
        h, w = p.shape[-2:]
        p2, t2 = p.reshape(-1, 1, h, w), t.reshape(-1, 1, h, w)
        pad = self.fss_window // 2
        for thr in self.fss_thresholds:
            fe = F.avg_pool2d((p2 >= thr).float(), self.fss_window, stride=1, padding=pad, count_include_pad=True)
            ft = F.avg_pool2d((t2 >= thr).float(), self.fss_window, stride=1, padding=pad, count_include_pad=True)
            sums = self.fss_sums[thr]
            sums[0] += float(((fe - ft) ** 2).sum())
            sums[1] += float((fe * fe).sum() + (ft * ft).sum())

    def compute(self) -> dict[str, float]:
        n = max(self.count, 1.0)
        out = {"mae_mmh": self.abs_sum / n, "rmse_mmh": (self.sq_sum / n) ** 0.5, "bias_mmh": self.err_sum / n}
        for thr, (hits, fa, miss) in self.table.items():
            out[f"pod@{thr:g}mmh"] = hits / max(hits + miss, 1.0)
            out[f"far@{thr:g}mmh"] = fa / max(hits + fa, 1.0)
            out[f"csi@{thr:g}mmh"] = hits / max(hits + fa + miss, 1.0)
        for thr, (mse, ref) in self.fss_sums.items():
            out[f"fss@{thr:g}mmh_{self.fss_window}km"] = 1.0 - mse / ref if ref > 0 else float("nan")
        return out
