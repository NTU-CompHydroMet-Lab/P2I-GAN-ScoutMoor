"""TorchMetrics-based evaluation suite for P2I-GAN benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.image import StructuralSimilarityIndexMeasure

EPS = 1e-10


def transform(output):
    """Convert normalized values back to rainfall intensity."""
    if isinstance(output, torch.Tensor):
        return torch.pow(10.0, output * 0.0625) * 0.036
    return (10.0 ** (output * 0.0625)) * 0.036


def _flatten_spatial(tensor: torch.Tensor) -> torch.Tensor:
    h, w = tensor.shape[-2], tensor.shape[-1]
    return tensor.reshape(-1, h, w)


class RegressionMetrics(Metric):
    """Aggregate MAE/RMSE/SSIM via TorchMetrics."""

    full_state_update = False

    def __init__(self, apply_transform: bool = True, data_range: float = 1.0):
        super().__init__()
        self.apply_transform = apply_transform
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

        self.add_state("abs_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("squared_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.detach().to(torch.float32)
        target = target.detach().to(torch.float32)

        if self.apply_transform:
            preds = transform(preds)
            target = transform(target)

        diff = preds - target
        self.abs_sum += diff.abs().sum()
        self.squared_sum += (diff ** 2).sum()
        self.n_obs += torch.tensor(diff.numel(), device=diff.device)

        preds_4d, target_4d = self._reshape_for_ssim(preds), self._reshape_for_ssim(target)
        self.ssim.update(preds_4d, target_4d)

    def _reshape_for_ssim(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 5:
            b, t, c, h, w = tensor.shape
            return tensor.reshape(b * t, c, h, w)
        if tensor.dim() == 4:
            return tensor
        raise ValueError("Expected tensor with shape [B, T, C, H, W] or [B, C, H, W].")

    def compute(self) -> dict[str, float]:
        mae = self.abs_sum / torch.clamp(self.n_obs, min=1.0)
        rmse = torch.sqrt(self.squared_sum / torch.clamp(self.n_obs, min=1.0))
        ssim_value = self.ssim.compute()
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "ssim": float(ssim_value),
        }


class CategoricalMetrics(Metric):
    """Probability of detection / false alarm / CSI / HSS across thresholds."""

    full_state_update = False

    def __init__(self, thresholds: Sequence[float]):
        super().__init__()
        thresholds_tensor = torch.tensor(thresholds, dtype=torch.float32)
        self.register_buffer("thresholds", thresholds_tensor.view(-1, 1))

        zeros = torch.zeros(len(thresholds), dtype=torch.float32)
        self.add_state("hits", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("misses", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("false", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("correct", default=zeros.clone(), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = transform(preds.detach().to(torch.float32))
        target = transform(target.detach().to(torch.float32))

        preds_flat = _flatten_spatial(preds)
        target_flat = _flatten_spatial(target)

        preds_vec = preds_flat.reshape(1, -1)
        target_vec = target_flat.reshape(1, -1)
        thresholded_pred = preds_vec >= self.thresholds
        thresholded_true = target_vec >= self.thresholds

        hits = torch.logical_and(thresholded_pred, thresholded_true).sum(dim=1)
        misses = torch.logical_and(~thresholded_pred, thresholded_true).sum(dim=1)
        false_alarms = torch.logical_and(thresholded_pred, ~thresholded_true).sum(dim=1)
        correct_negatives = torch.logical_and(~thresholded_pred, ~thresholded_true).sum(dim=1)

        self.hits += hits.to(torch.float32)
        self.misses += misses.to(torch.float32)
        self.false += false_alarms.to(torch.float32)
        self.correct += correct_negatives.to(torch.float32)

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for idx, thr in enumerate(self.thresholds.view(-1)):
            hits = self.hits[idx]
            misses = self.misses[idx]
            false = self.false[idx]
            correct = self.correct[idx]

            pod = hits / (hits + misses + EPS)
            far = false / (hits + false + EPS)
            csi = hits / (hits + misses + false + EPS)
            denom = (misses + false) * (false + correct) + (hits + misses) * (misses + correct)
            hss = 2 * (hits * correct - misses * false) / (denom + EPS)

            prefix = f"cat_thr{float(thr):.2f}"
            metrics[f"{prefix}/pod"] = float(pod)
            metrics[f"{prefix}/far"] = float(far)
            metrics[f"{prefix}/csi"] = float(csi)
            metrics[f"{prefix}/hss"] = float(hss)
        return metrics


class FractionalSkillScoreMetric(Metric):
    """FSS metric across thresholds and spatial scales."""

    full_state_update = False

    def __init__(self, thresholds: Sequence[float], scales: Sequence[int]):
        super().__init__()
        self.register_buffer("thresholds", torch.tensor(thresholds, dtype=torch.float32))
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.int64))

        zeros = torch.zeros(len(thresholds), len(scales), dtype=torch.float32)
        self.add_state("score_sum", default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state("counts", default=zeros.clone(), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = transform(preds.detach().to(torch.float32))
        target = transform(target.detach().to(torch.float32))

        preds_flat = _flatten_spatial(preds).unsqueeze(1)  # [N,1,H,W]
        target_flat = _flatten_spatial(target).unsqueeze(1)

        for ti, thr in enumerate(self.thresholds):
            pred_mask = (preds_flat >= thr).to(torch.float32)
            target_mask = (target_flat >= thr).to(torch.float32)
            for si, scale in enumerate(self.scales):
                frac_pred = self._fractional_mean(pred_mask, int(scale))
                frac_true = self._fractional_mean(target_mask, int(scale))
                numerator = torch.mean((frac_pred - frac_true) ** 2)
                denominator = torch.mean(frac_pred ** 2 + frac_true ** 2)
                fss = 1.0 - numerator / (denominator + EPS)

                self.score_sum[ti, si] += fss
                self.counts[ti, si] += 1

    def _fractional_mean(self, tensor: torch.Tensor, scale: int) -> torch.Tensor:
        pad = scale // 2
        return F.avg_pool2d(tensor, kernel_size=scale, stride=1, padding=pad)

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for ti, thr in enumerate(self.thresholds):
            for si, scale in enumerate(self.scales):
                if self.counts[ti, si] == 0:
                    continue
                avg_score = self.score_sum[ti, si] / self.counts[ti, si]
                metrics[f"fss_thr{float(thr):.2f}_s{int(scale)}"] = float(avg_score)
        return metrics


@dataclass
class MetricConfig:
    thresholds: Sequence[float] = (0.5, 2.0, 4.0, 8.0)
    scales: Sequence[int] = (1, 2, 4, 8)
    apply_transform: bool = True
    data_range: float = 1.0


class RainfallMetricSuite:
    """High-level orchestrator that bundles regression/categorical/FSS metrics."""

    def __init__(self, config: MetricConfig | None = None):
        cfg = config or MetricConfig()
        self.regression = RegressionMetrics(apply_transform=cfg.apply_transform, data_range=cfg.data_range)
        self.categorical = CategoricalMetrics(cfg.thresholds)
        self.fss = FractionalSkillScoreMetric(cfg.thresholds, cfg.scales)
        self.device: torch.device | None = None

    def to(self, device: torch.device):
        self.device = device
        self.regression = self.regression.to(device)
        self.categorical = self.categorical.to(device)
        self.fss = self.fss.to(device)
        return self

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.regression.update(preds, target)
        self.categorical.update(preds, target)
        self.fss.update(preds, target)

    def compute(self) -> dict[str, float]:
        metrics = {}
        metrics.update(self.regression.compute())
        metrics.update(self.categorical.compute())
        metrics.update(self.fss.compute())
        return metrics

    def reset(self) -> None:
        self.regression.reset()
        self.categorical.reset()
        self.fss.reset()
        if self.device is not None:
            # torchmetrics resets states to defaults on CPU; move them back.
            self.to(self.device)


__all__ = [
    "transform",
    "RegressionMetrics",
    "CategoricalMetrics",
    "FractionalSkillScoreMetric",
    "RainfallMetricSuite",
    "MetricConfig",
]
