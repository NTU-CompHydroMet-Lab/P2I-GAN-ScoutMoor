"""Advanced loss functions for precipitation GAN benchmarking."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ReconstructionLoss",
    "gan_loss",
    "pm_loss",
    "weighted_l1_distance",
    "softmax_temperature",
    "kl_divergence",
    "compute_forward_difference",
    "AdversarialLoss",
]


class ReconstructionLoss:
    """Weighted reconstruction loss: pixel pool + temporal regularization."""

    def __init__(self, k1_alpha: float = 0.0, use_pm: bool = False, pm_weight: float = 10.0):
        self.k1_alpha = k1_alpha
        self.use_pm = use_pm
        self.pm_weight = pm_weight

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
        # 主體使用 weighted L1；mask 參數保留以維持呼叫介面，但不再分 hole/valid。
        pool_loss = weighted_l1_distance(prediction, target)
        pred_diff = compute_forward_difference(prediction)
        true_diff = compute_forward_difference(target)
        pred_prob = softmax_temperature(pred_diff, 0.1)
        true_prob = softmax_temperature(true_diff, 0.1)
        reg_loss = kl_divergence(pred_prob, true_prob)
        pm_term = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        if self.use_pm:
            pm_term = pm_loss(prediction, target, omega=self.pm_weight)

        loss = pool_loss + self.k1_alpha * reg_loss + pm_term
        return loss, {
            "pool": float(pool_loss.detach()),
            "reg": float(reg_loss.detach()),
            "pm": float(pm_term.detach()),
        }


def weighted_l1_distance(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Weighted L1 distance from NowcastNet paper."""
    a, b, c = 0.50, 5.14, 0.12
    x_max = 0.70

    x_max_tensor = torch.tensor(x_max, device=x_true.device, dtype=x_true.dtype)
    w_max = a * torch.exp(b * x_max_tensor) + c
    w = a * torch.exp(b * x_true) + c
    weight = torch.where(x_true > x_max_tensor, w_max, w)
    return torch.mean(weight * torch.abs(x_pred - x_true))


def pm_loss(pred: torch.Tensor, target: torch.Tensor, omega: float = 10.0) -> torch.Tensor:
    """Pixel-wise sorted marginal matching loss."""

    pred = pred.float()
    target = target.float()

    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)

    k = max(1, math.ceil((1.0 - 0.99) * pred_flat.size(1)))

    pred_tail = torch.topk(pred_flat, k=k, dim=1, largest=True, sorted=True).values
    target_tail = torch.topk(target_flat, k=k, dim=1, largest=True, sorted=True).values

    l_pm = torch.mean((pred_tail - target_tail) ** 2)

    return omega * l_pm

def softmax_temperature(tensor: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature-scaled softmax over flattened spatial dims."""
    size = tensor.size()
    tensor = tensor.view(size[0], size[1], -1)
    softmax = F.softmax(tensor / temperature, dim=-1)
    return softmax.view(size)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence between temporal difference distributions."""
    p = p.view(p.size(0), p.size(1), -1)
    q = q.view(q.size(0), q.size(1), -1)
    return F.kl_div(p.log(), q, reduction="batchmean")


def compute_forward_difference(series: torch.Tensor) -> torch.Tensor:
    """Forward temporal difference."""
    return series[:, 1:] - series[:, :-1]


class AdversarialLoss(nn.Module):
    """Multi-mode adversarial loss."""

    def __init__(self, loss_type: str = "nsgan", target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.loss_type = loss_type
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        if loss_type == "nsgan":
            self.criterion = nn.BCELoss()
        elif loss_type == "lsgan":
            self.criterion = nn.MSELoss()
        elif loss_type == "hinge":
            self.criterion = nn.ReLU()
        else:
            raise ValueError(f"Unsupported GAN loss type: {loss_type}")

    def forward(self, outputs: torch.Tensor, is_real: bool, is_disc: bool | None = None) -> torch.Tensor:
        if self.loss_type == "hinge":
            if is_disc is None:
                raise ValueError("`is_disc` must be set when using hinge loss.")

            if is_disc:
                if is_real:
                    return self.criterion(1 - outputs).mean()
                else:
                    return self.criterion(1 + outputs).mean()
            else:
                # generator loss: -D(G(z))
                return (-outputs).mean()

        # NSGAN / LSGAN
        labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
        return self.criterion(outputs, labels)


_GAN_LOSS_CACHE: dict[tuple[str, float, float], AdversarialLoss] = {}


def gan_loss(
    logits: torch.Tensor,
    target_is_real: bool,
    *,
    loss_type: str = "nsgan",
    is_disc: bool = False,
    target_real_label: float = 1.0,
    target_fake_label: float = 0.0,
) -> torch.Tensor:
    """GAN loss helper mirroring :class:`AdversarialLoss`."""

    key = (loss_type, target_real_label, target_fake_label)
    loss_fn = _GAN_LOSS_CACHE.get(key)
    if loss_fn is None:
        loss_fn = AdversarialLoss(loss_type, target_real_label, target_fake_label)
        _GAN_LOSS_CACHE[key] = loss_fn

    if loss_fn.real_label.device != logits.device:
        loss_fn = loss_fn.to(logits.device)
        _GAN_LOSS_CACHE[key] = loss_fn  # cache moved module

    return loss_fn(logits, target_is_real, is_disc=is_disc)


