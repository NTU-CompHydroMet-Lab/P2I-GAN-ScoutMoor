from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter

from .io import align_length, crop_center, ensure_thw, select_by_mask


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def mean_frame_psnr(pred: np.ndarray, gt: np.ndarray, max_value: float = 200.0) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("Expected masked frame arrays with shape [T, N]")

    finite = np.isfinite(pred) & np.isfinite(gt)
    if not np.any(finite):
        return float("nan")

    pred_safe = np.where(finite, pred, 0.0)
    gt_safe = np.where(finite, gt, 0.0)
    counts = finite.sum(axis=1)
    valid = counts > 0
    if not np.any(valid):
        return float("nan")

    mse = np.divide(
        ((pred_safe - gt_safe) ** 2 * finite).sum(axis=1),
        counts,
        out=np.full(pred.shape[0], np.nan, dtype=np.float64),
        where=valid,
    )
    psnr = np.full(pred.shape[0], np.nan, dtype=np.float64)
    positive = valid & (mse > 0)
    psnr[positive] = 20.0 * np.log10(max_value) - 10.0 * np.log10(mse[positive])
    psnr[valid & (mse == 0)] = float("inf")

    finite_psnr = psnr[np.isfinite(psnr)]
    if finite_psnr.size == 0:
        return float("nan")
    return float(np.mean(finite_psnr))


def pss(pred: np.ndarray,
        gt: np.ndarray,
        bins: int = 50,
        min_value: float = 0.5,
        value_range: tuple[float, float] | None = None) -> float:
    pred = np.asarray(pred, dtype=np.float32).ravel()
    gt = np.asarray(gt, dtype=np.float32).ravel()

    pred = pred[np.isfinite(pred)]
    gt = gt[np.isfinite(gt)]

    if min_value is not None:
        pred = pred[pred > min_value]
        gt = gt[gt > min_value]

    if pred.size == 0 or gt.size == 0:
        return float("nan")

    if value_range is None:
        both = np.concatenate([pred, gt])
        vmin = float(both.min())
        vmax = float(both.max())
        if vmin == vmax:
            vmax = vmin + 1e-6
        value_range = (vmin, vmax)

    p_hist, _ = np.histogram(pred, bins=bins, range=value_range)
    g_hist, _ = np.histogram(gt, bins=bins, range=value_range)

    p_freq = p_hist / (p_hist.sum() + 1e-12)
    g_freq = g_hist / (g_hist.sum() + 1e-12)

    return float(np.minimum(p_freq, g_freq).sum())


def ssim2d(a: np.ndarray, b: np.ndarray, c1: float = 0.01**2, c2: float = 0.03**2) -> float:
    mu_a = a.mean()
    mu_b = b.mean()
    sig_a = ((a - mu_a) ** 2).mean()
    sig_b = ((b - mu_b) ** 2).mean()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    return float(num / (den + 1e-10))


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float()


def _ensure_btchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:  # [T,H,W]
        x = x.unsqueeze(0).unsqueeze(2)
    elif x.ndim == 4:  # [B,T,H,W]
        x = x.unsqueeze(2)
    return x


def _pool8(x: torch.Tensor) -> torch.Tensor:
    b, t, c, h, w = x.shape
    y = F.avg_pool2d(x.reshape(b * t, c, h, w), kernel_size=8, stride=8)
    return y.view(b, t, c, y.shape[-2], y.shape[-1])


def _ssim_reduce_torch(a: torch.Tensor,
                       b: torch.Tensor,
                       c1: float = 0.01**2,
                       c2: float = 0.03**2,
                       dims: tuple[int, ...] = (-2, -1)) -> torch.Tensor:
    mu_a = a.mean(dim=dims, keepdim=True)
    mu_b = b.mean(dim=dims, keepdim=True)
    sig_a = ((a - mu_a) ** 2).mean(dim=dims, keepdim=True)
    sig_b = ((b - mu_b) ** 2).mean(dim=dims, keepdim=True)
    sig_ab = ((a - mu_a) * (b - mu_b)).mean(dim=dims, keepdim=True)
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    out = num / (den + 1e-10)
    reduce_dims = sorted(((d if d >= 0 else out.ndim + d) for d in dims), reverse=True)
    for d in reduce_dims:
        out = out.squeeze(d)
    return out


def _prepare_spatiotemporal_tensors(pred: np.ndarray,
                                    gt: np.ndarray,
                                    use_pool8: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    pred_t = _ensure_btchw(_to_tensor(pred))
    gt_t = _ensure_btchw(_to_tensor(gt))
    if use_pool8:
        pred_t = _pool8(pred_t)
        gt_t = _pool8(gt_t)
    return pred_t[:, :, 0], gt_t[:, :, 0]


def _masked_series(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    masked = _apply_mask_mode(pred, gt, mask, mode)
    return masked["pred"], masked["gt"]


def _ssim_series_masked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str) -> float:
    pred_sel, gt_sel = _masked_series(pred, gt, mask, mode)
    pred_t = torch.from_numpy(pred_sel).float()
    gt_t = torch.from_numpy(gt_sel).float()
    vals = _ssim_reduce_torch(pred_t, gt_t, dims=(-1,))
    return float(vals.mean().item())


def _dtssim_series_masked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str, lag: int) -> float:
    pred_sel, gt_sel = _masked_series(pred, gt, mask, mode)
    pred_t = torch.from_numpy(pred_sel).float()
    gt_t = torch.from_numpy(gt_sel).float()
    if pred_t.shape[0] <= lag:
        return float("nan")
    pred_diff = pred_t[lag:] - pred_t[:-lag]
    gt_diff = gt_t[lag:] - gt_t[:-lag]
    vals = _ssim_reduce_torch(pred_diff, gt_diff, dims=(-1,))
    return float(vals.mean().item())


def ssim_spatial(pred: np.ndarray, gt: np.ndarray, use_pool8: bool = True) -> float:
    pred_t, gt_t = _prepare_spatiotemporal_tensors(pred, gt, use_pool8=use_pool8)
    vals = _ssim_reduce_torch(pred_t, gt_t)
    return float(vals.mean().item())


def delta_tssim(pred: np.ndarray, gt: np.ndarray, lag: int = 1, use_pool8: bool = True) -> float:
    pred_t, gt_t = _prepare_spatiotemporal_tensors(pred, gt, use_pool8=use_pool8)
    if pred_t.shape[1] <= lag:
        return float("nan")
    pred_diff = pred_t[:, lag:] - pred_t[:, :-lag]
    gt_diff = gt_t[:, lag:] - gt_t[:, :-lag]
    vals = _ssim_reduce_torch(pred_diff, gt_diff)
    return float(vals.mean().item())


def nse(pred: np.ndarray, gt: np.ndarray) -> float:
    num = np.sum((pred - gt) ** 2)
    den = np.sum((gt - np.mean(gt)) ** 2)
    return float(1.0 - num / (den + 1e-10))


def mean_nonnegative_frame_nse(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("Expected masked frame arrays with shape [T, N]")
    finite = np.isfinite(pred) & np.isfinite(gt)
    if not np.any(finite):
        return float("nan")

    pred_safe = np.where(finite, pred, 0.0)
    gt_safe = np.where(finite, gt, 0.0)
    counts = finite.sum(axis=1)
    valid = counts > 0
    if not np.any(valid):
        return float("nan")

    gt_mean = np.divide(
        gt_safe.sum(axis=1),
        counts,
        out=np.zeros(pred.shape[0], dtype=np.float64),
        where=valid,
    )
    num = ((pred_safe - gt_safe) ** 2 * finite).sum(axis=1)
    den = (((gt_safe - gt_mean[:, None]) ** 2) * finite).sum(axis=1)
    nse_frames = np.full(pred.shape[0], np.nan, dtype=np.float64)
    nse_frames[valid] = 1.0 - num[valid] / (den[valid] + 1e-10)
    nse_frames = nse_frames[np.isfinite(nse_frames) & (nse_frames >= 0.0)]
    if nse_frames.size == 0:
        return float("nan")
    return float(np.mean(nse_frames))


def transform_mmhr(arr: np.ndarray, divide_by_3: bool = True) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if divide_by_3:
        arr = arr / 3.0
    out = 10 ** (arr * 0.0625) * 0.036
    return np.clip(out, 0.0, 200.0)


def categorical_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float) -> dict[str, float]:
    pred_bin = pred >= threshold
    gt_bin = gt >= threshold
    hits = np.logical_and(pred_bin, gt_bin).sum()
    misses = np.logical_and(~pred_bin, gt_bin).sum()
    false_alarms = np.logical_and(pred_bin, ~gt_bin).sum()
    correct_negatives = np.logical_and(~pred_bin, ~gt_bin).sum()
    pod = hits / (hits + misses + 1e-10)
    far = false_alarms / (hits + false_alarms + 1e-10)
    csi = hits / (hits + misses + false_alarms + 1e-10)
    n_total = hits + misses + false_alarms + correct_negatives
    if n_total > 0:
        hss = 2 * (hits * correct_negatives - misses * false_alarms) / (
            misses**2 + false_alarms**2 + 2 * hits * correct_negatives +
             (misses + false_alarms) * (hits + correct_negatives) + 1e-10
        )
    else:
        hss = float("nan")
    return {"POD": float(pod), "FAR": float(far), "CSI": float(csi), "HSS": float(hss)}


def fss_window(obs: np.ndarray,
               pred: np.ndarray,
               threshold: float,
               win: int,
               valid_mask: np.ndarray | None = None) -> float:
    obs = ensure_thw(obs).astype(np.float32, copy=False)
    pred = ensure_thw(pred).astype(np.float32, copy=False)
    obs_bin = (obs >= threshold).astype(np.float32)
    pred_bin = (pred >= threshold).astype(np.float32)
    obs_frac = uniform_filter(obs_bin, size=(1, win, win), mode="constant")
    pred_frac = uniform_filter(pred_bin, size=(1, win, win), mode="constant")
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != obs.shape[1:]:
            raise ValueError(f"Mask shape {valid_mask.shape} != data shape {obs.shape[1:]}")
        num = np.mean(((obs_frac - pred_frac)[:, valid_mask]) ** 2)
        denom = np.mean((obs_frac[:, valid_mask] ** 2) + (pred_frac[:, valid_mask] ** 2))
    else:
        num = np.mean((obs_frac - pred_frac) ** 2)
        denom = np.mean(obs_frac ** 2 + pred_frac ** 2)
    return float(1.0 - num / (denom + 1e-10))


def _apply_mask_mode(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    if mode == "radar":
        pred_sel = select_by_mask(pred, mask, invert=True)
        gt_sel = select_by_mask(gt, mask, invert=True)
    elif mode == "gauge":
        pred_sel = select_by_mask(pred, mask, invert=False)
        gt_sel = select_by_mask(gt, mask, invert=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return {"pred": pred_sel, "gt": gt_sel}


def run_exp1(preds: dict[str, np.ndarray | dict[str, np.ndarray]],
             truth: np.ndarray | dict[str, np.ndarray],
             mask: np.ndarray,
             mode: str,
             crop_size: int,
             thresholds: tuple[float, ...] = (0.5, 2.0, 4.0, 8.0),
             fss_scales: tuple[int, ...] = (1, 2, 4, 8),
             use_pool8: bool = True,
             divide_by_3: bool = True,
             progress: bool = False) -> dict[str, dict[str, float]]:
    if isinstance(truth, dict):
        event_keys = list(truth.keys())
        if not event_keys:
            return {}

        truth_list = [truth[k] for k in event_keys if truth.get(k) is not None]
        truth_arr = np.concatenate(truth_list, axis=0) if truth_list else np.empty((0,))
        preds_concat: dict[str, np.ndarray] = {}
        for name, pred_src in preds.items():
            if not isinstance(pred_src, dict):
                continue
            pred_list = [pred_src[k] for k in event_keys if pred_src.get(k) is not None]
            pred_arr = np.concatenate(pred_list, axis=0) if pred_list else np.empty((0,))
            preds_concat[name] = pred_arr

        preds = preds_concat
        truth = truth_arr

    results: dict[str, dict[str, float]] = {}
    truth_raw_cropped = crop_center(truth, crop_size)
    truth_mmhr_cropped = transform_mmhr(truth_raw_cropped, divide_by_3=divide_by_3)
    pred_items = list(preds.items())
    total_methods = len(pred_items)
    fss_mask = mask if mode == "gauge" else ~mask
    if progress:
        print(f"[exp1] Starting metrics for {total_methods} methods")

    for idx, (name, pred) in enumerate(pred_items, start=1):
        if progress:
            print(f"[exp1] [{idx}/{total_methods}] Computing {name}")
        pred_raw, truth_raw_aligned = align_length(pred, truth_raw_cropped)
        pred_raw_cropped = crop_center(pred_raw, crop_size)
        pred_mmhr_cropped = transform_mmhr(pred_raw_cropped, divide_by_3=divide_by_3)
        _, truth_mmhr_aligned = align_length(pred_mmhr_cropped, truth_mmhr_cropped)
        masked_mmhr = _apply_mask_mode(pred_mmhr_cropped, truth_mmhr_aligned, mask, mode)
        pred_sel = masked_mmhr["pred"]
        gt_sel = masked_mmhr["gt"]
        masked_raw = _apply_mask_mode(pred_raw_cropped, truth_raw_aligned, mask, mode)
        pred_sel_raw = masked_raw["pred"]
        gt_sel_raw = masked_raw["gt"]

        # results[name] = {
        #     "PSNR": mean_frame_psnr(pred_sel, gt_sel),
        # }

        
        results[name] = {
            
            "MAE": mae(pred_sel, gt_sel),
            "RMSE": rmse(pred_sel, gt_sel),
            "PSS": pss(pred_sel_raw, gt_sel_raw),
            "SSIM": _ssim_series_masked(pred_mmhr_cropped, truth_mmhr_aligned, mask, mode),
            # "DTSSIM_L1": _dtssim_series_masked(pred_mmhr_cropped, truth_mmhr_aligned, mask, mode, lag=1),
            # "DTSSIM_L2": _dtssim_series_masked(pred_mmhr_cropped, truth_mmhr_aligned, mask, mode, lag=2),
            # "NSE": mean_nonnegative_frame_nse(pred_sel, gt_sel),
        }
        for thr in thresholds:
            key = f"CAT_{thr:g}"
            results[name][key] = categorical_metrics(pred_sel, gt_sel, thr)
            for scale in fss_scales:
                fss_key = f"FSS_{thr:g}_{scale}x{scale}"
                results[name][fss_key] = fss_window(
                    truth_mmhr_aligned,
                    pred_mmhr_cropped,
                    threshold=thr,
                    win=scale,
                    valid_mask=fss_mask,
                )

    if progress:
        print("[exp1] Metrics complete")
    return results
