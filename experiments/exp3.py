from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from .exp1 import transform_mmhr
from .io import align_length, crop_center, ensure_dir, select_by_mask


def nse(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    m = np.isfinite(pred) & np.isfinite(gt)
    if not np.any(m):
        return float("nan")
    pred = pred[m]
    gt = gt[m]
    num = np.sum((pred - gt) ** 2)
    den = np.sum((gt - np.mean(gt)) ** 2)
    return float(1.0 - num / (den + 1e-10))


def _select_values(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "radar":
        pred_sel = select_by_mask(pred, mask, invert=True)
        gt_sel = select_by_mask(gt, mask, invert=True)
    elif mode == "gauge":
        pred_sel = select_by_mask(pred, mask, invert=False)
        gt_sel = select_by_mask(gt, mask, invert=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return pred_sel.ravel(), gt_sel.ravel()


def _nse_per_frame(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    pred_sel = select_by_mask(pred, mask, invert=(mode == "radar")).astype(np.float64, copy=False)
    gt_sel = select_by_mask(gt, mask, invert=(mode == "radar")).astype(np.float64, copy=False)
    t = min(pred_sel.shape[0], gt_sel.shape[0])
    pred_sel = pred_sel[:t]
    gt_sel = gt_sel[:t]
    finite = np.isfinite(pred_sel) & np.isfinite(gt_sel)
    if not np.any(finite):
        return np.full(t, np.nan, dtype=np.float64)

    pred_safe = np.where(finite, pred_sel, 0.0)
    gt_safe = np.where(finite, gt_sel, 0.0)
    counts = finite.sum(axis=1)
    out = np.full(t, np.nan, dtype=np.float64)
    valid = counts > 0
    if not np.any(valid):
        return out

    gt_mean = np.divide(gt_safe.sum(axis=1), counts, out=np.zeros(t, dtype=np.float64), where=valid)
    num = ((pred_safe - gt_safe) ** 2 * finite).sum(axis=1)
    den = (((gt_safe - gt_mean[:, None]) ** 2) * finite).sum(axis=1)
    out[valid] = 1.0 - num[valid] / (den[valid] + 1e-10)
    return out


SCATTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#9467bd', '#8c564b']


def _prepare_xy(pred, true, min_value: float, residual: bool = False):
    tf = np.asarray(true, np.float64).ravel()
    pf = np.asarray(pred, np.float64).ravel()
    n = min(tf.size, pf.size)
    tf = tf[:n]
    pf = pf[:n]
    m = np.isfinite(tf) & np.isfinite(pf)
    x = tf[m]
    y = (pf[m] - tf[m]) if residual else pf[m]
    keep = x >= min_value
    # keep = np.ones_like(x, dtype=bool)
    return x[keep], y[keep]


def _deterministic_subsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    order = np.argsort(x, kind='mergesort')
    idx = np.linspace(0, x.size - 1, num=max_points, dtype=np.int64)
    idx = np.unique(idx)
    sel = order[idx]
    return x[sel], y[sel]


def _residual_multi_row(pred_list, true, labels,
                        colors=SCATTER_COLORS,
                        figsize=(18, 3), dpi=600,
                        lim_x=(0, 32), lim_y=(-24, 8),
                        xticks=(0, 4, 8, 12, 16, 20, 24, 28, 32),
                        yticks=(-24, -20, -16, -12, -8, -4, 0, 4, 8),
                        max_points=2000, alpha=0.25, s=6,
                        min_value: float = 0.1,
                        seed=42, save_path: str | None = None) -> None:
    n = len(pred_list)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    if n == 1:
        axes = [axes]

    for i, (pred, label) in enumerate(zip(pred_list, labels)):
        ax = axes[i]
        x_all, y_all = _prepare_xy(pred, true, min_value=min_value, residual=True)
        if x_all.size == 0:
            continue
        x_plot, y_plot = _deterministic_subsample(x_all, y_all, max_points)

        ax.scatter(x_plot, y_plot, s=s, alpha=alpha, color=colors[i],
                   edgecolors='none', rasterized=True, zorder=1)
        if x_all.size >= 2:
            slope, intercept, r, _, _ = stats.linregress(x_all, y_all)
            x_line = np.linspace(lim_x[0], lim_x[1], 200)
            ax.plot(x_line, np.zeros_like(x_line), color='gray', ls=':', lw=1.0)
            ax.plot(x_line, intercept + slope * x_line, 'k--', lw=1.0)
            ax.text(0.04, 0.82, f"R²={r**2:.3f}\nslope={slope:.3f}",
                    transform=ax.transAxes, fontsize=11)

        ax.set_title(label, fontsize=13, fontweight='bold', pad=4)
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', labelsize=10, width=1.0, length=3)
        ax.grid(False)
        ax.axhline(0, color='black', lw=1.0, ls='--')

        if i == 0:
            ax.set_ylabel("Residual (Pred - Obs, mm/h)", fontsize=12)
        ax.set_xlabel("Obs (mm/h)", fontsize=12)

    plt.tight_layout(pad=1.0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def _scatter_multi_row(pred_list, true, labels,
                       colors=SCATTER_COLORS,
                       figsize=(18, 3), dpi=600,
                       lim=(0, 32),
                       xticks=(0, 4, 8, 12, 16, 20, 24, 28, 32),
                       yticks=(0, 4, 8, 12, 16, 20, 24, 28, 32),
                       max_points=2000,
                       alpha=0.25, s=6,
                       min_value: float = 0.1,
                       seed=42, save_path: str | None = None) -> None:
    n = len(pred_list)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    if n == 1:
        axes = [axes]

    x1 = np.linspace(lim[0], lim[1], 200)

    for i, (pred, label) in enumerate(zip(pred_list, labels)):
        ax = axes[i]
        x_all, y_all = _prepare_xy(pred, true, min_value=min_value, residual=False)
        if x_all.size == 0:
            continue
        x_plot, y_plot = _deterministic_subsample(x_all, y_all, max_points)

        ax.scatter(x_plot, y_plot, s=s, alpha=alpha, color=colors[i],
                   edgecolors='none', rasterized=True, zorder=1)
        if x_all.size >= 2:
            slope, intercept, r, _, _ = stats.linregress(x_all, y_all)
            ax.plot(x1, x1, color='gray', ls=':', lw=1.0)
            ax.plot(x1, intercept + slope * x1, 'k--', lw=1.0)
            ax.text(0.04, 0.82, f"R²={r**2:.3f}\nslope={slope:.3f}",
                    transform=ax.transAxes, fontsize=11)

        ax.set_title(label, fontsize=13, fontweight='bold', pad=4)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect('equal', 'box')
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', labelsize=10, width=1.0, length=3)
        ax.grid(False)

        if i == 0:
            ax.set_ylabel("Pred (mm/h)", fontsize=12)
        ax.set_xlabel("Obs (mm/h)", fontsize=12)

    plt.tight_layout(pad=1.0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def _logfreq_plot(pred_list, true, labels,
                  colors=SCATTER_COLORS,
                  figsize=(6, 4), dpi=300,
                  lim=(0, 32), bins=48,
                  smooth_sigma: float = 2.2,
                  ma_window: int = 5,
                  min_positive: float = 1e-6,
                  save_path: str | None = None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    bin_edges = np.linspace(lim[0], lim[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _smooth_hist(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr / max(arr.sum(), 1)

        if smooth_sigma > 0:
            arr = gaussian_filter1d(arr, sigma=smooth_sigma, mode="nearest")

        if ma_window > 1:
            kernel = np.ones(ma_window, dtype=np.float64) / ma_window
            arr = np.convolve(arr, kernel, mode="same")

        arr = np.clip(arr, min_positive, None)
        return arr

    tf = np.asarray(true, np.float64).ravel()
    tf = tf[np.isfinite(tf)]
    if tf.size == 0:
        plt.close(fig)
        return

    hist_t, _ = np.histogram(tf, bins=bin_edges)
    freq_t = _smooth_hist(hist_t)
    ax.semilogy(bin_centers, freq_t, color='black', lw=2.0, label='Obs')

    for pred, label, color in zip(pred_list, labels, colors):
        pf = np.asarray(pred, np.float64).ravel()
        pf = pf[np.isfinite(pf)]
        if pf.size == 0:
            continue

        hist_p, _ = np.histogram(pf, bins=bin_edges)
        freq_p = _smooth_hist(hist_p)
        ax.semilogy(bin_centers, freq_p, color=color, lw=1.8, label=label)

    ax.set_xlim(*lim)
    ax.set_xlabel("Rainfall (mm/h)")
    ax.set_ylabel("Relative Frequency (log scale)")
    ax.grid(False)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def _nse_boxplot(nse_by_method: dict[str, list[float]],
                 out_path: str) -> None:
    methods = list(nse_by_method.keys())
    data = [nse_by_method[m] for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    box = ax.boxplot(
        data,
        labels=methods,
        patch_artist=True,
        showmeans=True,
        meanline=False,
        boxprops=dict(linewidth=1.2, color='black'),
        medianprops=dict(linewidth=2.0, color='black'),
        whiskerprops=dict(linewidth=1.2, color='black'),
        capprops=dict(linewidth=1.2, color='black'),
        flierprops=dict(marker='o', markersize=3, alpha=0.5, color='gray'),
        meanprops=dict(marker='D', markerfacecolor='white',
                       markeredgecolor='black', markersize=5),
    )
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
              '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('NSE', fontsize=14)
    ax.set_xlabel('Methods', fontsize=13)
    ax.set_title(f'NSE Comparison ({len(data[0])} Rain Events)', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.2, 1.0)
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def run_exp3(preds: dict[str, np.ndarray | dict[str, np.ndarray]],
             truth: np.ndarray | dict[str, np.ndarray],
             mask: np.ndarray,
             mode: str,
             crop_size: int,
             out_dir: str) -> dict[str, float]:
    ensure_dir(out_dir)
    metrics: dict[str, float] = {}
    already_transformed = False

    nse_by_method: dict[str, list[float]] = {}
    if isinstance(truth, dict):
        event_keys = list(truth.keys())
        truth_list = []
        preds_concat: dict[str, list[np.ndarray]] = {}
        for name in preds.keys():
            nse_by_method[name] = []
            preds_concat[name] = []

        for event_key in event_keys:
            truth_ev = truth.get(event_key)
            if truth_ev is None:
                continue
            truth_ev = transform_mmhr(truth_ev, divide_by_3=True)
            truth_ev = crop_center(truth_ev, crop_size)
            appended_truth = False

            for name, pred_src in preds.items():
                if not isinstance(pred_src, dict):
                    continue
                pred_ev = pred_src.get(event_key)
                if pred_ev is None:
                    continue
                pred_ev = transform_mmhr(pred_ev, divide_by_3=True)
                pred_ev, truth_aligned = align_length(pred_ev, truth_ev)
                pred_ev = crop_center(pred_ev, crop_size)

                if not appended_truth:
                    truth_list.append(truth_aligned)
                    appended_truth = True

                pred_event_sel, gt_event_sel = _select_values(pred_ev, truth_aligned, mask, mode)
                nse_event = nse(pred_event_sel, gt_event_sel)
                nse_by_method[name].append(nse_event)
                preds_concat[name].append(pred_ev)
                truth_ev = truth_aligned

        truth = np.concatenate(truth_list, axis=0) if truth_list else np.empty((0,))
        preds = {
            name: np.concatenate(frames, axis=0) if frames else np.empty((0,))
            for name, frames in preds_concat.items()
        }
        already_transformed = True

    if not already_transformed:
        truth = transform_mmhr(truth, divide_by_3=True)
    truth = crop_center(truth, crop_size)

    pred_list = []
    gt_flat = None
    labels = []

    for name, pred in preds.items():
        if not already_transformed:
            pred = transform_mmhr(pred, divide_by_3=True)

        pred, truth_aligned = align_length(pred, truth)
        pred = crop_center(pred, crop_size)

        pred_sel, gt_sel = _select_values(pred, truth_aligned, mask, mode)

        if nse_by_method.get(name):
            event_nse = np.asarray(nse_by_method[name], dtype=np.float64)
            event_nse = event_nse[np.isfinite(event_nse) & (event_nse >= 0.0)]
            metrics[f"NSE_{name}"] = float(np.mean(event_nse)) if event_nse.size else float("nan")
            nse_by_method[name] = event_nse.tolist()
        else:
            total_nse = nse(pred_sel, gt_sel)
            metrics[f"NSE_{name}"] = total_nse if np.isfinite(total_nse) and total_nse >= 0.0 else float("nan")

        pred_list.append(pred_sel)
        labels.append(name)

        if gt_flat is None:
            gt_flat = gt_sel

    if gt_flat is None:
        gt_flat = np.empty((0,), dtype=np.float64)

    _scatter_multi_row(
        pred_list,
        gt_flat,
        labels,
        figsize=(18, 3),
        lim=(0, 32),
        dpi=300,
        max_points=20000,
        alpha=0.6,
        s=10,
        min_value=0.0,
        save_path=os.path.join(out_dir, "scatter_panels.pdf"),
    )

    _residual_multi_row(
        pred_list,
        gt_flat,
        labels,
        figsize=(18, 3),
        dpi=300,
        max_points=20000,
        alpha=0.6,
        s=10,
        min_value=0.1,
        save_path=os.path.join(out_dir, "residual_panels.pdf"),
    )

    if nse_by_method:
        _nse_boxplot(nse_by_method, os.path.join(out_dir, "nse_boxplot.pdf"))
        _logfreq_plot(
            pred_list,
            gt_flat,
            labels,
            figsize=(6, 4),
            dpi=300,
            lim=(0, 32),
            bins=64,
            save_path=os.path.join(out_dir, "logfreq.pdf"),
        )

    return metrics