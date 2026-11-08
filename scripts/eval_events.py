"""Evaluate a checkpoint on named rain events, against IDW from the same gauge pixels.

Fields are read straight from the monthly NIMROD stores, so events need not lie in the cropped
domain store. Outputs land in ``--out``: ``metrics.json``, ``report.md``, and per event a panel around
the peak, a time series and (unless ``--no-gif``) a three-hour GIF.

    python scripts/eval_events.py --config weights/domain-v1/logs/config-*.json \\
        --checkpoint weights/domain-v1/best.pt --out experiments/results/domain-v1-pilot
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from p2igan_bench.codec import RainCodec  # noqa: E402
from p2igan_bench.config import load_config  # noqa: E402
from p2igan_bench.data.domain_dataset import RAW_PER_MMH, gauge_pixels  # noqa: E402
from p2igan_bench.metrics import RainScores  # noqa: E402
from p2igan_bench.models import build_generator  # noqa: E402
from p2igan_bench.modules.layer import idw_3d_knn  # noqa: E402

SOURCE = "/home/NAS/house/isaac-10009/Data/RPP/NIMROD/NIMROD_ZARR"
METHODS = {"model": "P2I-GAN", "idw": "IDW (same gauges)"}
COLORS = {"truth": "#333333", "model": "#1f5fbf", "idw": "#d9661f"}
FSS_WINDOWS = (1, 5, 11, 21)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="training config the checkpoint was trained with")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--events", type=Path, default=Path("p2igan_bench/config/eval/pilot_events.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", default=SOURCE, help="directory of monthly NIMROD stores")
    p.add_argument("--store", type=Path, default=Path("datasets/nimrod_domain.zarr"), help="domain store (grid only)")
    p.add_argument("--gauges", type=Path, default=Path("datasets/gauges_in_domain.csv"))
    p.add_argument("--stride", type=int, default=4, help="window step; overlapping predictions are averaged")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-gif", action="store_true")
    return p.parse_args()


def load_event(source, start, days, rows, cols):
    """Raw int16 fields (mm/h * 32, -1 missing) for ``days`` whole days from ``start``, 5-minute steps."""
    t0 = dt.datetime.fromisoformat(start)
    parts = []
    for d in range(days):
        day = t0 + dt.timedelta(days=d)
        store = zarr.open(f"{source}/{day:%Y%m}.zarr", mode="r")
        i0 = (day - day.replace(day=1)).days * 288
        parts.append(store["rainrate"][i0 : i0 + 288, rows[0] : rows[1], cols[0] : cols[1]])
    times = [t0 + dt.timedelta(minutes=5 * i) for i in range(288 * days)]
    return np.concatenate(parts), times


def sliding(fn, n, length, stride):
    starts = list(range(0, n - length + 1, stride))
    if starts[-1] != n - length:
        starts.append(n - length)
    acc, weight = None, np.zeros(n)
    for s in starts:
        y = fn(s)
        if acc is None:
            acc = np.zeros((n, *y.shape[1:]))
        acc[s : s + length] += y
        weight[s : s + length] += 1
    return acc / weight[:, None, None]


def ssim_frames(a, b, codec):
    """Mean SSIM over frames on the model scale (Gaussian window 11, sigma 1.5, data range 1)."""
    x = torch.from_numpy(codec.encode(a)).float()[:, None]
    y = torch.from_numpy(codec.encode(b)).float()[:, None]
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5**2))
    w = ((g[:, None] * g[None]) / g.sum() ** 2)[None, None]
    mx, my = F.conv2d(x, w), F.conv2d(y, w)
    sxx, syy, sxy = F.conv2d(x * x, w) - mx**2, F.conv2d(y * y, w) - my**2, F.conv2d(x * y, w) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def event_scores(est, truth, mask, codec):
    """Scores of ``est`` against ``truth`` (mm/h, NaN = missing) over frames without missing pixels."""
    full = ~np.isnan(truth).any(axis=(1, 2))
    e, t = est[full], truth[full]
    rs = RainScores(codec, fss_windows=FSS_WINDOWS)
    m = np.broadcast_to(mask, t.shape)
    rs.update(torch.from_numpy(codec.encode(e)).float(), torch.from_numpy(codec.encode(t)).float(),
              torch.from_numpy(np.ascontiguousarray(m)).float())
    out = rs.compute()
    # RainScores decodes from the model scale; recompute point scores on the exact fields.
    off = m == 0
    d = e[off] - t[off]
    out.update(mae_mmh=float(np.abs(d).mean()), rmse_mmh=float(np.sqrt((d**2).mean())), bias_mmh=float(d.mean()),
               r=float(np.corrcoef(e[off], t[off])[0, 1]))
    wet = (t >= 0.1).mean(axis=(1, 2)) >= 0.05
    out["ssim"] = ssim_frames(e[wet], t[wet], codec) if wet.any() else float("nan")
    out["frames_scored"] = int(full.sum())
    return out


def _cmap():
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("white")
    cmap.set_bad("#bbbbbb")
    return cmap, LogNorm(vmin=0.1, vmax=64)


def _show(ax, field, cmap, norm):
    return ax.imshow(np.where(np.isnan(field), np.nan, np.maximum(field, 1e-3)), cmap=cmap, norm=norm,
                     interpolation="nearest")


def plot_peak(name, fields, mask, times, peak, path):
    cmap, norm = _cmap()
    cols = [peak - 6, peak - 3, peak, peak + 3]
    rows = [("truth", "Radar (truth)"), *METHODS.items()]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.6), constrained_layout=True)
    gy, gx = np.nonzero(mask)
    for i, (key, label) in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            im = _show(ax, fields[key][c], cmap, norm)
            if key == "truth":
                ax.scatter(gx, gy, s=3, c="#d62728", linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{times[c]:%Y-%m-%d %H:%M} UTC", fontsize=10)
            if j == 0:
                ax.set_ylabel(label, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, extend="both").set_label("Rain rate (mm/h, log scale)")
    fig.suptitle(f"{name}: peak of domain-mean rain, 15-minute steps "
                 "(red: gauge pixels used as input; white: < 0.1 mm/h; grey: no radar data)", fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_series(name, fields, times, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for key, label in [("truth", "Radar (truth)"), *METHODS.items()]:
        lw = 2.2 if key == "truth" else 1.6
        axes[0].plot(times, np.nanmean(fields[key], axis=(1, 2)), color=COLORS[key], lw=lw, label=label)
        axes[1].plot(times, np.nanpercentile(fields[key], 99, axis=(1, 2)), color=COLORS[key], lw=lw, label=label)
    axes[0].set_ylabel("Domain mean (mm/h)")
    axes[1].set_ylabel("Domain 99th percentile (mm/h)")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"{name}: rain over the domain", fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_gif(name, fields, mask, times, peak, path):
    cmap, norm = _cmap()
    gy, gx = np.nonzero(mask)
    frames = []
    for k in range(max(peak - 18, 0), min(peak + 18, len(times))):
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), constrained_layout=True)
        for ax, (key, label) in zip(axes, [("truth", "Radar (truth)"), *METHODS.items()], strict=True):
            im = _show(ax, fields[key][k], cmap, norm)
            if key == "truth":
                ax.scatter(gx, gy, s=2, c="#d62728", linewidths=0)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.8, extend="both").set_label("mm/h")
        fig.suptitle(f"{name}  {times[k]:%Y-%m-%d %H:%M} UTC", fontsize=11)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, frames, duration=1 / 6, loop=0)


def plot_categorical(results, path):
    names = list(results)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), sharey="row", constrained_layout=True)
    for i, metric in enumerate(("pod", "far", "csi")):
        for j, thr in enumerate((1, 5, 10)):
            ax = axes[i, j]
            xs = np.arange(len(names))
            for off, (key, label) in zip((-0.12, 0.12), METHODS.items(), strict=True):
                ax.scatter(xs + off, [results[n][key][f"{metric}@{thr}mmh"] for n in names], s=40, color=COLORS[key],
                           label=label)
            ax.set_xticks(xs, names if i == 2 else [], rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(f">= {thr} mm/h", fontsize=11)
            if j == 0:
                ax.set_ylabel(metric.upper() + (" (lower is better)" if metric == "far" else ""))
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Categorical scores on pixels away from the gauges", fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(results, meta, path):
    cols = ["mae_mmh", "rmse_mmh", "bias_mmh", "r", "ssim", "csi@1mmh", "csi@5mmh", "csi@10mmh", "fss@5mmh_11km"]
    lines = [f"# Event evaluation: {meta['checkpoint']}", "",
             f"Epoch {meta['epoch']}, step {meta['global_step']}. Point and categorical scores use pixels away "
             "from the gauges; FSS uses the full field; SSIM is on the model scale over frames with at least 5% "
             "of the domain wet.", "",
             "| Event | Method | " + " | ".join(cols) + " |", "| --- | --- | " + " | ".join("---" for _ in cols) + " |"]
    for name, res in results.items():
        for key, label in METHODS.items():
            row = " | ".join(f"{res[key][c]:.3f}" for c in cols)
            lines.append(f"| {name if key == 'model' else ''} | {label} | {row} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    length = int(cfg["data"]["train"].get("sample_length", 16))
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    codec = RainCodec.from_dict(state.get("codec"))
    gen = build_generator(cfg).to(device).eval()
    gen.load_state_dict(state["generator"])

    grid = zarr.open_group(str(args.store), mode="r")
    rows, cols = grid.attrs["rows"], grid.attrs["cols"]
    gauges = gauge_pixels(str(args.gauges), grid["easting"][:], grid["northing"][:])
    h, w = rows[1] - rows[0], cols[1] - cols[0]
    mask = np.zeros((h, w), np.float32)
    mask[gauges[:, 0], gauges[:, 1]] = 1
    m_t = torch.from_numpy(mask).to(device)
    tz, ty, tx = torch.nonzero(m_t.expand(length, h, w) > 0, as_tuple=True)
    points = torch.stack([tx.float() / (w - 1), ty.float() / (h - 1), tz.float() / (length - 1)], -1)

    events = json.loads(args.events.read_text())["events"]
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for ev in events:
        raw, times = load_event(args.source, ev["start"], ev["days"], rows, cols)
        truth = np.where(raw >= 0, raw / RAW_PER_MMH, np.nan)
        x = codec.encode(np.nan_to_num(truth)).astype(np.float32)

        def model_window(s, x=x):
            xs = torch.from_numpy(x[s : s + length]).to(device)
            masks = m_t.expand(length, h, w)[None, :, None].contiguous()
            with torch.no_grad():
                return gen((xs * m_t)[None, :, None], masks)[0, :, 0].clamp(0, 1).cpu().numpy()

        def idw_window(s, x=x):
            xs = torch.from_numpy(x[s : s + length]).to(device)
            with torch.no_grad():
                out = idw_3d_knn(points, xs[tz, ty, tx], (length, h, w), k=4, rho=2.0, tau=0.05, chunk=16384,
                                 device=device, dtype=torch.float32, use_amp=False)
            return out.float().cpu().numpy()

        fields = {"truth": truth,
                  "model": codec.decode(sliding(model_window, len(x), length, args.stride)),
                  "idw": codec.decode(sliding(idw_window, len(x), length, args.stride))}
        results[ev["name"]] = {key: event_scores(fields[key], truth, mask, codec) for key in METHODS}
        results[ev["name"]]["window"] = [times[0].isoformat(), times[-1].isoformat()]
        results[ev["name"]]["radar_max_mmh"] = float(np.nanmax(truth))

        tag = ev["start"]
        peak = int(np.clip(np.nanargmax(np.nanmean(truth, axis=(1, 2))), 6, len(times) - 4))
        plot_peak(ev["name"], fields, mask, times, peak, args.out / f"{tag}_peak.png")
        plot_series(ev["name"], fields, times, args.out / f"{tag}_series.png")
        if not args.no_gif:
            write_gif(ev["name"], fields, mask, times, peak, args.out / f"{tag}_peak3h.gif")
        print(f"{ev['name']}: model csi@5 {results[ev['name']]['model']['csi@5mmh']:.3f}, "
              f"idw csi@5 {results[ev['name']]['idw']['csi@5mmh']:.3f}", flush=True)

    meta = {"checkpoint": str(args.checkpoint), "epoch": state.get("epoch"), "global_step": state.get("global_step"),
            "codec": codec.to_dict(), "stride": args.stride}
    (args.out / "metrics.json").write_text(json.dumps({"meta": meta, "events": results}, indent=1))
    plot_categorical({n: r for n, r in results.items()}, args.out / "summary_categorical.png")
    write_report(results, meta, args.out / "report.md")


if __name__ == "__main__":
    main()
