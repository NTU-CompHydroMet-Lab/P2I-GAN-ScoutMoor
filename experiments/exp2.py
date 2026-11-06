from __future__ import annotations

import os
import re
from collections.abc import Iterable

import numpy as np
from PIL import Image

from .exp1 import transform_mmhr
from .io import align_length, crop_center, ensure_dir, ensure_thw, load_mask, save_text

# Hard-coded palette for paper-style plots (matches notebook).
_PAPER_BOUNDS = [0, 0.5, 1, 2, 4, 8, 16, 200]
_PAPER_COLORS = [
    "#000000",
    "#46327e",
    "#277f8e",
    "#4ac16d",
    "#a0da39",
    "#fde725",
    "#ffffff",
]
_PAPER_SUB = 20


def _normalize_method_name(name: str) -> str:
    return re.sub(r"[^a-z0-9+]+", "", name.lower())


def _lookup_method_path(methods: dict[str, str], name: str) -> str | None:
    if name in methods:
        return methods[name]
    normalized = _normalize_method_name(name)
    for key, path in methods.items():
        if _normalize_method_name(key) == normalized:
            return path
    return None


def _to_uint8(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    frame = np.clip(frame, vmin, vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = (frame - vmin) / (vmax - vmin)
    return (norm * 255.0).astype(np.uint8)


def _save_frames(frames: np.ndarray, out_dir: str, vmin: float, vmax: float, prefix: str) -> None:
    ensure_dir(out_dir)
    frames = ensure_thw(frames)
    for i in range(frames.shape[0]):
        img = Image.fromarray(_to_uint8(frames[i], vmin, vmax))
        img.save(os.path.join(out_dir, f"{prefix}_{i:03d}.png"))


def _save_gif(frames: np.ndarray, out_path: str, vmin: float, vmax: float, fps: int) -> None:
    frames = ensure_thw(frames)
    imgs = [Image.fromarray(_to_uint8(frames[i], vmin, vmax)) for i in range(frames.shape[0])]
    if not imgs:
        return
    duration = int(1000 / max(fps, 1))
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)

def _save_combo_gif(frames_map: dict[str, np.ndarray],
                    out_path: str,
                    cmap,
                    norm,
                    fps: int,
                    input_mask: np.ndarray | None = None,
                    title: str | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    labels = list(frames_map.keys())
    frames_list = [ensure_thw(frames_map[k]) for k in labels]
    n = min(f.shape[0] for f in frames_list) if frames_list else 0
    if n <= 0:
        return
    mask_points = None
    if input_mask is not None:
        mask_points = np.argwhere(input_mask.astype(bool))

    imgs = []
    panel_width = 2.15
    fig_height = 4.95
    fig, axes = plt.subplots(1, len(labels), figsize=(panel_width * len(labels), fig_height), dpi=300)
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.008, right=0.998, wspace=0.002)
    if len(labels) == 1:
        axes = [axes]

    artists = []
    mask_bool = input_mask.astype(bool) if input_mask is not None else None
    masked_labels = {"input", "gauge", "radarmasked"}
    for ax, label, frames in zip(axes, labels, frames_list):
        if label.lower() in masked_labels and mask_bool is not None:
            bg = ax.imshow(np.zeros_like(frames[0]), cmap="gray", vmin=0.0, vmax=1.0)
            scatter = None
            if mask_points is not None and mask_points.size > 0:
                vals = frames[0][mask_bool]
                scatter = ax.scatter(
                    mask_points[:, 1],
                    mask_points[:, 0],
                    c=vals,
                    cmap=cmap,
                    norm=norm,
                    s=38,
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=5,
                )
            artists.append(("masked", bg, scatter, frames))
        else:
            im = ax.imshow(frames[0], cmap=cmap, norm=norm)
            artists.append(("image", im, None, frames))
        ax.set_title(label, fontsize=15, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", "box")
        for s in ax.spines.values():
            s.set_visible(False)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        ticks=_PAPER_BOUNDS,
    )
    tick_labels = [f"{b:g}" for b in _PAPER_BOUNDS[:-1]] + [""]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Rainfall (mm/h)", fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    suptitle = fig.suptitle("", fontsize=16, fontweight="bold") if title else None
    for t in range(n):
        for kind, artist, scatter, frames in artists:
            if kind == "masked":
                if scatter is not None and mask_bool is not None:
                    scatter.set_array(frames[t][mask_bool].ravel())
            else:
                artist.set_data(frames[t])
        if suptitle is not None:
            suptitle.set_text(f"{title} | Frame {t + 1}/{n}")
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        imgs.append(Image.fromarray(buf))
    plt.close(fig)

    duration = int(1000 / max(fps, 1))
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)


def _list_event_keys(path: str) -> list[str]:
    try:
        import zarr
    except Exception:
        return []

    z = zarr.open(path, mode="r")
    keys = []
    if hasattr(z, "group_keys"):
        keys = list(z.group_keys())
    if not keys and hasattr(z, "keys"):
        keys = [k for k in z.keys() if isinstance(k, str)]
    if not keys:
        return []

    def key_num(k: str) -> tuple[int, str]:
        m = re.search(r"event[_-]?(\\d+)", k, re.IGNORECASE)
        return (int(m.group(1)) if m else 10**9, k)

    return [k for k in sorted(keys, key=key_num)]


def _load_event_array(path: str, event_key: str) -> np.ndarray:
    try:
        import zarr
    except Exception:
        raise FileNotFoundError(f"zarr not available to load event {event_key} from {path}")

    z = zarr.open(path, mode="r")
    if event_key in z:
        return np.asarray(z[event_key])
    raise FileNotFoundError(f"Missing event {event_key} in {path}")


def _load_prepared_event(path: str,
                         event_key: str,
                         crop_size: int,
                         divide_by_3: bool,
                         cache: dict[tuple[str, str, int, bool], np.ndarray]) -> np.ndarray:
    cache_key = (path, event_key, crop_size, divide_by_3)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    arr = _load_event_array(path, event_key)
    arr = transform_mmhr(arr, divide_by_3=divide_by_3)
    arr = crop_center(arr, crop_size)
    cache[cache_key] = arr
    return arr


def run_exp2(preds: dict[str, str | np.ndarray],
             observation: str | np.ndarray,
             nimrod_path: str | np.ndarray | None,
             mask_train: np.ndarray,
             out_dir: str,
             crop_size: int,
             frames: int | None,
             vmin: float,
             vmax: float,
             gif_fps: int,
             divide_by_3: bool = True,
             mode: str = "radar") -> None:
    input_label = "Gauge" if mode == "gauge" else "Input"
    obs_label = "Radar" if mode == "gauge" else "Obs"
    use_input_mask = True
    if isinstance(observation, str):
        event_keys = _list_event_keys(observation)
        if not event_keys:
            raise FileNotFoundError(f"No event groups found in {observation}")
        event_keys = event_keys[:20]
        range_lines = []
        cmap, norm, _ = _build_paper_cmap()
        max_frames = 30
        event_cache: dict[tuple[str, str, int, bool], np.ndarray] = {}
        for idx, event_key in enumerate(event_keys, start=1):
            obs_ev = _load_prepared_event(
                observation, event_key, crop_size, divide_by_3, event_cache
            )[:max_frames]
            input_ev = obs_ev
            obs_ev_full = obs_ev
            if mode == "gauge" and isinstance(nimrod_path, str):
                nimrod_ev = _load_prepared_event(
                    nimrod_path, event_key, crop_size, divide_by_3, event_cache
                )
                nimrod_ev, input_ev = align_length(nimrod_ev, input_ev)
                nimrod_ev = nimrod_ev[:max_frames]
                input_ev = input_ev[:max_frames]
                obs_ev_full = nimrod_ev
            else:
                input_ev = input_ev[:max_frames]
                obs_ev_full = obs_ev_full[:max_frames]

            preds_ev: dict[str, np.ndarray] = {}
            for name, pred_src in preds.items():
                if isinstance(pred_src, str):
                    pred_ev = _load_prepared_event(
                        pred_src, event_key, crop_size, divide_by_3, event_cache
                    )
                else:
                    pred_ev = pred_src
                pred_ev, ref_aligned = align_length(pred_ev, obs_ev_full)
                pred_ev = pred_ev[:max_frames]
                preds_ev[name] = pred_ev
                obs_ev_full = ref_aligned[:pred_ev.shape[0]]
                input_ev = input_ev[:obs_ev_full.shape[0]]

            total_frames = min(input_ev.shape[0], obs_ev_full.shape[0])
            for pred_ev in preds_ev.values():
                total_frames = min(total_frames, pred_ev.shape[0])
            input_ev = input_ev[:total_frames]
            obs_ev_full = obs_ev_full[:total_frames]
            for name in list(preds_ev.keys()):
                preds_ev[name] = preds_ev[name][:total_frames]

            range_lines.append(
                f"{event_key}: frames 1-{total_frames} (count={total_frames})"
            )
            combo_frames = {
                input_label: input_ev,
                obs_label: obs_ev_full,
            }
            for name, pred_ev in preds_ev.items():
                combo_frames[name] = pred_ev

            title = f"{event_key} | total frames {total_frames}"
            out_path = os.path.join(out_dir, f"comparison_{event_key}.gif")
            _save_combo_gif(combo_frames, out_path, cmap, norm, gif_fps,
                            input_mask=mask_train if use_input_mask else None, title=title)

        save_text(os.path.join(out_dir, "event_ranges.txt"), range_lines)
        return

    obs_arr = np.asarray(observation)
    obs_arr = transform_mmhr(obs_arr, divide_by_3=divide_by_3)
    obs_arr = crop_center(obs_arr, crop_size)
    input_arr = obs_arr
    obs_arr_full = obs_arr
    if mode == "gauge" and nimrod_path is not None:
        nimrod_arr = np.asarray(nimrod_path)
        nimrod_arr = transform_mmhr(nimrod_arr, divide_by_3=divide_by_3)
        nimrod_arr, input_arr = align_length(nimrod_arr, input_arr)
        nimrod_arr = crop_center(nimrod_arr, crop_size)
        obs_arr_full = nimrod_arr
    else:
        nimrod_arr = None

    preds_aligned: dict[str, np.ndarray] = {}
    for name, pred in preds.items():
        pred_arr = np.asarray(pred)
        pred_arr = transform_mmhr(pred_arr, divide_by_3=divide_by_3)
        pred_arr, ref_aligned = align_length(pred_arr, obs_arr_full)
        pred_arr = crop_center(pred_arr, crop_size)
        preds_aligned[name] = pred_arr
        obs_arr_full = ref_aligned
        input_arr = input_arr[:obs_arr_full.shape[0]]

    total_frames = min(input_arr.shape[0], obs_arr_full.shape[0])
    for pred_arr in preds_aligned.values():
        total_frames = min(total_frames, pred_arr.shape[0])
    input_arr = input_arr[:total_frames]
    obs_arr_full = obs_arr_full[:total_frames]
    for name in list(preds_aligned.keys()):
        preds_aligned[name] = preds_aligned[name][:total_frames]

    combo_frames = {
        input_label: input_arr,
        obs_label: obs_arr_full,
    }
    for name, pred_arr in preds_aligned.items():
        combo_frames[name] = pred_arr
    cmap, norm, _ = _build_paper_cmap()
    title = f"Event 01 | total frames {total_frames}"
    _save_combo_gif(combo_frames, os.path.join(out_dir, "comparison_event_01.gif"),
                    cmap, norm, gif_fps,
                    input_mask=mask_train if use_input_mask else None, title=title)


def _build_paper_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    def hex_to_rgb01(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lerp(a, b, t):
        return tuple((1 - t) * ai + t * bi for ai, bi in zip(a, b))

    fine_bounds = []
    for i in range(len(_PAPER_BOUNDS) - 1):
        seg = np.linspace(_PAPER_BOUNDS[i], _PAPER_BOUNDS[i + 1], _PAPER_SUB + 1, endpoint=False)
        if i == 0:
            fine_bounds.extend(seg.tolist())
        else:
            fine_bounds.extend(seg[1:].tolist())
    fine_bounds.append(_PAPER_BOUNDS[-1])
    fine_bounds = np.asarray(fine_bounds, float)

    rgb_base = [hex_to_rgb01(h) for h in _PAPER_COLORS]
    grad_colors = []
    for i in range(len(rgb_base) - 1):
        c0, c1 = rgb_base[i], rgb_base[i + 1]
        for k in range(_PAPER_SUB):
            t = k / float(_PAPER_SUB - 1)
            grad_colors.append(lerp(c0, c1, t))
    grad_colors.append(rgb_base[-1])
    while len(grad_colors) < len(fine_bounds) - 1:
        grad_colors.append(rgb_base[-1])

    cmap = ListedColormap(grad_colors, name=f"seg{_PAPER_SUB}_smooth")
    norm = BoundaryNorm(fine_bounds, cmap.N, clip=True)
    return cmap, norm, fine_bounds


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[top:top + size, left:left + size]


def _load_event_images(folders: dict[str, str],
                       method_order: Iterable[str],
                       event_id: int,
                       select_idx: Iterable[int],
                       crop_size: int) -> tuple[np.ndarray, list[str]]:
    rain_str = f"rain{event_id}"
    sample_folder = os.path.join(folders.get("Gauge", ""), rain_str)
    if not os.path.isdir(sample_folder):
        raise FileNotFoundError(f"Missing sample folder: {sample_folder}")
    all_pngs = sorted(
        [f for f in os.listdir(sample_folder) if f.lower().endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    images = []
    labels = []
    for method in method_order:
        folder = folders.get(method, "")
        labels.append(method)
        frames = []
        if not folder:
            frames = [np.zeros((crop_size, crop_size), dtype=np.float32) for _ in select_idx]
            images.append(frames)
            continue
        for idx in select_idx:
            if idx >= len(all_pngs):
                frames.append(np.zeros((crop_size, crop_size), dtype=np.float32))
                continue
            path = os.path.join(folder, rain_str, all_pngs[idx])
            if not os.path.isfile(path):
                frames.append(np.zeros((crop_size, crop_size), dtype=np.float32))
                continue
            arr = np.array(Image.open(path).convert("F")).astype(np.float32) / 3.0
            arr = 10 ** (arr * 0.0625) * 0.036
            arr = _center_crop(arr, crop_size)
            frames.append(arr)
        images.append(frames)
    return np.asarray(images), labels


def _draw_block(ax_grid, images, method_order, select_idx, mask, mask_points, cmap, norm):
    last_im = None
    for t in range(images.shape[1]):
        for m in range(images.shape[0]):
            ax = ax_grid[t, m]
            label = method_order[m]
            if label in {"Input", "Gauge", "RadarMasked"}:
                ax.imshow(np.zeros_like(images[m, t]), cmap="gray", vmin=0.0, vmax=1.0)
                vals = images[m, t][mask == 1]
                ax.scatter(mask_points[:, 1], mask_points[:, 0],
                           c=vals, cmap=cmap, norm=norm,
                           s=38, edgecolors="black", linewidths=0.7, zorder=5)
                last_im = ax.images[-1] if ax.images else None
            else:
                last_im = ax.imshow(images[m, t], cmap=cmap, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            if t == 0:
                ax.set_title(method_order[m], fontsize=13)

            if m == 0:
                time_label = f"{t * 5} min"
                ax.text(-0.12, 0.5, time_label, transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, rotation=90)
    return last_im


def _event_key(event_id: int) -> str:
    return f"event_{int(event_id):02d}"


def run_exp2_paper(folders: dict[str, str],
                   method_order: Iterable[str],
                   events: Iterable[dict[str, object]],
                   mask_path: str,
                   crop_size: int,
                   out_dir: str,
                   output_pdf: str,
                   crop_pdf: bool = False,
                   crop_output: str = "cropped_stitched.pdf",
                   crop_y_ranges: tuple[tuple[float, float], ...] = ((0.019, 0.5), (0.58, 1.0)),
                   crop_zoom: float = 3.0,
                   crop_margin_left: float = 0.0,
                   crop_margin_right: float = 0.0) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.gridspec import GridSpec

    ensure_dir(out_dir)
    cmap, norm, _ = _build_paper_cmap()

    mask = load_mask(mask_path)
    mask = _center_crop(mask, crop_size)
    mask_points = np.argwhere(mask == 1)

    method_order = list(method_order)
    events = list(events)
    ncols = len(method_order)
    nrows_each = len(events[0]["select_idx"])
    total_rows = (nrows_each + 1) * len(events)

    fig = plt.figure(figsize=(2.1 * ncols, 1.9 * total_rows))
    gs = GridSpec(nrows=total_rows, ncols=ncols, figure=fig,
                  top=0.93, bottom=0.06, wspace=0.05, hspace=0.02)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.20, 0.88, 0.60, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", ticks=_PAPER_BOUNDS)
    tick_labels = [f"{b:g}" for b in _PAPER_BOUNDS[:-1]] + [""]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Rainfall (mm/h)", fontsize=13, labelpad=3)
    cbar.ax.tick_params(labelsize=11)

    row_cursor = 0
    event_cache: dict[tuple[str, str, int, bool], np.ndarray] = {}
    for event in events:
        ax_title = fig.add_subplot(gs[row_cursor, :])
        ax_title.axis("off")
        ax_title.text(-0.015, 0.2, event["title"],
                      fontsize=14, fontweight="bold", ha="left", va="center",
                      transform=ax_title.transAxes)
        row_cursor += 1

        ax_grid = np.empty((nrows_each, ncols), dtype=object)
        for r in range(nrows_each):
            for c in range(ncols):
                ax_grid[r, c] = fig.add_subplot(gs[row_cursor + r, c])

        imgs, labels = _load_event_images(folders, method_order, event["event_id"],
                                          event["select_idx"], crop_size)
        _ = _draw_block(ax_grid, imgs, labels, event["select_idx"], mask, mask_points, cmap, norm)
        row_cursor += nrows_each

    fig_path = os.path.join(out_dir, output_pdf)
    plt.tight_layout(rect=[0, 0, 1, 0.7])
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    if crop_pdf:
        _crop_pdf_panels(fig_path, os.path.join(out_dir, crop_output),
                         y_ranges=crop_y_ranges, zoom=crop_zoom,
                         margin_left=crop_margin_left, margin_right=crop_margin_right)


def run_exp2_paper_zarr(observation_path: str,
                        nimrod_path: str | None,
                        methods: dict[str, str],
                        events: Iterable[dict[str, object]],
                        mask_path: str,
                        crop_size: int,
                        out_dir: str,
                        output_pdf: str,
                        mode: str = "radar",
                        method_order: Iterable[str] | None = None,
                        crop_pdf: bool = False,
                        crop_output: str = "cropped_stitched.pdf",
                        crop_y_ranges: tuple[tuple[float, float], ...] = ((0.019, 0.5), (0.58, 1.0)),
                        crop_zoom: float = 3.0,
                        crop_margin_left: float = 0.0,
                        crop_margin_right: float = 0.0) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.gridspec import GridSpec

    ensure_dir(out_dir)
    cmap, norm, _ = _build_paper_cmap()

    mask = load_mask(mask_path)
    mask = _center_crop(mask, crop_size)
    mask_points = np.argwhere(mask == 1)

    if method_order is None:
        method_order = tuple(list(methods.keys()))
    method_order = list(method_order)
    prefix_labels = ["Gauge", "Radar"] if mode == "gauge" else ["Input", "Obs"]
    method_order = prefix_labels + [m for m in method_order if m not in {"Input", "Obs", "Gauge", "Radar", "Nimrod", "RadarMasked"}]
    events = list(events)
    ncols = len(method_order)
    nrows_each = len(events[0]["select_idx"])
    total_rows = (nrows_each + 1) * len(events)

    fig = plt.figure(figsize=(2.4 * ncols, 1.9 * total_rows))
    gs = GridSpec(nrows=total_rows, ncols=ncols, figure=fig,
                  top=0.93, bottom=0.06, wspace=0.04, hspace=0.02)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.20, 0.88, 0.60, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", ticks=_PAPER_BOUNDS)
    tick_labels = [f"{b:g}" for b in _PAPER_BOUNDS[:-1]] + [""]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Rainfall (mm/h)", fontsize=13, labelpad=3)
    cbar.ax.tick_params(labelsize=11)

    row_cursor = 0
    event_cache: dict[tuple[str, str, int, bool], np.ndarray] = {}
    for event in events:
        ax_title = fig.add_subplot(gs[row_cursor, :])
        ax_title.axis("off")
        ax_title.text(-0.015, 0.2, event["title"],
                      fontsize=14, fontweight="bold", ha="left", va="center",
                      transform=ax_title.transAxes)
        row_cursor += 1

        ax_grid = np.empty((nrows_each, ncols), dtype=object)
        for r in range(nrows_each):
            for c in range(ncols):
                ax_grid[r, c] = fig.add_subplot(gs[row_cursor + r, c])

        event_key = _event_key(int(event["event_id"]))
        select_idx = list(event["select_idx"])
        obs_ev = _load_prepared_event(
            observation_path, event_key, crop_size, True, event_cache
        )
        nimrod_ev = None
        if nimrod_path:
            nimrod_ev = _load_prepared_event(
                nimrod_path, event_key, crop_size, True, event_cache
            )

        images = []
        labels = []
        for method in method_order:
            labels.append(method)
            frames = []
            if method in {"Input", "Gauge", "RadarMasked"}:
                source = obs_ev
            elif method in {"Obs"}:
                source = nimrod_ev if (mode == "gauge" and nimrod_ev is not None) else obs_ev
            elif method in {"Radar", "Nimrod"}:
                source = nimrod_ev if nimrod_ev is not None else obs_ev
            else:
                method_path = _lookup_method_path(methods, method)
                if not method_path:
                    source = None
                else:
                    source = _load_prepared_event(
                        method_path, event_key, crop_size, True, event_cache
                    )

            for idx in select_idx:
                if source is None or idx >= source.shape[0]:
                    frames.append(np.zeros((crop_size, crop_size), dtype=np.float32))
                else:
                    frames.append(source[idx])
            images.append(frames)

        imgs = np.asarray(images)
        _ = _draw_block(ax_grid, imgs, labels, select_idx, mask, mask_points, cmap, norm)
        row_cursor += nrows_each

    fig_path = os.path.join(out_dir, output_pdf)
    plt.tight_layout(rect=[0, 0, 1, 0.7])
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    if crop_pdf:
        _crop_pdf_panels(fig_path, os.path.join(out_dir, crop_output),
                         y_ranges=crop_y_ranges, zoom=crop_zoom,
                         margin_left=crop_margin_left, margin_right=crop_margin_right)


def _crop_pdf_panels(pdf_path: str,
                     output_path: str,
                     y_ranges: tuple[tuple[float, float], ...],
                     zoom: float,
                     margin_left: float,
                     margin_right: float) -> None:
    import fitz

    def crop_regions_from_pdf_rel(page, rects_rel, zoom):
        (x0, y0, x1, y1) = page.rect
        rects_pts = []
        for (rx0, ry0, rx1, ry1) in rects_rel:
            rects_pts.append((
                x0 + rx0 * (x1 - x0),
                y0 + ry0 * (y1 - y0),
                x0 + rx1 * (x1 - x0),
                y0 + ry1 * (y1 - y0),
            ))
        images = []
        mat = fitz.Matrix(zoom, zoom)
        for (px0, py0, px1, py1) in rects_pts:
            clip = fitz.Rect(px0, py0, px1, py1)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        return images

    def stitch_images(imgs, direction="v", gap=8, bg=(255, 255, 255)):
        if direction == "h":
            h = max(im.height for im in imgs)
            w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
            canvas = Image.new("RGB", (w, h), bg)
            x = 0
            for im in imgs:
                y = (h - im.height) // 2
                canvas.paste(im, (x, y))
                x += im.width + gap
        else:
            w = max(im.width for im in imgs)
            h = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
            canvas = Image.new("RGB", (w, h), bg)
            y = 0
            for im in imgs:
                x = (w - im.width) // 2
                canvas.paste(im, (x, y))
                y += im.height + gap
        return canvas

    doc = fitz.open(pdf_path)
    page = doc[0]
    rects_rel = []
    for (y0, y1) in y_ranges:
        y0c = max(0.0, min(1.0, y0))
        y1c = max(0.0, min(1.0, y1))
        if y1c <= y0c:
            continue
        rects_rel.append((margin_left, y0c, 1 - margin_right, y1c))
    parts = crop_regions_from_pdf_rel(page, rects_rel, zoom=zoom)
    doc.close()
    if not parts:
        return
    stitched = stitch_images(parts, direction="v", gap=8, bg=(255, 255, 255))
    stitched.save(output_path)
