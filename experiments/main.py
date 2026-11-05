from __future__ import annotations

import os

from .config import build_config, get_mode_config
from .exp1 import run_exp1
from .exp2 import run_exp2, run_exp2_paper_zarr
from .exp3 import run_exp3
from .io import ensure_dir, load_mask, load_zarr_array, save_config_snapshot, save_json, save_text


def _crop_mask(mask, size):
    h, w = mask.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return mask[top:top + size, left:left + size]


def _require_path(path: str, label: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{label} is empty in experiments/config.py")
    return path.strip()


def _check_exists(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _debug_print_paths(cfg, observation_path: str, nimrod_path: str, method_paths, mask_train_path: str, mask_test_path: str) -> None:
    print("[experiments] Config summary")
    print(f"  mode: {cfg.mode}")
    print(f"  experiment_name: {cfg.experiment_name}")
    print(f"  observation_path: {observation_path}")
    print(f"  nimrod_path: {nimrod_path}")
    print(f"  mask_train_path: {mask_train_path}")
    print(f"  mask_test_path: {mask_test_path}")
    print("  methods:")
    for name, path in method_paths.items():
        print(f"    {name}: {path}")


def _format_table(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row):
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in rows)
    return lines


def _format_metric_value(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def _build_exp2_event_pdf_name(template, event, event_idx):
    event_id = int(event["event_id"])
    return template.format(event_id=event_id, event_idx=event_idx)


def _build_exp1_tables(metrics):
    if not metrics:
        return ["No metrics generated."]

    methods = list(metrics.keys())
    scalar_headers = ["Model", "MAE", "RMSE", "PSNR", "PSS", "SSIM", "DTSSIM_L1", "DTSSIM_L2", "NSE"]
    scalar_rows = []
    for method in methods:
        vals = metrics[method]
        scalar_rows.append([
            method,
            _format_metric_value(vals.get("MAE", float("nan"))),
            _format_metric_value(vals.get("RMSE", float("nan"))),
            _format_metric_value(vals.get("PSNR", float("nan"))),
            _format_metric_value(vals.get("PSS", float("nan"))),
            _format_metric_value(vals.get("SSIM", float("nan"))),
            _format_metric_value(vals.get("DTSSIM_L1", float("nan"))),
            _format_metric_value(vals.get("DTSSIM_L2", float("nan"))),
            _format_metric_value(vals.get("NSE", float("nan"))),
        ])

    cat_thresholds = sorted(
        [key for key in next(iter(metrics.values())).keys() if key.startswith("CAT_")],
        key=lambda x: float(x.split("_", 1)[1]),
    )
    cat_headers = ["Model"]
    for cat_key in cat_thresholds:
        thr = cat_key.split("_", 1)[1]
        cat_headers.extend([f"POD@{thr}", f"FAR@{thr}", f"CSI@{thr}", f"HSS@{thr}"])
    cat_rows = []
    for method in methods:
        vals = metrics[method]
        row = [method]
        for cat_key in cat_thresholds:
            cat_vals = vals.get(cat_key, {})
            row.extend([
                _format_metric_value(cat_vals.get("POD", float("nan"))),
                _format_metric_value(cat_vals.get("FAR", float("nan"))),
                _format_metric_value(cat_vals.get("CSI", float("nan"))),
                _format_metric_value(cat_vals.get("HSS", float("nan"))),
            ])
        cat_rows.append(row)

    fss_keys = sorted(
        [key for key in next(iter(metrics.values())).keys() if key.startswith("FSS_")],
        key=lambda x: (float(x.split("_")[1]), int(x.split("_")[2].split("x")[0])),
    )
    fss_lines = []
    if fss_keys:
        fss_headers = ["Model"] + [key.replace("FSS_", "") for key in fss_keys]
        fss_rows = []
        for method in methods:
            vals = metrics[method]
            fss_rows.append([method] + [_format_metric_value(vals.get(key, float("nan"))) for key in fss_keys])
        fss_lines = _format_table(fss_headers, fss_rows)

    lines = ["[Scalar Metrics]"]
    lines.extend(_format_table(scalar_headers, scalar_rows))
    lines.append("")
    lines.append("[Categorical Metrics]")
    lines.extend(_format_table(cat_headers, cat_rows))
    if fss_lines:
        lines.append("")
        lines.append("[FSS Metrics]")
        lines.extend(fss_lines)
    return lines


def main() -> None:
    cfg = build_config()
    mode_cfg = get_mode_config(cfg)
    observation_path = _require_path(mode_cfg.observation_path, f"{cfg.mode}.observation_path")
    nimrod_path = _require_path(getattr(mode_cfg, "nimrod_path", None), f"{cfg.mode}.nimrod_path")
    mask_train_path = _require_path(mode_cfg.mask_train_path, f"{cfg.mode}.mask_train_path")
    mask_test_path = _require_path(mode_cfg.mask_test_path, f"{cfg.mode}.mask_test_path")
    method_paths = {
        name: _require_path(path, f"{cfg.mode}.methods[{name!r}]")
        for name, path in mode_cfg.methods.items()
    }
    _debug_print_paths(cfg, observation_path, nimrod_path, method_paths, mask_train_path, mask_test_path)

    observation_path = _check_exists(observation_path, f"{cfg.mode}.observation_path")
    nimrod_path = _check_exists(nimrod_path, f"{cfg.mode}.nimrod_path")
    mask_train_path = _check_exists(mask_train_path, f"{cfg.mode}.mask_train_path")
    mask_test_path = _check_exists(mask_test_path, f"{cfg.mode}.mask_test_path")
    method_paths = {
        name: _check_exists(path, f"{cfg.mode}.methods[{name!r}]")
        for name, path in method_paths.items()
    }

    results_root = os.path.join(cfg.save_dir, cfg.experiment_name)
    ensure_dir(results_root)
    save_config_snapshot(os.path.join(results_root, "config.json"), cfg)

    print(f"[experiments] Loading observation events from: {observation_path}")
    observation_events = load_zarr_array(
        observation_path,
        return_events=True,
        crop_size=cfg.crop_size,
    )

    preds = {}
    for name, path in method_paths.items():
        print(f"[experiments] Loading prediction for {name}: {path}")
        try:
            preds[name] = load_zarr_array(
                path,
                return_events=True,
                crop_size=cfg.crop_size,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load prediction for method {name!r} from path: {path}"
            ) from exc

    mask_train = load_mask(mask_train_path)
    mask_test = load_mask(mask_test_path)
    mask_train = _crop_mask(mask_train, cfg.crop_size)
    mask_test = _crop_mask(mask_test, cfg.crop_size)

    if cfg.run_exp1:
        exp1_dir = os.path.join(results_root, "exp1")
        ensure_dir(exp1_dir)
        mask_for_mode = mask_train if cfg.mode == "radar" else mask_test
        print(f"[exp1] Output directory: {exp1_dir}")
        metrics = run_exp1(
            preds,
            observation_events,
            mask_for_mode,
            cfg.mode,
            cfg.crop_size,
            use_pool8=cfg.exp1_pool8,
            divide_by_3=True,
            progress=True,
        )
        save_json(os.path.join(exp1_dir, "metrics.json"), metrics)
        lines = []
        if (
            metrics
            and isinstance(next(iter(metrics.values())), dict)
            and isinstance(next(iter(next(iter(metrics.values())).values())), dict)
        ):
            # event -> method -> metrics
            for event_key, methods in metrics.items():
                lines.append(f"[{event_key}]")
                for name, vals in methods.items():
                    lines.append(f"{name}:")
                    for k, v in vals.items():
                        if isinstance(v, dict):
                            lines.append(f"  {k}:")
                            for sub_k, sub_v in v.items():
                                lines.append(f"    {sub_k}: {sub_v:.6f}")
                        else:
                            lines.append(f"  {k}: {v:.6f}")
                    lines.append("")
                lines.append("")
        else:
            lines = _build_exp1_tables(metrics)
        save_text(os.path.join(exp1_dir, "metrics.txt"), lines)
        print(f"[exp1] Saved metrics to {exp1_dir}")

    if cfg.run_exp2_gif:
        exp2_gif_dir = os.path.join(results_root, "exp2_gif")
        ensure_dir(exp2_gif_dir)
        run_exp2(
            preds=method_paths,
            observation=observation_path,
            nimrod_path=nimrod_path,
            mask_train=mask_train,
            out_dir=exp2_gif_dir,
            crop_size=cfg.crop_size,
            frames=None,
            vmin=cfg.visualization_vmin,
            vmax=cfg.visualization_vmax,
            gif_fps=cfg.gif_fps,
            divide_by_3=True,
            mode=cfg.mode,
        )

    if cfg.run_exp2_pdf:
        exp2_pdf_dir = os.path.join(results_root, "exp2_pdf")
        ensure_dir(exp2_pdf_dir)
        mask_path = cfg.exp2_paper_mask_path or mode_cfg.mask_train_path
        method_order = (
            cfg.exp2_paper_radar_method_order
            if cfg.mode == "radar"
            else cfg.exp2_paper_gauge_method_order
        )
        if not cfg.exp2_paper_events:
            raise ValueError("exp2_paper_events is empty in experiments/config.py")

        if cfg.exp2_paper_one_event_per_pdf:
            for event_idx, event in enumerate(cfg.exp2_paper_events, start=1):
                output_pdf = _build_exp2_event_pdf_name(
                    cfg.exp2_paper_output_template,
                    event,
                    event_idx,
                )
                run_exp2_paper_zarr(
                    observation_path=observation_path,
                    nimrod_path=nimrod_path,
                    methods=method_paths,
                    events=(event,),
                    mask_path=mask_path,
                    crop_size=cfg.crop_size,
                    out_dir=exp2_pdf_dir,
                    output_pdf=output_pdf,
                    mode=cfg.mode,
                    method_order=method_order,
                    crop_pdf=False,
                    crop_output=cfg.exp2_paper_crop_output,
                    crop_y_ranges=((0.019, 0.5), (0.58, 1.0)),
                    crop_zoom=3.0,
                    crop_margin_left=0.0,
                    crop_margin_right=0.0,
                )
        else:
            run_exp2_paper_zarr(
                observation_path=observation_path,
                nimrod_path=nimrod_path,
                methods=method_paths,
                events=cfg.exp2_paper_events,
                mask_path=mask_path,
                crop_size=cfg.crop_size,
                out_dir=exp2_pdf_dir,
                output_pdf=cfg.exp2_paper_output_pdf,
                mode=cfg.mode,
                method_order=method_order,
                crop_pdf=False,
                crop_output=cfg.exp2_paper_crop_output,
                crop_y_ranges=((0.019, 0.5), (0.58, 1.0)),
                crop_zoom=3.0,
                crop_margin_left=0.0,
                crop_margin_right=0.0,
            )

    if cfg.run_exp3:
        exp3_dir = os.path.join(results_root, "exp3")
        ensure_dir(exp3_dir)
        mask_for_mode = mask_train if cfg.mode == "radar" else mask_test
        metrics = run_exp3(preds, observation_events, mask_for_mode, cfg.mode, cfg.crop_size, exp3_dir)
        save_json(os.path.join(exp3_dir, "metrics.json"), metrics)
        lines = [f"{k}: {v:.6f}" for k, v in metrics.items()]
        save_text(os.path.join(exp3_dir, "metrics.txt"), lines)


if __name__ == "__main__":
    main()
