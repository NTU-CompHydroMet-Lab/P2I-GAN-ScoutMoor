from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_NAME_MAP = {
    # "c_ked": "ked_gauge.zarr",
    # "c_kre": "kre_gauge.zarr",
    # "c_ok": "ok_nimrod.zarr",
    # "c_rbf": "rbf_nimrod.zarr",
    # "sti_dk_val": "dk_mse_nimrod.zarr",
    # "sti_stdk_val": "stdk_nimrod.zarr",
    "c_ked_v2": "ked_gauge_v2.zarr",
    "c_kre_v2": "kre_gauge_v2.zarr",
}

DEFAULT_LABEL_MAP = {
    # "c_ked": "KED",
    # "c_kre": "KRE",
    # "c_ok": "OK",
    # "c_rbf": "RBF",
    # "sti_dk_val": "DK",
    # "sti_stdk_val": "STDK",
    "c_ked_v2": "KED",
    "c_kre_v2": "KRE",
}

DEFAULT_METHODS = ["c_ked_v2", "c_kre_v2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert datasets/result PNG folders into infer-compatible Zarr directories."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("datasets/result"),
        help="Root folder containing method subfolders such as c_ked/rain0/*.png.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("datasets/predict"),
        help="Root folder where output .zarr directories will be created.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional list of method folder names to convert. Default: all folders under src-root.",
    )
    parser.add_argument(
        "--name-map",
        nargs="*",
        default=None,
        help="Optional mappings like c_ked=ked_gauge.zarr c_ok=ok_nimrod.zarr.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directories.",
    )
    return parser.parse_args()


def parse_name_map(items: list[str] | None) -> dict[str, str]:
    mapping = dict(DEFAULT_NAME_MAP)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid mapping: {item}. Expected src=dst.")
        src, dst = item.split("=", 1)
        mapping[src] = dst
    return mapping


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return int(match.group(1)), path.stem
    return 10**9, path.stem


def read_png_stack(event_dir: Path) -> np.ndarray:
    pngs = sorted(event_dir.glob("*.png"), key=natural_key)
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {event_dir}")

    frames = []
    expected_hw = None
    for png_path in pngs:
        with Image.open(png_path) as img:
            arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"Unsupported PNG shape {arr.shape} in {png_path}")
        if expected_hw is None:
            expected_hw = arr.shape
        elif arr.shape != expected_hw:
            raise ValueError(
                f"Inconsistent frame size in {event_dir}: {arr.shape} != {expected_hw}"
            )
        frames.append(arr.astype(np.float32))

    data = np.stack(frames, axis=0)
    return data[:, None, :, :]


def event_name_from_dir(event_dir: Path) -> str:
    match = re.search(r"(\d+)$", event_dir.name)
    if not match:
        raise ValueError(f"Cannot infer event id from directory name: {event_dir.name}")
    event_id = int(match.group(1)) + 1
    return f"event_{event_id:02d}"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def write_zarr_v2_group(group_dir: Path, attrs: dict | None = None) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    write_json(group_dir / ".zgroup", {"zarr_format": 2})
    write_json(group_dir / ".zattrs", attrs or {})


def write_zarr_v2_array(array_dir: Path, data: np.ndarray) -> None:
    array_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        array_dir / ".zarray",
        {
            "chunks": list(data.shape),
            "compressor": None,
            "dtype": "<f4",
            "fill_value": 0.0,
            "filters": None,
            "order": "C",
            "shape": list(data.shape),
            "zarr_format": 2,
        },
    )
    (array_dir / "0.0.0.0").write_bytes(data.astype("<f4", copy=False).tobytes(order="C"))


def infer_domain(event_count: int) -> str:
    if event_count == 10:
        return "gauge"
    if event_count == 20:
        return "nimrod"
    return "unknown"


def convert_method(src_dir: Path, dst_dir: Path, overwrite: bool) -> None:
    event_dirs = sorted([p for p in src_dir.iterdir() if p.is_dir()], key=natural_key)
    if not event_dirs:
        raise FileNotFoundError(f"No event directories found in {src_dir}")

    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {dst_dir}")
        shutil.rmtree(dst_dir)

    method_name = DEFAULT_LABEL_MAP.get(src_dir.name, src_dir.name)
    write_zarr_v2_group(
        dst_dir,
        {
            "model_name": method_name,
            "source_root": str(src_dir),
            "source_format": "png_sequence",
            "event_count": len(event_dirs),
            "domain": infer_domain(len(event_dirs)),
            "output_scale": 255.0,
            "converted_by": "scripts/result_pngs_to_zarr.py",
        },
    )

    for event_dir in event_dirs:
        event_name = event_name_from_dir(event_dir)
        data = read_png_stack(event_dir)
        write_zarr_v2_array(dst_dir / event_name, data)


def main() -> None:
    args = parse_args()
    name_map = parse_name_map(args.name_map)

    if not args.src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {args.src_root}")

    methods = args.methods
    if methods is None:
        methods = DEFAULT_METHODS

    args.dst_root.mkdir(parents=True, exist_ok=True)

    for method in methods:
        src_dir = args.src_root / method
        if not src_dir.exists():
            raise FileNotFoundError(f"Method folder does not exist: {src_dir}")
        dst_name = name_map.get(method, f"{method}.zarr")
        if not dst_name.endswith(".zarr"):
            dst_name = f"{dst_name}.zarr"
        dst_dir = args.dst_root / dst_name
        print(f"[convert] {src_dir} -> {dst_dir}")
        convert_method(src_dir, dst_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
