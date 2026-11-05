from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


BASE_DIR = Path(".")
DEFAULT_NIMROD_ZARR = BASE_DIR / "datasets/nimrod/nimrod_test.zarr"
DEFAULT_MIDAS_ZARR = BASE_DIR / "datasets/midas/midas_test.zarr"
DEFAULT_TRAIN_MASK = BASE_DIR / "datasets/masks/gauge_mask_128_train.txt"
DEFAULT_TEST_MASK = BASE_DIR / "datasets/masks/gauge_mask_128_test.txt"
DEFAULT_OUTPUT = BASE_DIR / "experiments/results/event_rain_stats.csv"
NIMROD_CROP_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-event max and mean rainfall in mm/h for NIMROD and MIDAS."
    )
    parser.add_argument("--nimrod", type=Path, default=DEFAULT_NIMROD_ZARR)
    parser.add_argument("--midas", type=Path, default=DEFAULT_MIDAS_ZARR)
    parser.add_argument("--train-mask", type=Path, default=DEFAULT_TRAIN_MASK)
    parser.add_argument("--test-mask", type=Path, default=DEFAULT_TEST_MASK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def transform_mmhr(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64) / 3.0
    arr = 10 ** (arr * 0.0625) * 0.036
    return np.clip(arr, 0.0, 200.0)


def sort_event_key(key: str) -> tuple[int, str]:
    match = re.search(r"event[_-]?(\d+)", key, re.IGNORECASE)
    return (int(match.group(1)) if match else 10**9, key)


def list_events(zarr_path: Path) -> list[str]:
    import zarr

    root = zarr.open(str(zarr_path), mode="r")
    keys = list(root.group_keys()) if hasattr(root, "group_keys") else []
    if not keys and hasattr(root, "keys"):
        keys = list(root.keys())
    return sorted([key for key in keys if isinstance(key, str)], key=sort_event_key)


def open_event_array(zarr_path: Path, event_key: str):
    import zarr

    root = zarr.open(str(zarr_path), mode="r")
    if event_key not in root:
        raise FileNotFoundError(f"Missing {event_key} in {zarr_path}")
    return root[event_key]


def ensure_hw(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    if frame.ndim != 2:
        raise ValueError(f"Expected frame shape [H,W], got {frame.shape}")
    return frame


def ensure_thw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected event shape [T,H,W], got {arr.shape}")
    return arr


def center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    frame = ensure_hw(frame)
    height, width = frame.shape
    if size > min(height, width):
        raise ValueError(f"crop size {size} exceeds frame shape {frame.shape}")
    top = (height - size) // 2
    left = (width - size) // 2
    return frame[top:top + size, left:left + size]


def center_crop_slices(shape: tuple[int, ...], size: int) -> tuple[slice, slice]:
    height, width = shape[-2:]
    if size > min(height, width):
        raise ValueError(f"crop size {size} exceeds frame shape {(height, width)}")
    top = (height - size) // 2
    left = (width - size) // 2
    return slice(top, top + size), slice(left, left + size)


def read_frame(arr, t: int, crop_size: int | None = None) -> np.ndarray:
    if crop_size is None:
        return ensure_hw(arr[t])

    row_slice, col_slice = center_crop_slices(tuple(arr.shape), crop_size)
    if len(arr.shape) == 3:
        return ensure_hw(arr[t, row_slice, col_slice])
    if len(arr.shape) == 4 and arr.shape[1] == 1:
        return ensure_hw(arr[t, :, row_slice, col_slice])
    if len(arr.shape) == 4 and arr.shape[-1] == 1:
        return ensure_hw(arr[t, row_slice, col_slice, :])
    return center_crop(ensure_hw(arr[t]), crop_size)


def load_gauge_mask(train_mask_path: Path, test_mask_path: Path) -> np.ndarray:
    train_mask = np.loadtxt(train_mask_path) == 1
    test_mask = np.loadtxt(test_mask_path) == 1
    if train_mask.shape != test_mask.shape:
        raise ValueError(f"Mask shapes differ: {train_mask.shape} != {test_mask.shape}")
    mask = train_mask | test_mask
    point_count = int(mask.sum())
    if point_count != 99:
        raise ValueError(f"Expected 99 gauge points, got {point_count}")
    return mask


def event_stats(zarr_path: Path,
                event_key: str,
                mask: np.ndarray | None = None,
                crop_size: int | None = None,
                region: str = "full") -> dict[str, float | int | str]:
    arr = open_event_array(zarr_path, event_key)
    if len(arr.shape) < 3:
        raise ValueError(f"Expected event array with time dimension, got {arr.shape}")

    frame_count = int(arr.shape[0])

    # These zarr stores use event-sized chunks, so read each event once instead
    # of slicing frame-by-frame and repeatedly decompressing the same chunk.
    frames = ensure_thw(np.asarray(arr, dtype=np.float32))
    if crop_size is not None:
        row_slice, col_slice = center_crop_slices(frames.shape, crop_size)
        frames = frames[:, row_slice, col_slice]
    frames = transform_mmhr(frames)

    if mask is not None:
        if frames.shape[1:] != mask.shape:
            raise ValueError(f"{event_key}: frame shape {frames.shape[1:]} != mask shape {mask.shape}")
        values = frames[:, mask]
        point_count = int(mask.sum())
    else:
        values = frames
        point_count = int(np.prod(frames.shape[1:]))

    values = values[np.isfinite(values)]
    max_value = float(values.max()) if values.size else float("nan")
    mean_value = float(values.mean()) if values.size else float("nan")


    return {
        "event": event_key,
        "frames": frame_count,
        "region": region,
        "points": point_count,
        "max_mmhr": max_value,
        "mean_mmhr": mean_value,
    }


def collect_stats(dataset_name: str,
                  zarr_path: Path,
                  mask: np.ndarray | None = None,
                  crop_size: int | None = None,
                  region: str = "full") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event_key in list_events(zarr_path):
        stats = event_stats(
            zarr_path,
            event_key,
            mask=mask,
            crop_size=crop_size,
            region=region,
        )
        stats["dataset"] = dataset_name
        rows.append(stats)
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "event", "frames", "region", "points", "max_mmhr", "mean_mmhr"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_rows(rows: list[dict[str, object]]) -> None:
    print("dataset,event,frames,region,points,max_mmhr,mean_mmhr")
    for row in rows:
        print(
            f"{row['dataset']},{row['event']},{row['frames']},{row['region']},{row['points']},"
            f"{float(row['max_mmhr']):.6f},{float(row['mean_mmhr']):.6f}"
        )


def main() -> None:
    args = parse_args()
    gauge_mask = load_gauge_mask(args.train_mask, args.test_mask)

    rows = []
    rows.extend(
        collect_stats(
            "nimrod",
            args.nimrod,
            crop_size=NIMROD_CROP_SIZE,
            region="center_128x128",
        )
    )
    rows.extend(
        collect_stats(
            "midas",
            args.midas,
            mask=gauge_mask,
            region="train_plus_test_gauge_99pts",
        )
    )

    write_csv(rows, args.output)
    print_rows(rows)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
