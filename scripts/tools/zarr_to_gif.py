from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ZARR_PATH = Path(
    "datasets/infer/ked_gauge.zarr"
)

# Paper-style rainfall palette.
PAPER_BOUNDS = np.asarray([0, 0.5, 1, 2, 4, 8, 16, 200], dtype=np.float32)
PAPER_COLORS = [
    "#000000",
    "#46327e",
    "#277f8e",
    "#4ac16d",
    "#a0da39",
    "#fde725",
    "#ffffff",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read event arrays from a Zarr store and save colored GIFs."
    )
    parser.add_argument(
        "zarr_path",
        nargs="?",
        type=Path,
        default=DEFAULT_ZARR_PATH,
        help=f"Input Zarr path. Default: {DEFAULT_ZARR_PATH}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder. Default: <zarr_parent>/<zarr_stem>_gifs.",
    )
    parser.add_argument(
        "--events",
        nargs="*",
        default=None,
        help="Events to export, for example: 1 2 event_03. Default: all events.",
    )
    parser.add_argument("--fps", type=int, default=5, help="GIF frame rate.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames per event.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Optional center crop size before saving.",
    )
    parser.add_argument(
        "--no-mmhr-transform",
        action="store_true",
        help="Skip the raw-value to mm/h transform.",
    )
    return parser.parse_args()


def event_key(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\d+", value):
        return f"event_{int(value):02d}"
    return value


def sort_event_key(key: str) -> tuple[int, str]:
    match = re.search(r"event[_-]?(\d+)", key, re.IGNORECASE)
    return (int(match.group(1)) if match else 10**9, key)


def list_events(zarr_path: Path) -> list[str]:
    import zarr

    root = zarr.open(str(zarr_path), mode="r")
    if hasattr(root, "group_keys"):
        keys = list(root.group_keys())
    else:
        keys = []
    if not keys and hasattr(root, "keys"):
        keys = list(root.keys())
    return sorted([key for key in keys if isinstance(key, str)], key=sort_event_key)


def load_event(zarr_path: Path, key: str) -> np.ndarray:
    import zarr

    root = zarr.open(str(zarr_path), mode="r")
    if key not in root:
        raise FileNotFoundError(f"Missing event {key} in {zarr_path}")
    return np.asarray(root[key], dtype=np.float32)


def ensure_thw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got shape {arr.shape}")
    return arr


def center_crop(frames: np.ndarray, size: int | None) -> np.ndarray:
    frames = ensure_thw(frames)
    if size is None:
        return frames
    _, height, width = frames.shape
    if size > min(height, width):
        raise ValueError(f"crop size {size} exceeds input {height}x{width}")
    top = (height - size) // 2
    left = (width - size) // 2
    return frames[:, top:top + size, left:left + size]


def transform_mmhr(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32) / 3.0
    frames = 10 ** (frames * 0.0625) * 0.036
    return np.clip(frames, 0.0, 200.0)


def hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.asarray(
        [int(color[i:i + 2], 16) for i in (0, 2, 4)],
        dtype=np.float32,
    )


def colorize_rainfall(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(np.asarray(frame, dtype=np.float32), PAPER_BOUNDS[0], PAPER_BOUNDS[-1])
    colors = np.stack([hex_to_rgb(color) for color in PAPER_COLORS], axis=0)
    segment = np.searchsorted(PAPER_BOUNDS, frame, side="right") - 1
    segment = np.clip(segment, 0, len(PAPER_COLORS) - 2)

    low = PAPER_BOUNDS[segment]
    high = PAPER_BOUNDS[segment + 1]
    t = (frame - low) / np.maximum(high - low, 1e-6)

    rgb = colors[segment] * (1.0 - t[..., None]) + colors[segment + 1] * t[..., None]
    return rgb.astype(np.uint8)


def save_gif(frames: np.ndarray, out_path: Path, fps: int) -> None:
    images = [Image.fromarray(colorize_rainfall(frame), mode="RGB") for frame in ensure_thw(frames)]
    if not images:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = int(1000 / max(fps, 1))
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.zarr_path.parent / f"{args.zarr_path.stem}_gifs"

    available_events = list_events(args.zarr_path)
    if not available_events:
        raise FileNotFoundError(f"No events found in {args.zarr_path}")

    selected_events = [event_key(value) for value in args.events] if args.events else available_events
    missing = sorted(set(selected_events) - set(available_events), key=sort_event_key)
    if missing:
        raise FileNotFoundError(f"Missing events: {', '.join(missing)}")

    for key in selected_events:
        frames = load_event(args.zarr_path, key)
        frames = center_crop(frames, args.crop_size)
        if not args.no_mmhr_transform:
            frames = transform_mmhr(frames)
        if args.max_frames is not None:
            frames = frames[:args.max_frames]

        out_path = out_dir / f"{key}.gif"
        save_gif(frames, out_path, fps=args.fps)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
