from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass

import numpy as np


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


def crop_center(arr: np.ndarray, size: int) -> np.ndarray:
    arr = ensure_thw(arr)
    t, h, w = arr.shape
    if size > min(h, w):
        raise ValueError(f"crop size {size} exceeds input {h}x{w}")
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[:, top:top + size, left:left + size]


def load_mask(path: str) -> np.ndarray:
    mask = np.loadtxt(path)
    return mask.astype(bool)


def _crop_last2(arr: np.ndarray, size: int | None) -> np.ndarray:
    if size is None:
        return np.asarray(arr)
    arr = np.asarray(arr)
    if arr.ndim < 2:
        return arr
    h, w = arr.shape[-2:]
    if size > min(h, w):
        raise ValueError(f"crop size {size} exceeds input {h}x{w}")
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top:top + size, left:left + size]


def _open_zarr_xarray(path: str, key: str | None = None, crop_size: int | None = None) -> np.ndarray:
    import xarray as xr

    ds = xr.open_zarr(path)
    if hasattr(ds, "data_vars") and len(ds.data_vars) > 0:
        if key is None:
            key = list(ds.data_vars.keys())[0]
        arr = ds[key].values
    else:
        arr = ds.values
    return _crop_last2(arr, crop_size)


def _open_zarr_native(path: str, crop_size: int | None = None) -> np.ndarray:
    import zarr

    z = zarr.open(path, mode="r")
    if hasattr(z, "array_keys"):
        keys = list(z.array_keys())
        if keys:
            return _crop_last2(np.asarray(z[keys[0]]), crop_size)
    return _crop_last2(np.asarray(z), crop_size)


def _load_zarr_events_xarray(path: str, crop_size: int | None = None) -> dict[str, np.ndarray] | None:
    try:
        import xarray as xr
    except Exception:
        return None

    ds = xr.open_zarr(path)
    if hasattr(ds, "data_vars") and len(ds.data_vars) > 0:
        return {k: _crop_last2(np.asarray(ds[k].values), crop_size) for k in ds.data_vars.keys()}
    return None


def _load_zarr_events_native(path: str, crop_size: int | None = None) -> dict[str, np.ndarray] | None:
    try:
        import zarr
    except Exception:
        return None

    # Some converted stores are laid out as a directory of per-event arrays
    # without a root .zgroup file. Handle that shape explicitly.
    try:
        child_names = sorted(os.listdir(path))
    except Exception:
        child_names = []
    event_dirs = [
        name for name in child_names
        if os.path.isfile(os.path.join(path, name, ".zarray"))
    ]
    if event_dirs:
        return {
            name: _crop_last2(np.asarray(zarr.open(os.path.join(path, name), mode="r")), crop_size)
            for name in event_dirs
        }

    z = zarr.open(path, mode="r")
    if hasattr(z, "group_keys"):
        keys = list(z.group_keys())
        if keys:
            return {k: _crop_last2(np.asarray(z[k]), crop_size) for k in keys}
    if hasattr(z, "array_keys"):
        keys = list(z.array_keys())
        if keys:
            return {k: _crop_last2(np.asarray(z[k]), crop_size) for k in keys}
    return None


def load_zarr_array(path: str,
                    key: str | None = None,
                    return_events: bool = False,
                    crop_size: int | None = None) -> np.ndarray | dict[str, np.ndarray]:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("zarr path is empty")
    path = path.strip()
    if return_events:
        events = _load_zarr_events_xarray(path, crop_size=crop_size)
        if events is None:
            events = _load_zarr_events_native(path, crop_size=crop_size)
        if events:
            return events
    try:
        arr = _open_zarr_xarray(path, key=key, crop_size=crop_size)
    except Exception:
        arr = _open_zarr_native(path, crop_size=crop_size)
    return np.asarray(arr)


def align_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = ensure_thw(a)
    b = ensure_thw(b)
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def select_by_mask(arr: np.ndarray, mask: np.ndarray, invert: bool = False) -> np.ndarray:
    arr = ensure_thw(arr)
    mask = mask.astype(bool)
    if mask.shape != arr.shape[1:]:
        raise ValueError(f"Mask shape {mask.shape} != data shape {arr.shape[1:]}")
    if invert:
        mask = ~mask
    flat = arr.reshape(arr.shape[0], -1)
    return flat[:, mask.ravel()]


def mask_for_input(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = ensure_thw(arr)
    mask = mask.astype(bool)
    if mask.shape != arr.shape[1:]:
        raise ValueError(f"Mask shape {mask.shape} != data shape {arr.shape[1:]}")
    out = arr.copy()
    out[:, mask] = 0.0
    return out


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_config_snapshot(path: str, cfg) -> None:
    if is_dataclass(cfg):
        payload = asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        payload = cfg.__dict__
    else:
        payload = cfg
    save_json(path, payload)


def save_text(path: str, lines: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")
