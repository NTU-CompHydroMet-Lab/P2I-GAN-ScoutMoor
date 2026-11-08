"""Fixed-length windows from the cropped NIMROD domain store, observed at the gauge pixels.

Items match the legacy ``Dataset``: ``(video, masked_video, mask)``, each ``(T, H, W, 1)`` float32,
with rain rate on the P2I scale (``p2igan_bench.codec``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import zarr

from p2igan_bench.codec import P2I_CODEC

RAW_PER_MMH = 32.0


def encode(raw: torch.Tensor) -> torch.Tensor:
    """Raw store units (mm/h * 32) to the P2I scale in [0, 1]."""
    return P2I_CODEC.encode(raw.float().clamp_min(0) / RAW_PER_MMH)


def gauge_pixels(csv_path: str, easting: np.ndarray, northing: np.ndarray) -> np.ndarray:
    """Unique (row, col) of the gauges in the store's grid; gauges outside the grid are dropped."""
    table = pd.read_csv(csv_path)
    col = {int(e): i for i, e in enumerate(easting)}
    row = {int(n): i for i, n in enumerate(northing)}
    pix = {(row[n], col[e]) for e, n in zip(table["PixelE"], table["PixelN"], strict=True) if e in col and n in row}
    return np.array(sorted(pix), dtype=np.int64)


def select_windows(wet_frac, max_raw, missing, length, stride, wet_fraction, peak_raw, start_ok):
    """Start indices of windows with no missing frame and at least one frame passing both thresholds."""
    ok = (wet_frac >= wet_fraction) & (max_raw >= peak_raw)
    cs = np.concatenate([[0], np.cumsum(ok)])
    cm = np.concatenate([[0], np.cumsum(missing)])
    n = len(ok) - length + 1
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    keep = ((cs[length:] - cs[:n]) > 0) & ((cm[length:] - cm[:n]) == 0) & start_ok[:n]
    starts = np.nonzero(keep)[0]
    if stride > 1:
        picked, last = [], -stride
        for s in starts:
            if s >= last + stride:
                picked.append(s)
                last = s
        starts = np.asarray(picked, dtype=np.int64)
    return starts


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, split: dict):
        root = zarr.open_group(split["store"], mode="r")
        self.length = int(split.get("sample_length", 16))
        sel = {"wet_raw": 2, "wet_fraction": 0.3, "peak_raw": 65, **split.get("selection", {})}
        if sel["wet_raw"] != root.attrs["wet_raw_threshold"]:
            raise ValueError(f"store counts wet pixels at raw >= {root.attrs['wet_raw_threshold']}, "
                             f"config asks for {sel['wet_raw']}")

        minutes = root["time"][:]
        lo = int(np.datetime64(split["start"], "m").astype("int64"))
        hi = int((np.datetime64(split["end"], "D") + np.timedelta64(1, "D")).astype("datetime64[m]").astype("int64"))
        in_range = (minutes >= lo) & (minutes < hi)
        first, last = np.nonzero(in_range)[0][[0, -1]]
        span = slice(first, last + 1)

        n_pix = root["rainrate"].shape[1] * root["rainrate"].shape[2]
        wet = root["n_wet"][span] / n_pix
        max_raw = root["max_raw"][span]
        missing = (root["n_missing"][span] > 0) | ~root["present"][span]
        starts = select_windows(wet, max_raw, missing, self.length, int(split.get("stride", 1)),
                                sel["wet_fraction"], sel["peak_raw"], np.ones(len(wet), dtype=bool))
        if len(starts) == 0:
            raise ValueError(f"no windows selected in {split['start']}..{split['end']}")

        # Load only the frames some window needs; windows stay contiguous in the compact array.
        need = np.zeros(len(wet), dtype=bool)
        for k in range(self.length):
            need[starts + k] = True
        pos = np.cumsum(need) - 1
        frames = np.empty((int(need.sum()), *root["rainrate"].shape[1:]), dtype=np.int16)
        chunk = root["rainrate"].chunks[0]
        filled = 0
        for a in range(0, len(need), chunk):
            b = min(a + chunk, len(need))
            m = need[a:b]
            if m.any():
                block = root["rainrate"][first + a : first + b]
                frames[filled : filled + int(m.sum())] = block[m]
                filled += int(m.sum())
        self.frames = torch.from_numpy(frames).share_memory_()
        self.starts = torch.from_numpy(pos[starts]).long()
        self.times = minutes[span][starts]

        self.gauges = gauge_pixels(split["gauges"], root["easting"][:], root["northing"][:])
        h, w = root["rainrate"].shape[1:]
        if (split.get("h", h), split.get("w", w)) != (h, w):
            raise ValueError(f"store is {h}x{w}, config asks for {split.get('h')}x{split.get('w')}")
        grid = torch.zeros(h, w)
        grid[self.gauges[:, 0], self.gauges[:, 1]] = 1.0
        self.mask = grid.expand(self.length, h, w).unsqueeze(-1).contiguous()

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        video = encode(self.frames[s : s + self.length]).unsqueeze(-1)
        return video, video * self.mask, self.mask
