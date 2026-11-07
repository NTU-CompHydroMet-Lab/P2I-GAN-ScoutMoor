"""Crop the study domain out of the monthly UK NIMROD stores into one local Zarr store.

The output keeps the raw int16 rain rate (mm/h * 32, -1 = no data) on a regular
5-minute axis; timestamps absent from the source are filled with -1 and flagged
in ``present``. Per-frame statistics are stored alongside so training windows
can be selected without reading the fields again.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

SRC = "/home/NAS/house/isaac-10009/Data/RPP/NIMROD/NIMROD_ZARR"
STEP_MIN = 5
FRAMES_PER_CHUNK = 288
WET_RAW = 2  # > 0.036 mm/h, the old encoding's zero level


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", type=Path, default=Path("datasets/nimrod_domain.zarr"))
    p.add_argument("--start", default="202001", help="first month, YYYYMM")
    p.add_argument("--end", default="202512", help="last month, YYYYMM")
    p.add_argument("--rows", type=int, nargs=2, default=(1084, 1212), help="row slice in the NIMROD grid")
    p.add_argument("--cols", type=int, nargs=2, default=(723, 851), help="column slice in the NIMROD grid")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true", help="fill months not yet marked done in an existing store")
    return p.parse_args()


def months(start: str, end: str) -> list[str]:
    y, m = int(start[:4]), int(start[4:])
    out = []
    while f"{y:04d}{m:02d}" <= end:
        out.append(f"{y:04d}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_start(ym: str) -> datetime:
    return datetime(int(ym[:4]), int(ym[4:]), 1, tzinfo=UTC)


def minutes(dt: datetime) -> int:
    return int(dt.timestamp() // 60)


def create(out: Path, args, month_list: list[str]) -> None:
    t0 = minutes(month_start(month_list[0]))
    last = month_list[-1]
    nxt = months(last, "999912")[1]
    n = (minutes(month_start(nxt)) - t0) // STEP_MIN
    h, w = args.rows[1] - args.rows[0], args.cols[1] - args.cols[0]

    root = zarr.open_group(str(out), mode="w-")
    src0 = zarr.open(f"{args.src}/{month_list[0]}.zarr", mode="r")
    easting, northing = src0["Easting"][args.cols[0] : args.cols[1]], src0["Northing"][args.rows[0] : args.rows[1]]
    root.attrs.update(
        source=args.src,
        rows=list(args.rows),
        cols=list(args.cols),
        units="mm/h*32",
        missing_value=-1,
        step_minutes=STEP_MIN,
        time_units="minutes since 1970-01-01 UTC",
        wet_raw_threshold=WET_RAW,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    root.create_array("rainrate", shape=(n, h, w), chunks=(FRAMES_PER_CHUNK, h, w), dtype="int16", fill_value=-1)
    root.create_array("time", data=(t0 + STEP_MIN * np.arange(n)).astype("int64"))
    root.create_array("easting", data=np.asarray(easting, dtype="int32"))
    root.create_array("northing", data=np.asarray(northing, dtype="int32"))
    for name, dtype, fill in (("present", "bool", False), ("n_missing", "int32", h * w), ("n_wet", "int32", 0),
                              ("max_raw", "int16", -1)):
        root.create_array(name, shape=(n,), chunks=(FRAMES_PER_CHUNK,), dtype=dtype, fill_value=fill)
    root.create_array("month_done", data=np.zeros(len(month_list), dtype=bool))
    root.attrs["months"] = month_list


def process_month(task):
    out, src, ym, rows, cols = task
    root = zarr.open_group(str(out), mode="r+")
    t_axis0 = int(root["time"][0])
    s = zarr.open(f"{src}/{ym}.zarr", mode="r")
    epoch = minutes(month_start(ym))
    t_src = epoch + s["time"][:].astype("int64")
    idx = (t_src - t_axis0) // STEP_MIN
    if not ((t_src - t_axis0) % STEP_MIN == 0).all():
        raise ValueError(f"{ym}: timestamps off the {STEP_MIN}-minute grid")
    nxt = months(ym, "999912")[1]
    lo, hi = (epoch - t_axis0) // STEP_MIN, (minutes(month_start(nxt)) - t_axis0) // STEP_MIN
    if idx.min() < lo or idx.max() >= hi:
        raise ValueError(f"{ym}: timestamps outside the month")

    h, w = rows[1] - rows[0], cols[1] - cols[0]
    field = np.full((hi - lo, h, w), -1, dtype=np.int16)
    rr = s["rainrate"]
    step = rr.chunks[0]
    for a in range(0, rr.shape[0], step):
        b = min(a + step, rr.shape[0])
        field[idx[a:b] - lo] = rr[a:b, rows[0] : rows[1], cols[0] : cols[1]]
    present = np.zeros(hi - lo, dtype=bool)
    present[idx - lo] = True

    root["rainrate"][lo:hi] = field
    root["present"][lo:hi] = present
    root["n_missing"][lo:hi] = (field < 0).sum(axis=(1, 2))
    root["n_wet"][lo:hi] = (field >= WET_RAW).sum(axis=(1, 2))
    root["max_raw"][lo:hi] = field.max(axis=(1, 2))
    return ym, int(present.sum()), hi - lo


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    month_list = months(args.start, args.end)
    if args.out.exists():
        if not args.resume:
            raise FileExistsError(f"{args.out} exists; pass --resume to fill the remaining months")
        root = zarr.open_group(str(args.out), mode="r")
        if root.attrs["months"] != month_list:
            raise ValueError("existing store covers different months")
    else:
        create(args.out, args, month_list)
    root = zarr.open_group(str(args.out), mode="r")
    done = root["month_done"][:]
    todo = [(args.out, args.src, ym, args.rows, args.cols) for ym, d in zip(month_list, done, strict=True) if not d]
    logging.info("%d of %d months to process", len(todo), len(month_list))

    # Each month starts on a day boundary, so months write disjoint rainrate chunks.
    with ProcessPoolExecutor(args.workers) as pool:
        for ym, n_present, n_slots in pool.map(process_month, todo):
            rw = zarr.open_group(str(args.out), mode="r+")
            flags = rw["month_done"][:]
            flags[month_list.index(ym)] = True
            rw["month_done"][:] = flags
            logging.info("%s: %d of %d frames present", ym, n_present, n_slots)


if __name__ == "__main__":
    main()
