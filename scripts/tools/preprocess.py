# import os
# import h5py
# import zarr
# import numpy as np
# from datetime import datetime
# from tqdm import tqdm


# # ============================================================
# # 1. 路徑設定
# # ============================================================
# H5_DIR = "/home/NAS/homes/brick-10015/Nimrod_2d_data/NIMROD_Reader/event_h5_output"
# ZARR_ROOT_DIR = "datasets/midas"
# ZARR_PATH = os.path.join(ZARR_ROOT_DIR, "midas_test.zarr")

# os.makedirs(ZARR_ROOT_DIR, exist_ok=True)



# # ============================================================
# # 2. 事件表（對應你 LaTeX table）
# # ============================================================
# EVENT_TABLE = [
#     dict(id=1,  start="2021-01-15 22:00", end="2021-01-16 18:00", duration=20,
#          max_rg=26.0,  max_rd=22.9, mean_rg=6.6,  mean_rd=10.3),
#     dict(id=2,  start="2021-03-09 22:00", end="2021-03-11 00:00", duration=26,
#          max_rg=81.2,  max_rd=45.4, mean_rg=12.8, mean_rd=12.7),
#     dict(id=3,  start="2021-05-03 08:00", end="2021-05-04 05:00", duration=21,
#          max_rg=63.6,  max_rd=33.6, mean_rg=15.9, mean_rd=13.5),
#     dict(id=4,  start="2021-05-08 01:00", end="2021-05-08 21:00", duration=20,
#          max_rg=52.2,  max_rd=45.9, mean_rg=16.8, mean_rd=13.7),
#     dict(id=5,  start="2021-05-20 09:00", end="2021-05-22 02:00", duration=41,
#          max_rg=107.8, max_rd=96.5, mean_rg=18.4, mean_rd=18.6),
#     dict(id=6,  start="2021-05-23 11:00", end="2021-05-24 07:00", duration=20,
#          max_rg=29.0,  max_rd=33.5, mean_rg=9.5,  mean_rd=9.0),
#     dict(id=7,  start="2021-07-05 17:00", end="2021-07-06 13:00", duration=20,
#          max_rg=29.8,  max_rd=32.8, mean_rg=12.5, mean_rd=11.8),
#     dict(id=8,  start="2021-10-02 05:00", end="2021-10-03 02:00", duration=21,
#          max_rg=37.6,  max_rd=67.8, mean_rg=14.6, mean_rd=13.3),
#     dict(id=9,  start="2021-10-31 03:00", end="2021-11-01 07:00", duration=28,
#          max_rg=53.4,  max_rd=61.0, mean_rg=18.1, mean_rd=16.3),
#     dict(id=10, start="2021-12-07 08:00", end="2021-12-08 05:00", duration=21,
#          max_rg=38.8,  max_rd=34.4, mean_rg=9.0,  mean_rd=9.7),
#     # dict(id=11, start="2022-01-08 04:00", end="2022-01-08 22:00", duration=18,
#     #      max_rg=29.8,  max_rd=24.4, mean_rg=7.4,  mean_rd=8.1),
#     # dict(id=12, start="2022-03-16 11:00", end="2022-03-17 05:00", duration=18,
#     #      max_rg=27.8,  max_rd=27.5, mean_rg=7.9,  mean_rd=8.6),
#     # dict(id=13, start="2022-04-12 06:00", end="2022-04-12 19:00", duration=13,
#     #      max_rg=13.4,  max_rd=24.4, mean_rg=2.8,  mean_rd=3.0),
#     # dict(id=14, start="2022-05-11 02:00", end="2022-05-11 20:00", duration=18,
#     #      max_rg=22.2,  max_rd=23.6, mean_rg=6.3,  mean_rd=6.2),
#     # dict(id=15, start="2022-05-15 20:00", end="2022-05-16 11:00", duration=15,
#     #      max_rg=19.8,  max_rd=22.7, mean_rg=4.9,  mean_rd=5.1),
#     # dict(id=16, start="2022-06-18 15:00", end="2022-06-19 06:00", duration=15,
#     #      max_rg=18.6,  max_rd=25.3, mean_rg=3.5,  mean_rd=4.8),
#     # dict(id=17, start="2022-08-25 05:00", end="2022-08-25 19:00", duration=14,
#     #      max_rg=34.6,  max_rd=32.5, mean_rg=4.9,  mean_rd=5.5),
#     # dict(id=18, start="2022-09-30 09:00", end="2022-10-01 03:00", duration=18,
#     #      max_rg=51.4,  max_rd=31.7, mean_rg=11.9, mean_rd=10.4),
#     # dict(id=19, start="2022-10-31 18:00", end="2022-11-01 12:00", duration=18,
#     #      max_rg=32.8,  max_rd=30.9, mean_rg=11.6, mean_rd=11.7),
#     # dict(id=20, start="2022-11-23 03:00", end="2022-11-23 19:00", duration=16,
#     #      max_rg=21.6,  max_rd=25.4, mean_rg=8.3,  mean_rd=8.5),
# ]

# EVENT_TABLE = sorted(EVENT_TABLE, key=lambda x: x["id"])

# # ============================================================
# # 3. 建立 Zarr root 與 dataset-level attrs
# # ============================================================
# root = zarr.open(ZARR_PATH, mode="w")

# root.attrs.update({
#     "dataset_name": "MIDAS_2D_val",
#     "description": "MIDAS rain gauge data for storm events",
#     "num_events": 10,
#     "spatial_shape": [128, 128],
#     "time_unit": "minutes",
#     "time_resolution": 5,
#     "value_unit": "mm/h",
#     "missing_value": 0.0,
#     "created_by": "brick-10015",
#     "creation_date": "2026-01-16",
#     "source": "MIDAS gauge data provided by the UK Met Office",
# })


# # ============================================================
# # 4. 逐事件轉換 h5 -> zarr + event-level attrs
# # ============================================================
# for evt in tqdm(EVENT_TABLE, desc="Converting events"):
#     eid = evt["id"]
#     h5_path = os.path.join(H5_DIR, f"event_{eid}.h5")
#     event_name = f"event_{eid:02d}"

#     with h5py.File(h5_path, "r") as f:
#         data = f["frames"][:]

#     # squeeze channel if needed
#     if data.ndim == 4 and data.shape[1] == 1:
#         data = data[:, 0]

#     T, H, W = data.shape

#     arr = root.create_dataset(
#         name=event_name,
#         data=data.astype(np.float32),
#         chunks=(1, H, W),
#         compressor=zarr.Blosc(cname="zstd", clevel=3),
#         overwrite=True,
#     )

#     # event-level metadata
#     arr.attrs.update({
#         "event_id": eid,
#         "start_time": evt["start"],
#         "end_time": evt["end"],
#         "duration_hours": evt["duration"],
#         "num_frames": T,
#         "max_rainfall_rg_mm": evt["max_rg"],
#         "max_rainfall_rd_mm": evt["max_rd"],
#         "mean_rainfall_rg_mm": evt["mean_rg"],
#         "mean_rainfall_rd_mm": evt["mean_rd"],
#         "source_file": f"{eid}.h5",
#     })

# print(f"Zarr dataset created at: {ZARR_PATH}")


import os
import re
import h5py
import zarr
import numpy as np
from tqdm import tqdm

# ============================================================
# 1. 路徑設定
# ============================================================
H5_DIR = "/home/NAS/homes/brick-10015/Nimrod_2d_data/NIMROD_Train_datasets/hdf5"
ZARR_PATH = "datasets/nimrod/nimrod_train.zarr"

WINDOW = 20
STRIDE = 1

os.makedirs(os.path.dirname(ZARR_PATH), exist_ok=True)

# ============================================================
# 2. 依時間戳排序 h5 檔案
# ============================================================
def extract_timestamp(fname):
    return int(re.search(r"\d+", fname).group())

h5_files = sorted(
    [f for f in os.listdir(H5_DIR) if f.endswith(".h5")],
    key=extract_timestamp
)

# ============================================================
# 3. 建立 Zarr root
# ============================================================
root = zarr.open(ZARR_PATH, mode="w")
events_grp = root.create_group("events")
index_grp  = root.create_group("index")

root.attrs.update({
    "dataset_name": "NIMROD_2D_train",
    "description": "NIMROD radar events (2016–2020), event-based storage",
    "frame_unit": "mm/h (uint8 encoded)",
    "spatial_shape": [1024, 1024],
    "suggested_window": WINDOW,
    "created_by": "brick-10015",
})

# ============================================================
# 4. 逐檔轉換 + 建立 window index
# ============================================================
window_index = []

event_id = 0

for fname in tqdm(h5_files, desc="Packing events"):
    ts = extract_timestamp(fname)
    h5_path = os.path.join(H5_DIR, fname)

    with h5py.File(h5_path, "r") as f:
        frames = f["frames"][:]   # (T, H, W)

    T, H, W = frames.shape
    event_key = f"{ts}"

    # --- 建立 event group ---
    evt = events_grp.create_group(event_key)
    SPATIAL_CHUNK = 128  # 或 256
    arr = evt.create_dataset(
    "frames",
    data=frames,
    chunks=(WINDOW, SPATIAL_CHUNK, SPATIAL_CHUNK),
    dtype=np.uint8,
    compressor=zarr.Blosc(cname="zstd", clevel=3),
    )


    evt.attrs.update({
        "event_id": event_id,
        "timestamp": ts,
        "num_frames": T,
        "source_file": fname,
    })

    # --- 建立 sliding window index ---
    for start in range(0, T - WINDOW + 1, STRIDE):
        window_index.append([event_id, start, WINDOW])

    event_id += 1

# ============================================================
# 5. 儲存 index table
# ============================================================
index_arr = index_grp.create_dataset(
    "windows",
    data=np.array(window_index, dtype=np.int32),
    chunks=(1024, 3),
    compressor=zarr.Blosc(cname="zstd", clevel=3),
)

index_arr.attrs.update({
    "columns": ["event_id", "start_t", "length"],
    "description": "Sliding window index for training",
})

print(f"Zarr training dataset created at: {ZARR_PATH}")
print(f"Total training samples (windows): {len(window_index)}")
