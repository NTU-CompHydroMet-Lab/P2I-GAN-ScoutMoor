import os
import re

import h5py
import numpy as np
import torch
import zarr
from torchvision.transforms import Compose, Lambda

transform = Compose([
    Lambda(lambda x: torch.tensor(x))  # Convert to tensor
])

def create_mask(video_tensor, mask_file):
    """Return a float mask (T, H, W, C) with 1 at the gauge pixels listed in ``mask_file``.

    ``mask_file`` is a text grid of 0/1 with the same H x W as ``video_tensor``;
    the same gauges are observed in every frame.
    """
    T, H, W, C = video_tensor.shape
    mask = torch.zeros((T, H, W, C), dtype=torch.float32)
    with open(mask_file) as f:
        mask_matrix = np.loadtxt(f)
    mask_matrix = torch.tensor(mask_matrix, dtype=torch.bool).unsqueeze(-1)
    if mask_matrix.shape != (H, W, 1):
        raise ValueError(f"Mask matrix in {mask_file} does not match video spatial dimensions {H}x{W}")
    for t in range(T):
        mask[t, :, :, :] = mask_matrix
    return mask

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        """HDF5 directory or Zarr group of events, masked with a fixed gauge mask."""
        self.video_folder = args['data_root']
        self.transform = transform
        self.is_zarr = str(self.video_folder).endswith(".zarr")
        self.zarr_root = None
        if self.is_zarr:
            self.zarr_root = zarr.open(self.video_folder, mode="r")
            self.video_files = sorted(self.zarr_root.array_keys())
        else:
            self.video_files = sorted(
                [
                    os.path.join(self.video_folder, f)
                    for f in os.listdir(self.video_folder)
                    if f.endswith('.h5')
                ],
                key=lambda f: extract_number(os.path.basename(f))
            )

        mask_cfg = args.get('mask', {})
        if mask_cfg.get('type') != 'stis' or not mask_cfg.get('file'):
            raise ValueError("Only mask.type='stis' with mask.file is supported.")
        self.mask_file = mask_cfg['file']

        self.width = args['w']
        self.height = args['h']
        self.sample_length = args.get('sample_length')

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        if idx >= len(self.video_files):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.video_files)} samples.")

        return self.process_file(self.video_files[idx])

    def process_file(self, file_path):
        if self.is_zarr:
            return self.process_zarr(file_path)
        if file_path.endswith('.h5'):
            return self.process_hdf5(file_path)
        raise ValueError(f"Unsupported file format: {file_path}")

    def process_hdf5(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            video_data = f['frames'][:]
     
            # 確保數據格式正確
            if video_data.ndim == 3:  # 如果是 (T, H, W)
                video_data = video_data[..., np.newaxis]  # 變成 (T, H, W, 1)
        return self.post_process(video_data)

    def process_zarr(self, key):
        if self.zarr_root is None:
            raise RuntimeError("Zarr root not initialized.")
        video_data = self.zarr_root[key][:]
        if video_data.ndim == 3:
            video_data = video_data[..., np.newaxis]
        elif video_data.ndim == 4 and video_data.shape[-1] != 1:
            video_data = np.mean(video_data, axis=-1, keepdims=True)
        return self.post_process(video_data)

    def post_process(self, video_data):
        # 維持原時間長度或根據 sample_length 截斷
        if self.sample_length is not None:
            T = min(self.sample_length, video_data.shape[0])
            video_data = video_data[:T]
        
        video_data = video_data.astype(np.float32) / 255.0
        if self.transform:
            video_data = self.transform(video_data)

        mask = create_mask(video_data, self.mask_file)
        masked_video = video_data.clone()
        masked_video = masked_video * mask

        video_data = self._crop_center(video_data)
        masked_video = self._crop_center(masked_video)
        mask = self._crop_center(mask)
        return video_data, masked_video, mask

    def _crop_center(self, data):
        if data.shape[1] == self.height and data.shape[2] == self.width:
            return data
        old_height, old_width = data.shape[1], data.shape[2]
        start_x = max((old_width - self.width) // 2, 0)
        start_y = max((old_height - self.height) // 2, 0)
        end_y = start_y + self.height
        end_x = start_x + self.width
        return data[:, start_y:end_y, start_x:end_x, :]
