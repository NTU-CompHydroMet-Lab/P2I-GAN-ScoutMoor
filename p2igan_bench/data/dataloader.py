from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from .sti_dataset import Dataset


class P2IDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        data_cfg = cfg["data"]
        train_cfg = cfg["train"]
        if any(split and split.get("kind") == "domain" for split in data_cfg.values()):
            raise NotImplementedError("data split kind 'domain' is not implemented yet")
        self.num_workers = train_cfg.get("num_workers", 0)
        self.pin_memory = train_cfg.get("pin_memory", True)
        self.persistent_workers = train_cfg.get("persistent_workers", True)
        self.prefetch_factor = train_cfg.get("prefetch_factor", 2)

        train_cfg = data_cfg["train"]
        self.train_args = self._build_dataset_args(train_cfg)
        shared_params = self._extract_shared_params(self.train_args)

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        valid_cfg = data_cfg.get("valid")
        self.valid_args = None
        self.valid_shuffle = False

        self.train_dataset = Dataset(self.train_args)
        if valid_cfg:
            # validation inherits train w/h/sample_length/mask so its clips match the model length
            self.valid_args = self._build_dataset_args(valid_cfg, defaults=shared_params)
            self.valid_shuffle = bool(valid_cfg.get("shuffle", False))
            self.valid_dataset = Dataset(self.valid_args)

        test_cfg = data_cfg.get("test")
        self.test_args = None
        self.test_shuffle = False
        if test_cfg:
            test_defaults = self._drop_sample_length(shared_params)
            self.test_args = self._build_dataset_args(test_cfg, defaults=test_defaults)
            self.test_shuffle = bool(test_cfg.get("shuffle", False))
            self.test_dataset = Dataset(self.test_args)

    def train_dataloader(self):
        train_bs = self.cfg["train"]["batch_size"]
        if self.train_dataset is None:
            return None
        return self._create_loader(self.train_dataset, shuffle=True, batch_size=train_bs)

    def val_dataloader(self):
        if self.valid_dataset is None:
            return None
        train_bs = self.cfg["train"]["batch_size"]
        return self._create_loader(self.valid_dataset, shuffle=self.valid_shuffle, batch_size=train_bs)

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        test_bs = 1
        return self._create_loader(self.test_dataset, shuffle=self.test_shuffle, batch_size=test_bs)

    def _create_loader(self, dataset, shuffle, batch_size):
        collate_fn = None
        if getattr(dataset, "is_zarr", False) and getattr(dataset, "sample_length", None) is None:
            collate_fn = self._collate_variable_length
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    def _build_dataset_args(self, split_cfg, defaults=None):
        defaults = defaults or {}
        args = {}

        for key in ("w", "h", "sample_length"):
            if key in defaults:
                args[key] = defaults[key]
            if key in split_cfg:
                value = split_cfg[key]
                if value is None and key in args:
                    args.pop(key)
                elif value is not None:
                    args[key] = value

        mask_cfg = deepcopy(defaults.get("mask", {}))
        if split_cfg.get("mask"):
            mask_cfg.update(split_cfg["mask"])
        if mask_cfg:
            args["mask"] = mask_cfg

        if "data_root" not in split_cfg:
            raise KeyError("Dataset config requires 'data_root'.")
        args["data_root"] = split_cfg["data_root"]

        return args

    def _extract_shared_params(self, dataset_args):
        shared = {}
        for key in ("w", "h", "sample_length"):
            if key in dataset_args:
                shared[key] = dataset_args[key]
        if "mask" in dataset_args:
            shared["mask"] = deepcopy(dataset_args["mask"])
        return shared

    def _drop_sample_length(self, params):
        params = deepcopy(params)
        params.pop("sample_length", None)
        return params

    @staticmethod
    def _collate_variable_length(batch):
        videos, masked_videos, masks = zip(*batch, strict=True)
        max_len = max(v.shape[0] for v in videos)

        def _pad(seq):
            if seq.shape[0] == max_len:
                return seq
            pad_len = max_len - seq.shape[0]
            pad = seq[-1:].repeat(pad_len, 1, 1, 1)
            return torch.cat([seq, pad], dim=0)

        videos = torch.stack([_pad(v) for v in videos], dim=0)
        masked_videos = torch.stack([_pad(v) for v in masked_videos], dim=0)
        masks = torch.stack([_pad(v) for v in masks], dim=0)
        return videos, masked_videos, masks
