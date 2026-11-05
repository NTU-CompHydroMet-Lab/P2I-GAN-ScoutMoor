from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from experiments.config import ModeConfig
from experiments.io import load_zarr_array


def sample_values(arr: np.ndarray, max_samples: int = 1_000_000, seed: int = 42) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size <= max_samples:
        return flat
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.size, size=max_samples, replace=False)
    return flat[idx]


def plot_hist(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, out_path: str) -> None:
    plt.figure(figsize=(8, 4), dpi=150)
    plt.hist(a, bins=200, alpha=0.6, label=label_a, density=True)
    plt.hist(b, bins=200, alpha=0.6, label=label_b, density=True)
    plt.yscale("log")
    plt.xlabel("Value")
    plt.ylabel("Density (log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def describe(name: str, arr: np.ndarray, max_samples: int = 1_000_000) -> np.ndarray:
    samples = sample_values(arr, max_samples=max_samples)
    shape = arr.shape
    if samples.size == 0:
        print(f"{name}: shape={shape}, count=0 (no finite values)")
        return samples
    print(
        f"{name}: shape={shape}, count={samples.size}, "
        f"min={samples.min():.6f}, max={samples.max():.6f}, "
        f"mean={samples.mean():.6f}, std={samples.std():.6f}"
    )
    return samples


def main() -> None:
    radar_mode = ModeConfig(
        observation_path=(
            "datasets/nimrod/nimrod_test.zarr"
        ),
        truth_path=(
            "datasets/nimrod/nimrod_test.zarr"
        ),
        methods={
            "P2IGAN": (
                "datasets/infer/p2igan_test.zarr"
            ),
            "DK": (
                "datasets/infer/dk_test.zarr"
            ),
            "STDK": (
                "datasets/infer/stdk_test.zarr"
            ),
        },
        mask_train_path=(
            "datasets/masks/gauge_mask_128_train.txt"
        ),
        mask_test_path=(
            "datasets/masks/gauge_mask_128_test.txt"
        ),
    )

    obs = load_zarr_array(radar_mode.observation_path)
    obs_s = describe("observation", obs)

    for name, path in radar_mode.methods.items():
        pred = load_zarr_array(path)
        pred_s = describe(name, pred)
        out_path = f"zarr_value_hist_{name.lower()}.png"
        plot_hist(obs_s, pred_s, "observation", name, out_path)
        print(f"Saved histogram to {out_path}")


if __name__ == "__main__":
    main()
