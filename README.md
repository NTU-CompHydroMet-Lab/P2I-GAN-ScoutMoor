

## Points to images: deep-learning enhanced spatial-temporal rainfall modeling from point measurements.

P2I-GAN is a deep generative framework that treats rainfall interpolation as a video inpainting task, enabling reconstruction of spatio-temporal rainfall fields from highly sparse and irregular rain-gauge observations.
This repository provides training scripts, data preprocessing workflows, visualization tools, and evaluation utilities so researchers can rapidly experiment with point-to-image reconstruction using an open-source benchmark.

⭐ If P2IGAN is helpful to your projects, please help star this repo. Thanks! 🤗

---

## Update

- **2026.05.18**: Update model weight of P2IGAN, DK, STDK, and Midas dataset.
- **2026.02.10**: Our code and model are publicly available. 🐳 
- **2025.12.24**: The training pipelines are updated.
- **2025.12.06**: This repo is created.


## Results

#### 👨🏻‍🎨 Radar-Input

<img src="assets/comparison_event_04.gif">

<img src="assets/comparison_event_05.gif">

#### 🎨 Gauge Input

<img src="assets/gauge_event_03.gif">

<img src="assets/gauge_event_07.gif">



## Overview
![overall_structure](assets/P2IGAN_pipeline.png)


## Dependencies and Installation

1. Clone Repo

   ```bash
   git clone https://github.com/NTU-CompHydroMet-Lab/P2I-GAN-benchmark
   ```

2. Install dependencies with uv (Python version and packages are pinned by `uv.lock`)

   ```bash
   pip install -U uv
   cd P2I-GAN-benchmark
   uv sync
   ```

   Run commands through `uv run …`, or activate `.venv` first.
 
## Get Started
### Prepare pretrained models
Download our pretrained models from [Releases V0.1.0](https://drive.google.com/drive/folders/12Yz7JDCZiR-BESAiTT0Ypp6b-7KJfpv9?usp=sharing) The directory structure is arranged as:

```
folder
   |- dk
      |- latest.pt
   |- stdk
      |- latest.pt
   |- p2igan
      |- latest.pt
   |- p2igan_plus
      |- latest.pt
   |- midas
      |- midas_test.zarr
```

### Local data layout

All config paths are relative to the repository root, so run every command from there. `datasets/`, `weights/` and `legacy/` are git-ignored and hold local copies of the original developer's files:

| Path | Contents |
| --- | --- |
| `datasets/nimrod_h5/{train,test,val}` | HDF5 events used by the shipped configs |
| `datasets/masks/` | Gauge masks (`gauge_mask_{128,1024}_{train,test}.txt`) |
| `datasets/midas/midas_test.zarr` | MIDAS gauge test events |
| `datasets/nimrod/` | `nimrod_test.zarr`, `nimrod_train.zarr` |
| `datasets/infer/`, `datasets/predict/` | Stored model and baseline outputs used by `experiments/` |
| `weights/` | Released checkpoints, one directory per run |
| `legacy/mlruns/` | MLflow metadata of the original runs (no artifacts) |

`scripts/tools/` holds the scripts that built these datasets. Their raw inputs are still read from `/home/NAS/homes/brick-10015/Nimrod_2d_data`, which is not copied.

### Tests

```bash
uv run pytest -m "not slow"   # strict checkpoint loading, golden vs. author output
uv run pytest                 # adds v9 inference on all MIDAS test events (CUDA, ~40 s)
```

The golden reference `datasets/golden/v9_midas_gauge.zarr` was produced by tag `baseline-v0`; its attrs record the GPU, driver and torch version. Inference is bit-exact only on that setup. On another GPU, expect differences of about 2 (on the 0–255 scale). On CPU, results are not comparable, because the IDW in `InputBlock` runs in float16 on CUDA and in float32 on CPU.

### Inference with Fake Data

Since we cannot share the original Nimrod (radar) with you, we provide MIDAS (gauge) datasets we use in our experiment instead. After downloaded the datasets, you can just use code here to see the result.

```bash
python scripts/infer.py \
  --config p2igan_bench/config/p2igan_gan_baseline.json \
  --checkpoint weights/p2igan_gan_v9/latest.pt \
  --output datasets/infer/p2igan_gauge.zarr
```

## Dataset Preparation

Training and validation read directories of HDF5 files; testing reads a Zarr group.

**HDF5 (train / valid).** Each `.h5` file holds a dataset `frames` of shape `(T, H, W)`, `uint8`. The loader keeps the first `sample_length` frames, divides by 255 and centre-crops to `h × w`; the gauge mask file must match the full `H × W`. The shipped data are 20 × 1024 × 1024 radar frames encoded as `v = 48·log10(R / 0.036)` with `R` in mm/h.

```
datasets/nimrod_h5
├── train   # 201601011320a.h5, 201601011320b.h5, ...
├── test    # used as data.valid by the shipped configs
└── val     # 1.h5 ... 20.h5 (test events)
```

**Zarr (test).** A Zarr group whose arrays are events of shape `(T, H, W)`, for example `midas_test.zarr/event_01` with shape `(241, 128, 128)`. Whole events are read and inferred with a sliding window (`--stride`, `--overlap` in `scripts/infer.py`).

In our experiments the data sources are **Nimrod (radar)** and **MIDAS (rain gauge)**.

## Training
Training configs are in [`p2igan_bench/config/`](./p2igan_bench/config/).

Run one of the following commands for training:
```shell
 # For training P2IGAN
 python scripts/train.py --config p2igan_bench/config/p2igan_gan_baseline.json
 
 # For monitoring in mlflow
 mlflow ui --backend-store-uri file:<project_path>/mlruns --port 5000

  # For infer P2IGAN
 python scripts/infer.py --config p2igan_bench/config/p2igan_gan_baseline.json

```


## Evaluation
Run from the repository root:
```shell
 python -m experiments.main
```
## License
This project is licensed under the MIT License — see the LICENSE file for details.

## Citation
```
@software{wang2026p2igan,
  author       = {Wang, Bing-Zhang and Wang, Li-Pen},
  title        = {P2I-GAN Benchmark: Deep Generative Framework for Spatio-Temporal Rainfall Reconstruction from Sparse Gauges},
  year         = {2026},
  version      = {0.1.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18623976},
  url          = {https://github.com/NTU-CompHydroMet-Lab/P2I-GAN-benchmark}
}
```

## Contact
If you have any questions about the technical issues, please feel free to reach me out at r13521608@caece.net. 

