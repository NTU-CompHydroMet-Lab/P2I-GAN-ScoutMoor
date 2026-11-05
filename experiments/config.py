from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class ModeConfig:
    observation_path: str
    methods: Dict[str, str]
    mask_train_path: str
    mask_test_path: str
    nimrod_path: str

@dataclass
class ExperimentConfig:
    experiment_name: str = "exp2_test"
    description: str = "fix keep method."
    save_dir: str = "results"
    mode: str = "gauge"  # "radar" or "gauge"
    run_exp1: bool = False
    run_exp2_gif: bool = False
    run_exp2_pdf: bool = True
    run_exp3: bool = False
    crop_size: int = 128
    visualization_vmin: float = 0.0
    visualization_vmax: float = 32.0
    gif_fps: int = 6
    exp1_pool8: bool = True
    exp2_paper_output_pdf: str = "two_events_stacked_titles.pdf"
    exp2_paper_one_event_per_pdf: bool = True
    exp2_paper_output_template: str = "event_{event_idx:02d}_id_{event_id:02d}.pdf"
    exp2_paper_crop_output: str = "cropped_stitched.pdf"
    exp2_paper_mask_path: Optional[str] = None
    exp2_paper_radar_method_order: Tuple[str, ...] = (
        "Radar",
        "P2I-GAN+",
        "P2I-GAN",
        "DK",
        "STDK",
        "OK",
        "RBF",
    )
    # exp2_paper_radar_method_order: Tuple[str, ...] = (
    #     "Obs", 
    #     "DK (Full)", 
    #     "STDK (Full)", 
    #     "DK (w/o pm)", 
    #     "STDK (w/o pm)", 
    #     "DK (mse loss)", 
    #     "STDK (mse loss)",  
    # )
    exp2_paper_gauge_method_order: Tuple[str, ...] = (
        "Gauge",
        "Radar",
        "P2I-GAN+",
        "P2I-GAN",
        "DK",
        "STDK",
        "KED",
        "KRE",
    )

    # exp2_paper_events: Tuple[Dict[str, object], ...] = (
    #     {
    #         "event_id": 5,
    #         "select_idx": (25, 26, 27),
    #         "title": "Stratiform . Start Time : 2021-05-20 11:00:00 UTC",
    #     },
    #     {
    #         "event_id": 19,
    #         "select_idx": (19, 20, 21),
    #         "title": "Convective . Start Time : 2022-10-31 19:35:00 UTC",
    #     },
    # )

    # exp2_paper_events: Tuple[Dict[str, object], ...] = (
    #     {
    #         "event_id": 5,
    #         "select_idx": (25, 26, 27),
    #         "title": "Summer . Start Time : 2021-05-20 11:00:00 UTC",
    #     },
    #     {
    #         "event_id": 10,
    #         "select_idx": (26, 27, 28),
    #         "title": "Winter . Start Time : 2021-12-07 10:10:00 UTC",
    #     },
    # )

    
    exp2_paper_events: Tuple[Dict[str, object], ...] = (
        {"event_id": 1, "select_idx": [90, 91, 92, 93, 94, 95],
         "title": "Start Time : 2021-01-16 05:30:00 UTC"},
        {"event_id": 2, "select_idx": [60, 61, 62, 63, 64, 65],
         "title": "Start Time : 2021-03-10 03:00:00 UTC"},
        {"event_id": 3, "select_idx": [60, 61, 62, 63, 64, 65],
         "title": "Start Time : 2021-05-03 13:00:00 UTC"},
        {"event_id": 4, "select_idx": [90, 91, 92, 93, 94, 95],
         "title": "Start Time : 2021-05-08 08:30:00 UTC"},
        {"event_id": 5, "select_idx": [90, 91, 92, 93, 94, 95],
         "title": "Start Time : 2021-05-20 16:30:00 UTC"},
        {"event_id": 6, "select_idx": [90, 91, 92, 93, 94, 95],
         "title": "Start Time : 2021-05-23 18:30:00 UTC"},
        {"event_id": 7, "select_idx": [90, 91, 92, 93, 94, 95],
         "title": "Start Time : 2021-07-06 00:30:00 UTC"},
        {"event_id": 8, "select_idx": [90, 91, 92, 93, 94, 95],
         "title": "Start Time : 2021-10-02 12:30:00 UTC"},
        {"event_id": 9, "select_idx": [90, 91, 92, 93, 94, 95],
         "title": "Start Time : 2021-10-31 10:30:00 UTC"},
        {"event_id": 10, "select_idx": [90, 91, 92, 93, 94, 95],
         "title": "Start Time : 2021-12-07 15:30:00 UTC"},
    )

    exp2_paper_folders: Dict[str, str] = field(default_factory=dict)
    data: Dict[str, ModeConfig] = field(default_factory=dict)

def build_config() -> ExperimentConfig:
    radar_mode = ModeConfig(
        observation_path=(
            "datasets/nimrod/nimrod_test.zarr"
        ),
        nimrod_path=(
            "datasets/nimrod/nimrod_test.zarr"
        ),
        # methods={
        #     "DK (Full)": (
        #         "datasets/infer/dk_v3_pm_nimrod.zarr"
        #     ),
        #     "STDK (Full)": (
        #         "datasets/infer/stdk_v3_pm_nimrod.zarr"
        #     ),
        #     "DK (w/o pm)": (
        #         "datasets/predict/v2/dk_nimrod.zarr"
        #     ),
        #     "STDK (w/o pm)": (
        #         "datasets/predict/v2/stdk_nimrod.zarr"
        #     ),
        #     "DK (mse loss)": (
        #         "datasets/predict/dk_mse_nimrod.zarr"
        #     ),
        #     "STDK (mse loss)": (
        #         "datasets/predict/stdk_nimrod.zarr"
        #     ),
        # },
        methods={
            "P2I-GAN+": (
                "datasets/infer/p2igan_gan_baseline_v3_pm_nimrod.zarr"
            ),
            "P2I-GAN": (
                "datasets/infer/p2igan_baseline_v3_pm_nimrod.zarr"
            ),
            "DK": (
                "datasets/infer/dk_v3_pm_nimrod.zarr"
            ),
            "STDK": (
                "datasets/infer/stdk_v3_pm_nimrod.zarr"
            ),
            "OK": (
                "datasets/infer/ok_nimrod.zarr"
            ),
            "RBF": (
                "datasets/infer/rbf_nimrod.zarr"
            ),
        },
        mask_train_path=(
            "datasets/masks/gauge_mask_128_train.txt"
        ),
        mask_test_path=(
            "datasets/masks/gauge_mask_128_test.txt"
        ),
    )

    gauge_mode = ModeConfig(
        observation_path=(
            "datasets/midas/midas_test.zarr"
        ),
        nimrod_path=(
            "datasets/nimrod/nimrod_test.zarr"
        ),
        methods={
            "P2I-GAN+": (
                "datasets/infer/p2igan_gan_baseline_v3_pm_gauge.zarr"
            ),
            "P2I-GAN": (
                "datasets/infer/p2igan_baseline_v3_pm_gauge.zarr"
            ),
            "DK": (
                "datasets/infer/dk_v3_pm_gauge.zarr"
            ),
            "STDK": (
                "datasets/infer/stdk_v3_pm_gauge.zarr"
            ),
            "KRE": (
                "datasets/predict/kre_gauge_v2.zarr"
            ),
            "KED": (
                "datasets/predict/ked_gauge_v2.zarr"
            ),
        },
        mask_train_path=(
            "datasets/masks/gauge_mask_128_train.txt"
        ),
        mask_test_path=(
            "datasets/masks/gauge_mask_128_test.txt"
        ),
    )

    cfg = ExperimentConfig()
    cfg.data = {"radar": radar_mode, "gauge": gauge_mode}
    return cfg


def get_mode_config(cfg: ExperimentConfig) -> ModeConfig:
    mode_cfg = cfg.data.get(cfg.mode)
    if mode_cfg is None:
        raise ValueError(f"Unknown mode: {cfg.mode}")
    return mode_cfg
