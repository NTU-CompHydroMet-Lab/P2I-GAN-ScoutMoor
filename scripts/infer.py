from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from p2igan_bench.data.dataloader import P2IDataModule
from p2igan_bench.models import build_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for P2I-GAN benchmark models")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("p2igan_bench/config/p2igan_baseline.json"),
        help="Path to JSON/YAML config file.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory containing latest.pt.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override data.test.data_root.")
    parser.add_argument("--output", type=Path, default=None, help="Output zarr path.")
    parser.add_argument("--passes", type=int, default=1, help="Number of inference passes to average.")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda:0, cpu).")
    parser.add_argument("--log-every", type=int, default=50, help="Log progress every N samples.")
    parser.add_argument("--stride", type=int, default=16, help="Sliding window length.")
    parser.add_argument("--overlap", type=int, default=12, help="Sliding window overlap.")
    parser.add_argument("--output-scale", type=float, default=255.0, help="Scale factor for outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output zarr.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(f)
        return json.load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return args.checkpoint

    base_dir = args.model_dir or Path(cfg.get("save_dir", "weights"))
    base_dir = Path(base_dir)
    if base_dir.is_file():
        return base_dir

    latest = base_dir / "latest.pt"
    if latest.exists():
        return latest

    if base_dir.exists():
        candidates = sorted(base_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            logging.warning("latest.pt not found, falling back to %s", candidates[0])
            return candidates[0]

    raise FileNotFoundError(f"Checkpoint not found under {base_dir}")


def prepare_batch(batch, device: torch.device):
    frames, masked_frames, masks = batch
    frames = frames.permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
    masked_frames = masked_frames.permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
    masks = masks.permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
    return frames, masked_frames, masks


def main(args: argparse.Namespace | None = None) -> None:
    parsed = args or parse_args()
    logging.basicConfig(
        level=getattr(logging, parsed.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Loading config from %s", parsed.config)
    cfg = load_config(parsed.config)
    seed_everything(cfg.get("seed", 42))

    if parsed.data_root is not None:
        cfg.setdefault("data", {}).setdefault("test", {})["data_root"] = str(parsed.data_root)

    if parsed.device:
        cfg["device"] = parsed.device

    checkpoint_path = resolve_checkpoint(cfg, parsed)
    logging.info("Using checkpoint %s", checkpoint_path)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")

    data_module = P2IDataModule(cfg)
    test_loader = data_module.test_dataloader()
    if test_loader is None:
        raise RuntimeError("Test dataloader is not configured. Ensure data.test exists in the config.")

    dataset = test_loader.dataset
    num_samples = len(dataset)
    if num_samples == 0:
        raise RuntimeError("Test dataset is empty.")

    model_name = cfg["model"]["name"]
    output_path = parsed.output
    if output_path is None:
        save_dir = Path(parsed.model_dir or cfg.get("save_dir", "weights"))
        output_path = save_dir / f"test{model_name}.zarr"

    output_path = Path(output_path)
    if output_path.exists():
        if parsed.overwrite:
            import shutil

            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Output already exists: {output_path}")

    logging.info("Writing predictions to %s", output_path)
    group = zarr.open_group(str(output_path), mode="w")
    group.attrs.update(
        {
            "config_path": str(parsed.config),
            "checkpoint": str(checkpoint_path),
            "model_name": model_name,
            "data_root": cfg.get("data", {}).get("test", {}).get("data_root"),
            "passes": int(parsed.passes),
            "output_scale": float(parsed.output_scale),
        }
    )
    if hasattr(dataset, "video_files"):
        group.attrs["files"] = [str(p) for p in dataset.video_files]

    generator = build_generator(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["generator"] if isinstance(checkpoint, dict) and "generator" in checkpoint else checkpoint
    generator.load_state_dict(state_dict)
    generator.eval()

    passes = max(1, int(parsed.passes))
    log_every = max(1, int(parsed.log_every))
    stride = max(1, int(parsed.stride))
    overlap = max(0, int(parsed.overlap))
    step = max(1, stride - overlap)

    with torch.no_grad():
        for pass_idx in range(passes):
            logging.info("Starting pass %d/%d", pass_idx + 1, passes)
            start = time.time()
            offset = 0
            for batch_idx, batch in enumerate(test_loader):
                frames, masked_frames, masks = prepare_batch(batch, device)
                video_length = frames.shape[1]
                logging.info(
                    "Event %d | frames=%d h=%d w=%d c=%d",
                    offset,
                    video_length,
                    frames.shape[-2],
                    frames.shape[-1],
                    frames.shape[2],
                )

                comp_frames_accum = np.zeros(
                    (video_length, frames.shape[2], frames.shape[3], frames.shape[4]),
                    dtype=np.float32,
                )
                comp_frames_weight = np.zeros((video_length, 1, 1, 1), dtype=np.float32)

                for start_idx in range(0, video_length, step):
                    end_idx = start_idx + stride
                    if end_idx > video_length:
                        pad_len = end_idx - video_length

                        def _pad(x, pad_len=pad_len):
                            return torch.cat([x, x[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)], dim=1)

                        current_frames = _pad(masked_frames[:, start_idx:end_idx])
                        current_masks = _pad(masks[:, start_idx:end_idx])
                        valid_len = video_length - start_idx
                    else:
                        current_frames = masked_frames[:, start_idx:end_idx]
                        current_masks = masks[:, start_idx:end_idx]
                        valid_len = stride

                    with torch.no_grad():
                        output = generator(current_frames, current_masks)
                        preds_np = output.detach().cpu().numpy().astype(np.float32)

                    for i in range(valid_len):
                        idx = start_idx + i
                        pred_frame = preds_np[0, i]
                        comp_frames_accum[idx] += pred_frame
                        comp_frames_weight[idx] += 1.0

                comp_frames = comp_frames_accum / np.maximum(comp_frames_weight, 1e-5)
                comp_frames = comp_frames * float(parsed.output_scale)
                comp_frames = np.clip(comp_frames, 0.0, None)

                event_name = f"event_{offset + 1:02d}"

                if pass_idx == 0:
                    event_ds = group.create_array(
                        event_name,
                        shape=comp_frames.shape,
                        chunks=comp_frames.shape,
                        dtype="float32",
                        overwrite=True,
                    )
                    event_ds[:] = comp_frames
                else:
                    current = group[event_name][:]
                    group[event_name][:] = current + (comp_frames - current) / float(pass_idx + 1)

                offset += 1
                if (batch_idx + 1) % log_every == 0 or offset >= num_samples:
                    elapsed = time.time() - start
                    rate = offset / max(elapsed, 1e-6)
                    logging.info(
                        "Pass %d/%d | %d/%d samples | %.2f samples/sec",
                        pass_idx + 1,
                        passes,
                        offset,
                        num_samples,
                        rate,
                    )

    logging.info("Inference completed. Output saved to %s", output_path)


if __name__ == "__main__":
    main()
