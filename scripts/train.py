from __future__ import annotations

import argparse
import contextlib
import logging
import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from p2igan_bench.config import load_config, validate_config
from p2igan_bench.data.dataloader import P2IDataModule
from p2igan_bench.models import build_discriminator, build_generator
from p2igan_bench.modules import ReconstructionLoss, gan_loss
from p2igan_bench.run_logger import RunLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P2I-GAN benchmark model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("p2igan_bench/config/p2igan_baseline.json"),
        help="Path to JSON/YAML config file.",
    )
    parser.add_argument("--save-dir", type=Path, default=None, help="Run directory; overrides save_dir in the config")
    parser.add_argument("--experiment-name", type=str, default=None, help="W&B group")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    parser.add_argument(
        "--run-validation",
        dest="run_validation",
        action="store_true",
        help="Enable running validation each epoch (overrides config).",
    )
    parser.add_argument(
        "--skip-validation",
        dest="run_validation",
        action="store_false",
        help="Skip validation during training (overrides config).",
    )
    parser.set_defaults(run_validation=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: dict[str, Any]):
        cfg = validate_config(cfg)
        self.cfg = cfg
        seed_everything(cfg.get("seed", 42))
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        logging.info("Initializing data module...")
        self.data_module = P2IDataModule(cfg)
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        train_cfg = cfg.get("train", {})
        self.run_validation = bool(train_cfg.get("use_validation", True))
        logging.info(
            "Data loaders ready | train=%s, val=%s",
            len(self.train_loader) if self.train_loader is not None else 0,
            len(self.val_loader) if self.val_loader is not None else 0,
        )
        self.train_steps_per_epoch = max(1, len(self.train_loader))

        logging.info("Building models...")
        self.generator = build_generator(cfg).to(self.device)
        self.discriminator = build_discriminator(cfg).to(self.device) if cfg["loss"].get("use_gan", 0) else None

        opt_cfg = cfg["train"]["optimizer"]
        self.opt_g = Adam(
            self.generator.parameters(),
            lr=opt_cfg["lr"],
            betas=(opt_cfg.get("beta1", 0.0), opt_cfg.get("beta2", 0.99)),
        )
        self.opt_d = None
        if self.discriminator is not None:
            self.opt_d = Adam(
                self.discriminator.parameters(),
                lr=opt_cfg["lr"],
                betas=(opt_cfg.get("beta1", 0.0), opt_cfg.get("beta2", 0.99)),
            )

        self.rec_loss = ReconstructionLoss(
            k1_alpha=cfg["loss"].get("k1_weight", 0.0),
            use_pm=bool(cfg["loss"].get("use_pm", 0)),
            pm_weight=float(cfg["loss"].get("pm_weight", 10.0)),
        )
        self.use_gan = bool(cfg["loss"].get("use_gan", 0))
        self.gan_loss_type = cfg["loss"].get("gan_loss", "hinge")
        self.gan_real_label = cfg["loss"].get("target_real_label", 1.0)
        self.gan_fake_label = cfg["loss"].get("target_fake_label", 0.0)

        self.save_dir = Path(cfg.get("save_dir", "weights"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = cfg["train"]
        self.log_every = int(train_cfg.get("log_step", 100))
        self.global_step = 0
        self.max_steps = train_cfg.get("iterations")
        self.max_epochs = train_cfg.get("max_epochs")
        if self.max_epochs is None:
            loader_len = max(1, len(self.train_loader))
            if self.max_steps:
                self.max_epochs = math.ceil(self.max_steps / loader_len)
            else:
                self.max_epochs = train_cfg.get("niter", 1)
        if self.max_steps is None:
            self.max_steps = self.max_epochs * max(1, len(self.train_loader))

        self.best_val = float("inf")
        self.last_rec_loss = 0.0
        self.last_adv_loss = 0.0
        self.last_dis_loss = 0.0
        viz_cfg = cfg.get("viz", {})
        self.viz_scale = str(viz_cfg.get("scale", "gt_pred")).lower()
        self.viz_vmin = viz_cfg.get("vmin")
        self.viz_vmax = viz_cfg.get("vmax")

        self.start_epoch = 1
        self.wandb_run_id = None
        self.logger = RunLogger(False, "", None, None, None, self.save_dir, {})
        self._maybe_resume()

        torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", False))
        self.amp = contextlib.nullcontext
        if train_cfg.get("amp", "off") == "bf16":
            self.amp = lambda: torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        self.gen = self.generator
        self.dis = self.discriminator
        if train_cfg.get("compile", False):
            # InputBlock's data-dependent kNN fails to compile under bf16 autocast.
            if hasattr(self.generator, "input"):
                self.generator.input.forward = torch.compiler.disable(self.generator.input.forward)
            self.gen = torch.compile(self.generator)
            if self.discriminator is not None:
                self.dis = torch.compile(self.discriminator)

    def _maybe_resume(self) -> None:
        """Continue from ``save_dir/latest.pt`` if it exists; a fresh run needs a new ``save_dir``."""
        path = self.save_dir / "latest.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(state["generator"])
        self.opt_g.load_state_dict(state["optimizer_g"])
        if self.discriminator is not None:
            self.discriminator.load_state_dict(state["discriminator"])
            self.opt_d.load_state_dict(state["optimizer_d"])
        self.global_step = int(state["global_step"])
        self.start_epoch = int(state["epoch"]) + 1
        self.best_val = float(state.get("best_val", float("inf")))
        self.wandb_run_id = state.get("wandb_run_id")
        logging.info(
            "Resumed from %s: epoch %d done, global_step %d, best_val %.4f",
            path,
            self.start_epoch - 1,
            self.global_step,
            self.best_val,
        )

    def _gpu_stats(self) -> dict[str, float]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {}
        alloc = torch.cuda.memory_allocated(self.device) / 1e6
        reserved = torch.cuda.memory_reserved(self.device) / 1e6
        logging.info("GPU mem | allocated=%.1fMB reserved=%.1fMB", alloc, reserved)
        return {"perf/gpu_allocated_mb": alloc, "perf/gpu_reserved_mb": reserved}

    def train(self) -> None:
        if self.global_step >= self.max_steps:
            logging.info("Already at max steps (%d); nothing to train.", self.max_steps)
            return
        self.logger = RunLogger(
            enabled=bool(self.cfg.get("log_to_wandb", False)),
            project=self.cfg.get("wandb_project", "p2igan-scoutmoor"),
            name=self.cfg.get("run_name"),
            group=self.cfg.get("experiment_name"),
            run_id=self.wandb_run_id,
            out_dir=self.save_dir,
            config={
                **self.cfg,
                "train_samples": len(self.train_loader.dataset),
                "val_samples": len(self.val_loader.dataset) if self.val_loader is not None else 0,
            },
        )
        self.wandb_run_id = self.logger.run_id
        try:
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                logging.info("Epoch %d/%d starting...", epoch, self.max_epochs)
                train_loss = self._train_one_epoch(epoch)
                metrics = {"epoch/train_loss": train_loss, "epoch/rec_loss": self.last_rec_loss}
                if self.use_gan:
                    metrics.update({"epoch/adv_loss": self.last_adv_loss, "epoch/dis_loss": self.last_dis_loss})
                logging.info("Epoch %d completed | train_loss=%.4f | global_step=%d", epoch, train_loss, self.global_step)
                images = {
                    "examples/train": self._log_examples(
                        self.train_loader, prefix="train", epoch=epoch, max_batches=1, samples_per_batch=1
                    )
                }

                validated = self.run_validation and self.val_loader is not None
                if validated:
                    logging.info("Running validation...")
                    val_loss = self._evaluate_rec_loss(self.val_loader)
                    metrics["val/loss"] = val_loss
                    logging.info("Validation done | val_loss=%.4f", val_loss)

                self._save_checkpoint(self.save_dir / "latest.pt", epoch)
                if validated and val_loss < self.best_val:
                    self.best_val = val_loss
                    checkpoint = self.save_dir / "best.pt"
                    self._save_checkpoint(checkpoint, epoch)
                    logging.info("New best model saved at %s (val_loss=%.4f)", checkpoint, self.best_val)
                if validated:
                    metrics["val/best_loss"] = self.best_val
                images["examples/val"] = self._log_examples(self.val_loader, prefix="val", epoch=epoch)
                self.logger.log_epoch(epoch, metrics, images)

                if self.global_step >= self.max_steps:
                    logging.info("Reached max steps (%d). Stopping training.", self.max_steps)
                    break
        finally:
            self.logger.finish()

    def _train_one_epoch(self, epoch: int) -> float:
        self.generator.train()
        if self.discriminator is not None:
            self.discriminator.train()

        running_loss = 0.0
        rec_running = 0.0
        adv_running = 0.0
        dis_running = 0.0
        steps = 0
        progress_interval = max(1, self.train_steps_per_epoch // 20)

        for batch in self.train_loader:

            frames, masked_frames, masks = self._prepare_batch(batch)
            with self.amp():
                preds = self.gen(masked_frames, masks)
            preds = preds.float()
            loss_g, loss_dict = self.rec_loss(preds, frames, masks)
            rec_loss_val = float(loss_g.detach())
            rec_running += rec_loss_val
            adv_loss_val = 0.0

            if steps == 0:
                logging.info(
                    "Batch shapes | frames=%s masked=%s masks=%s preds=%s",
                    tuple(frames.shape),
                    tuple(masked_frames.shape),
                    tuple(masks.shape),
                    tuple(preds.shape),
                )

            if self.use_gan and self.discriminator is not None:

                for p in self.discriminator.parameters():
                    p.requires_grad_(True)

                with self.amp():
                    logits_fake = self.dis(preds.detach()).float()
                    logits_real = self.dis(frames).float()
                loss_d = (
                    gan_loss(
                        logits_real,
                        True,
                        loss_type=self.gan_loss_type,
                        is_disc=True,
                        target_real_label=self.gan_real_label,
                        target_fake_label=self.gan_fake_label,
                    )
                    + gan_loss(
                        logits_fake,
                        False,
                        loss_type=self.gan_loss_type,
                        is_disc=True,
                        target_real_label=self.gan_real_label,
                        target_fake_label=self.gan_fake_label,
                    )
                ) * 0.5
                if steps == 0:
                    logging.info(
                        "Disc step | logits_real=%s logits_fake=%s loss_d=%.4f",
                        tuple(logits_real.shape),
                        tuple(logits_fake.shape),
                        float(loss_d.detach()),
                    )
                self.opt_d.zero_grad()
                loss_d.backward()
                if steps == 0:
                    logging.info("Disc backward done (step=%d)", self.global_step)
                self.opt_d.step()

                for p in self.discriminator.parameters():
                    p.requires_grad_(False)

                with self.amp():
                    logits_fake_for_g = self.dis(preds).float()
                adv_loss = gan_loss(
                    logits_fake_for_g,
                    True,
                    loss_type=self.gan_loss_type,
                    is_disc=False,
                    target_real_label=self.gan_real_label,
                    target_fake_label=self.gan_fake_label,
                ) * self.cfg["loss"].get("adversarial_weight", 0.01)
                loss_g = loss_g + adv_loss
                adv_loss_val = float(adv_loss.detach())
            else:
                loss_d = None

            self.opt_g.zero_grad()
            loss_g.backward()
            self.opt_g.step()

            if self.use_gan and self.discriminator is not None:
                for p in self.discriminator.parameters():
                    p.requires_grad_(True)
                    
            running_loss += float(loss_g.detach())
            adv_running += adv_loss_val
            dis_running += float(loss_d.detach()) if loss_d is not None else 0.0
            steps += 1
            self.global_step += 1

            if steps % progress_interval == 0 or steps == self.train_steps_per_epoch:
                pct = steps / self.train_steps_per_epoch
                bar_len = 20
                filled = int(bar_len * pct)
                bar = "|" * filled + "." * (bar_len - filled)
                logging.info(
                    "Epoch %d/%d |%s| %.1f%% (step %d/%d)",
                    epoch,
                    self.max_epochs,
                    bar,
                    pct * 100,
                    steps,
                    self.train_steps_per_epoch,
                )

            if self.global_step % self.log_every == 0:
                step_metrics = {"train/loss": float(loss_g.detach()), "train/rec_loss": rec_loss_val}
                if self.use_gan:
                    step_metrics["train/adv_loss"] = adv_loss_val
                    if loss_d is not None:
                        step_metrics["train/dis_loss"] = float(loss_d.detach())
                step_metrics.update({f"train/{key}": value for key, value in loss_dict.items()})
                logging.info(
                    "Epoch %d | step %d/%d | loss=%.4f",
                    epoch,
                    self.global_step,
                    self.max_steps,
                    float(loss_g.detach()),
                )
                self.logger.log_step(self.global_step, {**step_metrics, **self._gpu_stats()})

            if self.global_step >= self.max_steps:
                break

        self.last_rec_loss = rec_running / max(1, steps)
        self.last_adv_loss = adv_running / max(1, steps)
        self.last_dis_loss = dis_running / max(1, steps)
        return running_loss / max(1, steps)

    def _evaluate_rec_loss(self, loader: DataLoader | None) -> float:
        if loader is None:
            return 0.0
        self.generator.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for batch in loader:
                frames, masked_frames, masks = self._prepare_batch(batch)
                with self.amp():
                    preds = self.gen(masked_frames, masks)
                loss, _ = self.rec_loss(preds.float(), frames, masks)
                total_loss += float(loss.detach())
                batches += 1
        return total_loss / max(1, batches)

    def _log_examples(
        self,
        loader: DataLoader | None,
        prefix: str,
        epoch: int,
        max_batches: int = 5,
        samples_per_batch: int = 1,
    ) -> list[Path]:
        paths: list[Path] = []
        if loader is None:
            return paths
        self.generator.eval()
        save_dir = self.save_dir / "artifacts"
        save_dir.mkdir(parents=True, exist_ok=True)

        for b_idx, batch in zip(range(max_batches), loader, strict=False):
            frames, masked_frames, masks = self._prepare_batch(batch)
            with torch.no_grad():
                preds = self.generator(masked_frames, masks)

            num = min(samples_per_batch, frames.size(0))
            for idx in range(num):
                gt = frames[idx]  # [T, C, H, W]
                pd = preds[idx].clamp(0, 1)

                gt_min = float(gt.min().detach())
                gt_max = float(gt.max().detach())
                pd_min = float(pd.min().detach())
                pd_max = float(pd.max().detach())

                if self.viz_scale == "fixed" and self.viz_vmin is not None and self.viz_vmax is not None:
                    vmin, vmax = float(self.viz_vmin), float(self.viz_vmax)
                elif self.viz_scale == "gt":
                    vmin, vmax = gt_min, gt_max
                else:
                    vmin, vmax = min(gt_min, pd_min), max(gt_max, pd_max)

                gt_color, gt_stats = self._colorize_sequence(gt, vmin=vmin, vmax=vmax)
                pd_color, pd_stats = self._colorize_sequence(pd, vmin=vmin, vmax=vmax)

                seq = torch.cat([gt_color, pd_color], dim=0)
                grid = make_grid(seq, nrow=gt.shape[0], padding=2)
                out_path = save_dir / f"{prefix}_epoch{epoch}_batch{b_idx}_ex{idx}.png"
                save_image(grid, out_path)
                self._annotate_image(
                    out_path,
                    f"GT min/mean/max: {gt_stats[0]:.3f}/{gt_stats[1]:.3f}/{gt_stats[2]:.3f} | "
                    f"Pred min/mean/max: {pd_stats[0]:.3f}/{pd_stats[1]:.3f}/{pd_stats[2]:.3f} | cmap=viridis",
                )
                paths.append(out_path)
        return paths

    def _colorize_sequence(self, seq: torch.Tensor, vmin: float | None = None, vmax: float | None = None):
        # seq: [T, C, H, W]
        t, c, h, w = seq.shape
        stats = (
            float(seq.min().detach()),
            float(seq.mean().detach()),
            float(seq.max().detach()),
        )
        out_frames = []
        cmap = matplotlib.colormaps["viridis"]
        if vmin is None or vmax is None:
            vmin, vmax = stats[0], stats[2]
        for i in range(t):
            frame = seq[i]
            if frame.shape[0] == 1:
                frame = frame[0]
            else:
                frame = frame.mean(dim=0)
            np_frame = frame.detach().cpu().numpy()
            norm = (np_frame - vmin) / (vmax - vmin + 1e-6)
            colored = cmap(norm)[..., :3]  # HWC
            out_frames.append(torch.from_numpy(colored).permute(2, 0, 1))
        return torch.stack(out_frames, dim=0), stats

    def _annotate_image(self, path: Path, text: str) -> None:
        try:
            img = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((5, 5), text, fill=(255, 255, 255), font=font)
            img.save(path)
        except Exception:
            return

    def _prepare_batch(self, batch: Iterable[torch.Tensor]):
        frames, masked_frames, masks = batch
        frames = frames.permute(0, 1, 4, 2, 3).to(self.device, non_blocking=True)
        masked_frames = masked_frames.permute(0, 1, 4, 2, 3).to(self.device, non_blocking=True)
        masks = masks.permute(0, 1, 4, 2, 3).to(self.device, non_blocking=True)
        return frames, masked_frames, masks

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "generator": self.generator.state_dict(),
            "optimizer_g": self.opt_g.state_dict(),
            "best_val": self.best_val,
            "wandb_run_id": self.wandb_run_id,
        }
        if self.discriminator is not None and self.opt_d is not None:
            state["discriminator"] = self.discriminator.state_dict()
            state["optimizer_d"] = self.opt_d.state_dict()
        torch.save(state, path)


def main(args: argparse.Namespace | None = None) -> None:
    parsed = args or parse_args()
    logging.info("Loading config from %s", parsed.config)
    config = load_config(parsed.config)
    train_cfg = config.setdefault("train", {})
    if getattr(parsed, "save_dir", None):
        config["save_dir"] = str(parsed.save_dir)
    if parsed.experiment_name:
        config["experiment_name"] = parsed.experiment_name
    if parsed.run_name:
        config["run_name"] = parsed.run_name
    if parsed.run_validation is not None:
        train_cfg["use_validation"] = bool(parsed.run_validation)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main(args)
