"""Experiment logging to Weights & Biases; every call is a no-op when logging is off."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any


def git_commit() -> str:
    """Short HEAD hash, suffixed ``-dirty`` when the tree has uncommitted changes."""
    repo = Path(__file__).resolve().parent
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=repo)
        if not head.stdout.strip():
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo)
        return f"{head.stdout.strip()}-dirty" if dirty.stdout.strip() else head.stdout.strip()
    except OSError:
        return "unknown"


class RunLogger:
    def __init__(
        self,
        enabled: bool,
        project: str,
        name: str | None,
        group: str | None,
        run_id: str | None,
        out_dir: Path,
        config: dict[str, Any],
    ):
        self.run = None
        self.run_id = run_id
        if not enabled:
            return
        try:
            import wandb

            # A run id carried in the checkpoint lets a resumed job continue the same run.
            self.run = wandb.init(
                project=project,
                name=name,
                group=group,
                id=run_id,
                resume="allow",
                config={**config, "git_commit": git_commit()},
                dir=str(out_dir),
            )
            self.run_id = self.run.id
            self.run.define_metric("global_step")
            self.run.define_metric("epoch")
            for pattern in ("train/*", "perf/*"):
                self.run.define_metric(pattern, step_metric="global_step")
            for pattern in ("epoch/*", "val/*", "examples/*"):
                self.run.define_metric(pattern, step_metric="epoch")
            self._image = wandb.Image
        except Exception as exc:
            logging.warning("W&B logging disabled: %s", exc)
            self.run = None

    def log_step(self, global_step: int, metrics: dict[str, float]) -> None:
        if self.run is not None:
            self.run.log({"global_step": global_step, **metrics})

    def log_epoch(self, epoch: int, metrics: dict[str, float], images: dict[str, list[Path]] | None = None) -> None:
        if self.run is None:
            return
        payload: dict[str, Any] = {"epoch": epoch, **metrics}
        for key, paths in (images or {}).items():
            payload[key] = [self._image(str(p)) for p in paths]
        self.run.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
