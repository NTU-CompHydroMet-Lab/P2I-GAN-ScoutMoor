"""Rain-rate codec for the model scale.

P2I scale: ``v = 48 * log10(R / 0.036)`` clipped to 0-255, model input ``x = v / 255``. Rain at or
below 0.036 mm/h maps to 0. Decoding returns 0 below one encoding step (``v < 1``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RainCodec:
    name: str = "p2i_log"
    r0: float = 0.036
    v_per_decade: float = 48.0
    v_max: float = 255.0

    def encode(self, rain):
        """mm/h to the model scale in [0, 1]; works on tensors and arrays."""
        if isinstance(rain, torch.Tensor):
            r = torch.clamp(rain.float(), min=1e-6)
            v = self.v_per_decade * torch.log10(r / self.r0)
            return v.clamp(0.0, self.v_max) / self.v_max
        r = np.maximum(np.asarray(rain, dtype=np.float64), 1e-6)
        v = self.v_per_decade * np.log10(r / self.r0)
        return np.clip(v, 0.0, self.v_max) / self.v_max

    def decode(self, x):
        """Model scale to mm/h."""
        if isinstance(x, torch.Tensor):
            v = x.float() * self.v_max
            rain = self.r0 * torch.pow(10.0, v / self.v_per_decade)
            return torch.where(v >= 1.0, rain, torch.zeros_like(rain))
        v = np.asarray(x, dtype=np.float64) * self.v_max
        return np.where(v >= 1.0, self.r0 * 10.0 ** (v / self.v_per_decade), 0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> RainCodec:
        return cls(**data) if data else cls()


P2I_CODEC = RainCodec()
