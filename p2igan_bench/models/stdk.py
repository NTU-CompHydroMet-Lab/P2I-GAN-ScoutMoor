from __future__ import annotations

import torch
import torch.nn as nn

from p2igan_bench.modules.layer import BaseNetwork


class DKMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 100, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def wendland_c2(d: torch.Tensor) -> torch.Tensor:
    """
    Wendland C2 compactly supported radial basis.
    d: normalized distance, expected >=0.
    """
    out = torch.zeros_like(d)
    mask = d <= 1.0
    dm = d[mask]
    out[mask] = ((1 - dm) ** 6) * (35 * dm**2 + 18 * dm + 3) / 3
    return out


class DKPhi1D(nn.Module):
    """
    1D radial Wendland basis with multi-resolution knots for temporal encoding.
    """

    def __init__(self, num_basis=(10, 19, 37, 73), spacings=None):
        super().__init__()
        self.num_basis = tuple(num_basis)
        self.spacings = spacings
        self._cache = {}

    @staticmethod
    def _make_level_knots(T: int, spacing: int, device):
        return torch.arange(0, T, spacing, device=device).unsqueeze(1)  # [K,1]

    @staticmethod
    def _subsample_knots_uniform(knots: torch.Tensor, M: int):
        K_full = knots.shape[0]
        if M >= K_full:
            return knots
        idx = torch.linspace(0, K_full - 1, steps=M, device=knots.device)
        idx = torch.round(idx).long().clamp(0, K_full - 1)
        return knots[idx]

    def build(self, T: int, device):
        if self.spacings is None:
            base = max(1, int(round(T / 4)))
            spacings = [max(1, base // (2**i)) for i in range(len(self.num_basis))]
        else:
            spacings = list(self.spacings)
            assert len(spacings) == len(self.num_basis)

        knots_all = []
        theta_all = []
        for M, sp in zip(self.num_basis, spacings, strict=True):
            knots_full = self._make_level_knots(T, sp, device=device)
            knots_sub = self._subsample_knots_uniform(knots_full, M)  # [M,1]
            knots_all.append(knots_sub)
            theta_all.append(torch.full((knots_sub.shape[0],), 2.5 * float(sp), device=device))

        knots = torch.cat(knots_all, dim=0)  # [K,1]
        theta = torch.cat(theta_all, dim=0)  # [K]
        return knots, theta

    def forward(self, T: int, device, dtype):
        key = (T, str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]

        knots, theta = self.build(T, device=device)
        grid = torch.arange(T, device=device).unsqueeze(1).float()  # [T,1]
        d = torch.cdist(grid, knots.float(), p=2)  # [T,K]
        d_norm = d / theta.unsqueeze(0)
        phi = wendland_c2(d_norm).to(dtype=dtype)  # [T,K]
        self._cache[key] = phi
        return phi


class STDKGenerator(BaseNetwork):
    """
    STDK（Spatial–Temporal DeepKriging）

    - φ_s : 2D radial Wendland multi-res (DeepKriging-style)
    - φ_t : 1D DKPhi1D
    - Z   : concat over T of 79 visible points
    - MLP : shared (100–100–100–1)
    """

    def __init__(self, config, length=16, init_weights=True,
                 num_basis_space=(10, 19, 37, 73),
                 num_basis_time=(10, 19, 37, 73)):
        super().__init__()
        self.config = config
        self.length = length

        # spatial / temporal basis
        self.dk_phi_space = DKPhi2DSubsampledMultiRes(num_basis_space)
        self.dk_phi_time  = DKPhi1D(num_basis=num_basis_time)

        K_s = sum(num_basis_space)   # 139
        # Temporal basis count depends on T when subsampling; compute actual K_t for fixed length.
        knots_t, _ = self.dk_phi_time.build(self.length, device=torch.device("cpu"))
        K_t = knots_t.shape[0]

        feature_dim = K_s + K_t + self.length * 79
        self._mlp = DKMLP(feature_dim=feature_dim)

        self._cache = {}

        if init_weights:
            self.init_weights()

    def _get_phi_space(self, H, W, device, dtype):
        key = ("phi_s", H, W, device, dtype)
        if key not in self._cache:
            self._cache[key] = self.dk_phi_space(H, W, device, dtype)
        return self._cache[key]      # [HW, K_s]

    def _get_phi_time(self, T, device, dtype):
        key = ("phi_t", T, device, dtype)
        if key not in self._cache:
            self._cache[key] = self.dk_phi_time(T, device=device, dtype=dtype)
        return self._cache[key]      # [T, K_t]

    @torch.no_grad()
    def _select_visible_indices(self, m_flat_bt_hw, k=79):
        _, idx = torch.topk(m_flat_bt_hw, k=k, dim=2,
                            largest=True, sorted=False)
        return idx

    def forward(self, masked_frames, masks):
        b, t, c, h, w = masked_frames.size()
        assert t == self.length

        device = masked_frames.device
        dtype  = masked_frames.dtype
        HW = h * w

        # φ_s, φ_t
        phi_s = self._get_phi_space(h, w, device, dtype)   # [HW, K_s]
        phi_t = self._get_phi_time(t, device, dtype)       # [T,  K_t]

        K_s = phi_s.shape[1]
        K_t = phi_t.shape[1]

        # expand
        phi_s_rep = phi_s.unsqueeze(0).expand(t, HW, K_s)
        phi_t_rep = phi_t.unsqueeze(1).expand(t, HW, K_t)
        phi_s_rep = phi_s_rep.unsqueeze(0).expand(b, t, HW, K_s)
        phi_t_rep = phi_t_rep.unsqueeze(0).expand(b, t, HW, K_t)

        # visible points
        x_flat = masked_frames.view(b, t, HW).to(dtype=dtype)
        m_flat = masks.view(b, t, HW).to(dtype=torch.float32)

        idx79 = self._select_visible_indices(m_flat, k=79)
        z79   = torch.gather(x_flat, 2, idx79)             # [B,T,79]
        z_seq = z79.reshape(b, t * 79)

        z_rep = z_seq.unsqueeze(1).unsqueeze(2).expand(
            b, t, HW, t * 79
        )

        feats = torch.cat([phi_s_rep, phi_t_rep, z_rep], dim=3)

        y = self._mlp(feats.reshape(b * t * HW, K_s + K_t + t * 79))
        y = y.view(b, t, h, w)
        return y.unsqueeze(2)

class DKPhi2DSubsampledMultiRes(nn.Module):
    """
    2D radial Wendland basis with multi-resolution knots, but subsampled to keep K small.

    num_basis_per_level: e.g. (10, 19, 37, 73)  -> total K = 139
    spacings: optional; if None, auto-generate from H/W and level count
    theta = 2.5 * spacing per level (DeepKriging setting)
    """
    def __init__(self, num_basis_per_level=(10, 19, 37, 73), spacings=None):
        super().__init__()
        self.num_basis_per_level = tuple(num_basis_per_level)
        self.spacings = spacings  # can be None; decided at build time

        # cached per (H,W,device,dtype)
        self._cache = {}

    @staticmethod
    def _make_level_grid_knots(H: int, W: int, spacing: int, device):
        ys = torch.arange(0, H, spacing, device=device)
        xs = torch.arange(0, W, spacing, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        knots = torch.stack([yy, xx], dim=-1).reshape(-1, 2)  # [K_full,2]
        return knots

    @staticmethod
    def _subsample_knots_uniform(knots: torch.Tensor, M: int):
        """
        Deterministic uniform subsampling from a grid: pick roughly evenly spaced indices.
        knots: [K_full,2]
        returns [M,2]
        """
        K_full = knots.shape[0]
        if M >= K_full:
            return knots
        # evenly spaced indices
        idx = torch.linspace(0, K_full - 1, steps=M, device=knots.device)
        idx = torch.round(idx).long().clamp(0, K_full - 1)
        return knots[idx]

    def build(self, H: int, W: int, device):
        """
        Build knots and per-knot theta (level-dependent). Kept on device.
        """
        # choose spacings if not provided:
        # coarse -> fine; ensure >=1
        if self.spacings is None:
            # heuristic: start at ~H/4 then halve each level, clamp >=1
            base = max(1, int(round(min(H, W) / 4)))
            spacings = [max(1, base // (2**i)) for i in range(len(self.num_basis_per_level))]
        else:
            spacings = list(self.spacings)
            assert len(spacings) == len(self.num_basis_per_level)

        knots_all = []
        theta_all = []
        for M, sp in zip(self.num_basis_per_level, spacings, strict=True):
            knots_full = self._make_level_grid_knots(H, W, sp, device=device)
            knots_sub  = self._subsample_knots_uniform(knots_full, M)  # [M,2]
            knots_all.append(knots_sub)
            theta_all.append(torch.full((knots_sub.shape[0],), 4.0 * float(sp), device=device))

        knots = torch.cat(knots_all, dim=0)                # [K,2]
        theta = torch.cat(theta_all, dim=0)                # [K]
        return knots, theta

    def forward(self, H: int, W: int, device, dtype):
        key = (H, W, str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]

        knots, theta = self.build(H, W, device=device)     # knots [K,2], theta [K]

        # grid [HW,2]
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([yy, xx], dim=-1).reshape(-1, 2).float()  # [HW,2]

        # distances [HW,K]
        d = torch.cdist(grid, knots.float(), p=2)          # Euclidean

        # normalize by per-knot theta
        d_norm = d / theta.unsqueeze(0)                    # broadcast [HW,K]

        # Wendland (compact support happens inside)
        phi = wendland_c2(d_norm).to(dtype=dtype)          # [HW,K]

        self._cache[key] = phi
        return phi


InpaintGenerator = STDKGenerator
