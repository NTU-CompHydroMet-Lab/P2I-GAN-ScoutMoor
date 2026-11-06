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


class DKPhi2DSubsampledMultiRes(nn.Module):
    """
    2D radial Wendland basis with multi-resolution knots, subsampled to keep K small.
    """

    def __init__(self, num_basis_per_level=(10, 19, 37, 73), spacings=None):
        super().__init__()
        self.num_basis_per_level = tuple(num_basis_per_level)
        self.spacings = spacings
        self._cache = {}

    @staticmethod
    def _make_level_grid_knots(H: int, W: int, spacing: int, device):
        ys = torch.arange(0, H, spacing, device=device)
        xs = torch.arange(0, W, spacing, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        knots = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
        return knots

    @staticmethod
    def _subsample_knots_uniform(knots: torch.Tensor, M: int):
        K_full = knots.shape[0]
        if M >= K_full:
            return knots
        idx = torch.linspace(0, K_full - 1, steps=M, device=knots.device)
        idx = torch.round(idx).long().clamp(0, K_full - 1)
        return knots[idx]

    def build(self, H: int, W: int, device):
        if self.spacings is None:
            base = max(1, int(round(min(H, W) / 4)))
            spacings = [max(1, base // (2**i)) for i in range(len(self.num_basis_per_level))]
        else:
            spacings = list(self.spacings)
            assert len(spacings) == len(self.num_basis_per_level)

        knots_all = []
        theta_all = []
        for M, sp in zip(self.num_basis_per_level, spacings, strict=True):

            # In the original DeepKriging formulation, the spatial embedding is constructed 
            # using multi-resolution radial basis functions defined on a dense grid of knots, 
            # where the total number of basis functions grows quadratically with spatial resolution 
            # in two dimensions.

            # However, in dense-grid settings such as image-based interpolation, constructing the full 
            # set of knots at each resolution level leads to prohibitive memory and computational costs.

            # To make the method feasible in this context, we adopt a low-rank approximation by subsampling 
            # a fixed number of knots at each resolution level while preserving the original radial Wendland 
            # basis formulation. This results in a reduced but multi-resolution spatial embedding that maintains 
            # the theoretical structure of DeepKriging, while enabling practical training and inference on dense 
            # spatial grids.
            
            knots_full = self._make_level_grid_knots(H, W, sp, device=device)
            knots_sub = self._subsample_knots_uniform(knots_full, M)
            knots_all.append(knots_sub)

            # In the original DeepKriging framework, the support radius of the Wendland 
            # basis is set to 2.5 times the knot spacing, assuming a dense and regular grid 
            # of knots at each resolution level.

            # In our implementation, due to the use of subsampled knots for computational 
            # feasibility, the effective distance between neighboring knots becomes larger 
            # than that of the original full-grid setting. When retaining the original support 
            # radius, this leads to insufficient overlap between neighboring basis functions, 
            # which manifests as localized point-like artifacts in dense spatial predictions.

            # To compensate for the reduced knot density and restore sufficient overlap among 
            # radial basis functions, we increase the support radius multiplier from 2.5 to 4.0. 
            # This adjustment does not alter the functional form of the basis functions, but 
            # ensures smoother spatial coverage and mitigates knot imprinting artifacts induced 
            # by low-rank approximation.
            
            theta_all.append(torch.full((knots_sub.shape[0],), 4.0 * float(sp), device=device))

        knots = torch.cat(knots_all, dim=0)
        theta = torch.cat(theta_all, dim=0)
        return knots, theta

    def forward(self, H: int, W: int, device, dtype):
        key = (H, W, str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]

        knots, theta = self.build(H, W, device=device)
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([yy, xx], dim=-1).reshape(-1, 2).float()

        d = torch.cdist(grid, knots.float(), p=2)
        d_norm = d / theta.unsqueeze(0)
        phi = wendland_c2(d_norm).to(dtype=dtype)

        self._cache[key] = phi
        return phi


class DKGenerator(BaseNetwork):
    """
    DK (DeepKriging) with spatial basis only (per-frame).
    """

    def __init__(self, config, length=16, init_weights=True,
                 num_basis_space=(10, 19, 37, 73), visible_k=79):
        super().__init__()
        self.config = config
        self.length = length
        self.visible_k = visible_k

        self.dk_phi_space = DKPhi2DSubsampledMultiRes(num_basis_space)
        K_s = sum(num_basis_space)
        feature_dim = K_s + visible_k
        self._mlp = DKMLP(feature_dim=feature_dim)

        self._cache = {}

        if init_weights:
            self.init_weights()

    def _get_phi_space(self, H, W, device, dtype):
        key = ("phi_s", H, W, device, dtype)
        if key not in self._cache:
            self._cache[key] = self.dk_phi_space(H, W, device, dtype)
        return self._cache[key]

    @torch.no_grad()
    def _select_visible_indices(self, m_flat_bt_hw, k=79):
        _, idx = torch.topk(m_flat_bt_hw, k=k, dim=2, largest=True, sorted=False)
        return idx

    def forward(self, masked_frames, masks):
        b, t, c, h, w = masked_frames.size()
        assert t == self.length

        device = masked_frames.device
        dtype = masked_frames.dtype
        HW = h * w

        phi_s = self._get_phi_space(h, w, device, dtype)  # [HW, K_s]
        K_s = phi_s.shape[1]
        phi_s_rep = phi_s.unsqueeze(0).expand(b, HW, K_s)

        x_flat = masked_frames.view(b, t, HW).to(dtype=dtype)
        m_flat = masks.view(b, t, HW).to(dtype=torch.float32)

        idxk = self._select_visible_indices(m_flat, k=self.visible_k)
        z = torch.gather(x_flat, 2, idxk)  # [B,T,K]

        outputs = []
        for ti in range(t):
            z_t = z[:, ti]  # [B,K]
            z_rep = z_t.unsqueeze(1).expand(b, HW, self.visible_k)
            feats = torch.cat([phi_s_rep, z_rep], dim=2)
            y = self._mlp(feats.reshape(b * HW, K_s + self.visible_k))
            outputs.append(y.view(b, h, w))

        return torch.stack(outputs, dim=1).unsqueeze(2)
