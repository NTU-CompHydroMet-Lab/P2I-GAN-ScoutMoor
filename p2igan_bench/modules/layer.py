"""Reusable layers and utilities for the P2I-GAN models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deconv_pytorch import DOConv2d


class BaseNetwork(nn.Module):
    """Base class that provides weight initialization helpers."""

    def __init__(self):
        super().__init__()

    def init_weights(self, init_type: str = "kaiming", gain: float = 0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True, transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super().__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock_do(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x):
        return self.main(x) + x


class DownsampleDuplicateChannels(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.t = length

    def forward(self, x):
        b, c, h, w = x.size()
        if c % self.t != 0:
            raise ValueError(f"channels {c} must be divisible by {self.t}")

        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(b * self.t, c // self.t, h // 2, w // 2)
        x = x.repeat_interleave(2, dim=1)
        x = x.view(b, self.t * (c // self.t) * 2, h // 2, w // 2)
        return x


_GRID_CACHE: dict[tuple[int, int, int, str, torch.dtype], torch.Tensor] = {}


def _get_grid_points(D, H, W, device, dtype):
    key = (D, H, W, str(device), dtype)
    gp = _GRID_CACHE.get(key)
    if gp is None:
        z = torch.linspace(0, 1, D, device=device, dtype=dtype)
        y = torch.linspace(0, 1, H, device=device, dtype=dtype)
        x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        gz, gy, gx = torch.meshgrid(z, y, x, indexing="ij")
        gp = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).contiguous()
        _GRID_CACHE[key] = gp
    return gp


def idw_3d_knn(points_xyz, values, out_shape, k=8, rho=2.0, tau=1e-3, chunk=16384, device=None, dtype=None, use_amp=True):
    if device is None:
        device = points_xyz.device
    if dtype is None:
        dtype = torch.float32

    points_xyz = points_xyz.to(device=device, dtype=dtype, non_blocking=True)
    values = values.to(device=device, dtype=dtype, non_blocking=True)

    D, H, W = out_shape
    gp_all = _get_grid_points(D, H, W, device, dtype)
    Q = gp_all.shape[0]
    out_vals = torch.empty(Q, device=device, dtype=dtype)

    use_cuda_amp = use_amp and device.type == "cuda"

    with torch.amp.autocast("cuda", enabled=use_cuda_amp):
        for start in range(0, Q, chunk):
            end = min(start + chunk, Q)
            gp = gp_all[start:end]

            dists = torch.cdist(gp, points_xyz)
            d_k, idx_k = torch.topk(dists, k, dim=1, largest=False)
            vals_k = values[idx_k]

            if abs(rho - 2.0) < 1e-6:
                inv = 1.0 / (d_k + tau)
                w = inv * inv
            else:
                w = 1.0 / (d_k + tau).pow(rho)

            w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
            out_vals[start:end] = (vals_k * w).sum(dim=1)

    return out_vals.reshape(D, H, W)


class AttentionBlock(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=1)

    def forward(self, x):
        gate = self.conv(x)
        out = x + x * gate
        return F.relu(out)


class InputBlock(nn.Module):
    def __init__(self, depth=4, k=4, rho=2.0, tau=0.05, chunk=1000):
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(16) for _ in range(depth)])
        self.k = k
        self.rho = rho
        self.tau = tau
        self.chunk = chunk

    def forward(self, input, mask):
        B, D, H, W = input.shape
        x = input.permute(0, 2, 3, 1).contiguous()
        x = x.view(B * H * W, D, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        outputs = []
        for b in range(B):
            frame_proc = x[b]
            mask_b = mask[b]

            idx = torch.nonzero(mask_b > 0, as_tuple=True)
            if len(idx) != 3 or idx[0].numel() == 0:
                outputs.append(torch.zeros(D, H, W, device=input.device).unsqueeze(0))
                continue

            tz, ty, tx = idx
            points = torch.stack(
                [
                    tx.float() / max(W - 1, 1),
                    ty.float() / max(H - 1, 1),
                    tz.float() / max(D - 1, 1),
                ],
                dim=-1,
            )

            values = frame_proc[tz, ty, tx]

            out = idw_3d_knn(
                points,
                values,
                (D, H, W),
                k=self.k,
                rho=self.rho,
                tau=self.tau,
                chunk=self.chunk,
                device=input.device,
                dtype=torch.float16 if input.device.type == "cuda" else torch.float32,
                use_amp=True,
            )

            outputs.append(out.unsqueeze(0))

        return torch.cat(outputs, dim=0)


class UPPos(nn.Module):
    def __init__(self, in_ch, out_ch, T, H, W):
        super().__init__()
        self.T = T
        self.pos = nn.Parameter(torch.zeros(1, 1, H, W))
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, _, _ = x.shape
        x = self.up(x)
        pos = 2 * torch.sigmoid(self.pos) - 1
        pos = pos.repeat(1, C, 1, 1)
        x = x + x * pos
        x = self.proj(x)
        return F.relu(x, inplace=True)

class UP(nn.Module):
    def __init__(self, in_ch, out_ch, T, H, W):
        super().__init__()
        self.T = T
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, _, _ = x.shape
        x = self.up(x)
        x = self.proj(x)
        return F.relu(x, inplace=True)



def C2(cin, cout, k=3, s=1, p=1):
    return nn.utils.spectral_norm(nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p))


def C3(cin, cout, kt=3, ks=3, st=(1, 1, 1), pt=(1, 1, 1)):
    return nn.utils.spectral_norm(nn.Conv3d(cin, cout, kernel_size=(kt, ks, ks), stride=st, padding=pt))


__all__ = [
    "BaseNetwork",
    "BasicConv_do",
    "ResBlock_do",
    "DownsampleDuplicateChannels",
    "InputBlock",
    "UPPos",
    "UP",
    "C2",
    "C3",
]
