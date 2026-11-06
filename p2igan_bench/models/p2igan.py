"""Warp U-Net GAN generator/discriminator for precipitation nowcasting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2igan_bench.modules.layer import (
    C2,
    C3,
    BaseNetwork,
    BasicConv_do,
    DownsampleDuplicateChannels,
    InputBlock,
    ResBlock_do,
    UPPos,
)


class P2IGenerator(BaseNetwork):
    def __init__(self, config, length: int = 16, num_res: int = 4, init_weights: bool = True):
        super().__init__()

        data_cfg = config["data"]["train"]
        self.H = data_cfg["h"]
        self.W = data_cfg["w"]
        length = data_cfg.get("sample_length", length)
        self.length = length

        ResBlock = ResBlock_do
        ConvBlock = BasicConv_do

        self.input = InputBlock(depth=2, k=4, rho=2.0, tau=0.05, chunk=16384)

        base_channel = 64
        self.Decoder = nn.ModuleList(
            [
                EBlock(base_channel, num_res, ResBlock=ResBlock),
                EBlock(base_channel * 2, num_res, ResBlock=ResBlock),
                EBlock(base_channel * 4, num_res, ResBlock=ResBlock),
                EBlock(base_channel * 8, num_res, ResBlock=ResBlock),
            ]
        )

        self.ConvsOut = nn.ModuleList([ConvBlock(base_channel, length, kernel_size=1, relu=False, stride=1, groups=4)])

        self.UP = nn.ModuleList(
            [
                UPPos(in_ch=base_channel * 2, out_ch=base_channel, H=self.H, W=self.W, T=length),
                UPPos(in_ch=base_channel * 4, out_ch=base_channel * 2, H=self.H // 2, W=self.W // 2, T=length),
                UPPos(in_ch=base_channel * 8, out_ch=base_channel * 4, H=self.H // 4, W=self.W // 4, T=length),
            ]
        )

        self.Convsin = nn.ModuleList([ConvBlock(length, base_channel, kernel_size=3, relu=False, stride=1, groups=4)])
        self.downsample = DownsampleDuplicateChannels(length=length)

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames, masks):
        b, t, c, h, w = masked_frames.size()
        masked_frames = masked_frames.view(b, c * t, h, w)
        masks = masks.view(b, c * t, h, w)

        x = self.input(masked_frames, masks).float()

        x_ = self.Convsin[0](x) + x.repeat_interleave(4, dim=1)
        x_ = x_.view(b, t * c * 4, h, w)
        x_2 = self.downsample(x_)
        x_4 = self.downsample(x_2)
        x_8 = self.downsample(x_4)

        use_amp = masked_frames.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            x_2 = x_2.float()
            x_4 = x_4.float()
            x_8 = x_8.float()

        res1 = self.Decoder[3](x_8)
        res1 = res1 + torch.zeros_like(x_8, requires_grad=True)
        res1 = self.UP[2](res1)

        x_4 = x_4 + res1
        x_4 = x_4 + torch.zeros_like(x_4, requires_grad=True)
        res2 = self.Decoder[2](x_4)
        res2 = self.UP[1](res2)

        x_2 = res2
        x_2 = x_2 + torch.zeros_like(x_2, requires_grad=True)
        res3 = self.Decoder[1](x_2)
        res3 = self.UP[0](res3)

        x_ = res3
        x_ = x_ + torch.zeros_like(x_, requires_grad=True)
        z = self.Decoder[0](x_)
        z = z + torch.zeros_like(z, requires_grad=True)
        z = self.ConvsOut[0](z)

        output = torch.tanh(z).view(b, t, c, h, w)
        return output


class P2IDiscriminator(BaseNetwork):
    def __init__(self, in_channels: int = 16, init_weights: bool = True):
        super().__init__()
        self.in_channels = in_channels

        self.d2d = nn.Sequential(
            C2(in_channels, 64, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, True),
            C2(64, 128, k=3, s=2, p=1),
            nn.LeakyReLU(0.2, True),
            C2(128, 256, k=3, s=2, p=1),
            nn.LeakyReLU(0.2, True),
            C2(256, 256, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, True),
            C2(256, 1, k=3, s=1, p=1),
        )

        self.d3d = nn.Sequential(
            C3(1, 32, kt=3, ks=3, st=(1, 2, 2), pt=(1, 1, 1)),
            nn.LeakyReLU(0.2, True),
            C3(32, 64, kt=3, ks=3, st=(1, 2, 2), pt=(1, 1, 1)),
            nn.LeakyReLU(0.2, True),
            C3(64, 128, kt=3, ks=3, st=(1, 2, 2), pt=(1, 1, 1)),
            nn.LeakyReLU(0.2, True),
            C3(128, 128, kt=3, ks=3, st=(2, 1, 1), pt=(1, 1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv3d(128, 1, kernel_size=1)),
        )

        self.alpha2d = nn.Parameter(torch.tensor(0.0))
        self.alpha3d = nn.Parameter(torch.tensor(0.0))

        if init_weights:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        b, t, c, h, w = x.shape

        # 2D branch
        out2d = self.d2d(x.reshape(b, t * c, h, w))

        # 3D branch
        out3d = self.d3d(x.permute(0, 2, 1, 3, 4))
        out3d_2d = out3d.mean(dim=2)

        if out3d_2d.shape[-2:] != out2d.shape[-2:]:
            out3d_2d = F.interpolate(out3d_2d, size=out2d.shape[-2:], mode="bilinear", align_corners=False)

        w2 = torch.sigmoid(self.alpha2d)
        fused = (w2 * out2d + out3d_2d) 
        
        return fused.view(b, -1)
    
    
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock_do):
        super().__init__()
        layers = [ResBlock(out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

