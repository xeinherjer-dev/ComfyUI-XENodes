"""
ITMLUT (Adaptive Interval 3D-LUT / Pure PyTorch).

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Adapted from ITMLUT:
- Paper: Redistributing the Precision and Content in 3D-LUT-based Inverse Tone-mapping for HDR/WCG Display
         (ACM SIGGRAPH CVMP 2023)
- Repository: https://github.com/AndreGuo/ITMLUT (MPL-2.0 License)
- Authors: Cheng Guo, Leidong Fan, Qian Zhang, Hanyuan Liu, Kanglin Liu, Xiuhua Jiang

The trilinear interpolation over adaptive intervals in `pure_ailut_transform`
is a pure PyTorch adaptation of the AiLUT-Transform from AdaInt under Apache License 2.0:
- Paper: AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-Time Image Enhancement (CVPR 2022)
- Repository: https://github.com/ImCharlesY/AdaInt (Apache License 2.0)
- Authors: Canqian Yang, Meiguang Jin, Xu Jia, Yi Xu, Ying Chen

See THIRD_PARTY_LICENSES.md for full license texts.
"""
from __future__ import annotations

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


@functools.lru_cache(maxsize=16)
def _get_u_vertices_and_mid(n_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.linspace(0.0, 1.0, n_dim, device=device, dtype=dtype).view(1, 1, n_dim)
    mid = ((3.0 * torch.pi * u) - torch.cos(3.0 * torch.pi * u) + 1.0) / (3.0 * torch.pi + 2.0)
    return u, mid


def pure_ailut_transform(img: torch.Tensor, lut: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch implementation of AiLUT-Transform (Trilinear interpolation on 3D LUT with non-uniform vertices).
    Uses adaptive interval search via `torch.bucketize` followed by hardware-accelerated 5D `F.grid_sample`
    (3D trilinear interpolation in a single fused kernel).

    Args:
        img: [B, 3, H, W] in [0, 1]
        lut: [B, 3, D, D, D] (output values for R, G, B at 3D grid points (b, g, r))
        vertices: [B, 3, D] (coordinates for R, G, B channels along the 1D axes)
    Returns:
        [B, 3, H, W]
    """
    B, C, H, W = img.shape
    D = lut.shape[2]
    eps = 1e-10
    scale = 2.0 / (D - 1)

    img_c = img.clamp(0.0, 1.0)
    grids: list[torch.Tensor] = []

    for b in range(B):
        img_b = img_c[b]
        vert_b = vertices[b]
        coords: list[torch.Tensor] = []
        for c in range(3):
            ch = img_b[c].contiguous().view(-1)
            v = vert_b[c].contiguous().view(-1)
            idx = (torch.bucketize(ch, v) - 1).clamp(0, D - 2)
            v0 = v[idx]
            v1 = v[idx + 1]
            d = (ch - v0) / (v1 - v0 + eps)
            coords.append(((idx.to(img.dtype) + d) * scale - 1.0).view(H, W))
        grids.append(torch.stack(coords, dim=-1))

    grid_batch = torch.stack(grids, dim=0).unsqueeze(1)  # (B, 1, H, W, 3)
    sampled = F.grid_sample(lut, grid_batch, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(2)


class ITMLUT_BrightProb(nn.Module):
    def __init__(self, threshold: float = 0.75):
        super().__init__()
        self.t = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.t) / (1.0 - self.t), min=0.0, max=1.0)


class ITMLUT_DarkProb(nn.Module):
    def __init__(self, threshold: float = 0.25):
        super().__init__()
        self.t = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.t) / (0.0 - self.t), min=0.0, max=1.0)


class ITMLUT_MidProb(nn.Module):
    def __init__(self, threshold_b: float = 0.75, threshold_d: float = 0.25):
        super().__init__()
        self.prob_b = ITMLUT_BrightProb(threshold=threshold_b)
        self.prob_d = ITMLUT_DarkProb(threshold=threshold_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1.0 - self.prob_d(x) - self.prob_b(x), min=0.0, max=1.0)


class ITMLUT_BasicResBlock(nn.Module):
    def __init__(self, nc: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv0 = nn.Conv2d(nc, nc, kernel_size, stride=stride, padding=padding)
        self.conv1 = nn.Conv2d(nc, nc, kernel_size, stride=stride, padding=padding)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act(self.conv0(x))
        res = self.act(self.conv1(res))
        return x + res


class ITMLUT_GlobPriorEncoder(nn.Module):
    def __init__(self, in_nc: int = 3, out_nc: int = 128):
        super().__init__()
        self.enocoder = nn.Sequential(
            nn.Conv2d(in_nc, 16, 1),
            ITMLUT_BasicResBlock(16),
            nn.Conv2d(16, 32, 1),
            ITMLUT_BasicResBlock(32),
            nn.Conv2d(32, 64, 1),
            ITMLUT_BasicResBlock(64),
            nn.Conv2d(64, out_nc, 1),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enocoder(x).squeeze(-1).squeeze(-1)


class ITMLUT_Classifier(nn.Module):
    def __init__(self, in_nc: int = 128, out_nc: int = 5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_nc, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(128, out_nc),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ITMLUT_LUTGenAndFuse(nn.Module):
    def __init__(self, in_fea_nc: int = 128, nc: int = 3, n_dim: int = 17, n_lut: int = 5):
        super().__init__()
        self.weights_gen = ITMLUT_Classifier(in_nc=in_fea_nc, out_nc=n_lut)
        self.basic_luts = nn.Parameter(torch.zeros([n_lut, nc, n_dim, n_dim, n_dim]))
        self.n_lut = n_lut
        self.n_dim = n_dim
        self.nc = nc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.weights_gen(x))
        return torch.einsum("bn,ncdhw->bcdhw", weights, self.basic_luts)


def _itmlut_mid_curve(x: torch.Tensor) -> torch.Tensor:
    return ((3.0 * torch.pi * x) - torch.cos(3.0 * torch.pi * x) + 1.0) / (3.0 * torch.pi + 2.0)


class ITMLUTNet(nn.Module):
    """
    Neural 3D-LUT Inverse Tone Mapping Network (ITMLUT).
    Generates dynamic 3D LUTs per frame with adaptive interval sampling.
    Runs in pure PyTorch for ultra-fast, real-time video and image conversion.
    """

    def __init__(self, nc: int = 3, n_lut: int = 5, n_dim: int = 17, bright_t: float = 0.55, dark_t: float = 0.45):
        super().__init__()
        self.n_lut = n_lut
        self.n_dim = n_dim
        self.nc = nc

        self.prior_gen = nn.AvgPool2d(8)
        self.bright_prob = ITMLUT_BrightProb(threshold=bright_t)
        self.mid_prob = ITMLUT_MidProb(threshold_b=bright_t, threshold_d=dark_t)
        self.dark_prob = ITMLUT_DarkProb(threshold=dark_t)

        self.encoder_img_b = ITMLUT_GlobPriorEncoder(in_nc=nc, out_nc=128)
        self.lut_gen_b = ITMLUT_LUTGenAndFuse(in_fea_nc=128, nc=nc, n_dim=n_dim, n_lut=n_lut)

        self.encoder_img_m = ITMLUT_GlobPriorEncoder(in_nc=nc, out_nc=128)
        self.lut_gen_m = ITMLUT_LUTGenAndFuse(in_fea_nc=128, nc=nc, n_dim=n_dim, n_lut=n_lut)

        self.encoder_img_d = ITMLUT_GlobPriorEncoder(in_nc=nc, out_nc=128)
        self.lut_gen_d = ITMLUT_LUTGenAndFuse(in_fea_nc=128, nc=nc, n_dim=n_dim, n_lut=n_lut)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b = img.shape[0]
        device = img.device

        prior = self.prior_gen(img)
        weights_b = self.encoder_img_b(prior)
        weights_m = self.encoder_img_m(prior)
        weights_d = self.encoder_img_d(prior)

        LUTs_b = self.lut_gen_b(weights_b)
        LUTs_m = self.lut_gen_m(weights_m)
        LUTs_d = self.lut_gen_d(weights_d)

        u_vertices, mid_curve = _get_u_vertices_and_mid(self.n_dim, device, img.dtype)

        avg_rgb = torch.mean(img, dim=(-2, -1), keepdim=False).unsqueeze(-1)  # (B, 3, 1)
        exp_b = 1.4 + 0.8 * avg_rgb
        vertices_b = u_vertices ** (1.0 / exp_b)

        exp_d = 2.2 - 0.8 * avg_rgb
        vertices_d = u_vertices ** exp_d

        vertices_m = mid_curve.expand(b, self.nc, self.n_dim)

        prob_b = self.bright_prob(img)
        prob_d = self.dark_prob(img)
        prob_m = torch.clamp(1.0 - prob_d - prob_b, min=0.0, max=1.0)

        # Batch all 3 LUT transforms (Bright, Mid, Dark) into a single 5D grid_sample call
        all_luts = torch.cat([LUTs_b, LUTs_m, LUTs_d], dim=0)
        all_verts = torch.cat([vertices_b, vertices_m, vertices_d], dim=0)
        all_imgs = img.expand(3, -1, -1, -1, -1).reshape(3 * b, *img.shape[1:]) if b == 1 else img.repeat(3, 1, 1, 1)
        out_b, out_m, out_d = pure_ailut_transform(all_imgs, all_luts, all_verts).chunk(3, dim=0)

        out = out_b * prob_b
        out.addcmul_(out_m, prob_m)
        out.addcmul_(out_d, prob_d)
        return out
