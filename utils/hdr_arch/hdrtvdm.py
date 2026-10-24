"""
HDRTVDM (CVPR 2023): Luminance Segmented Network (TriSegNet w. Restormer).

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Adapted from HDRTVDM:
- Paper: Learning a Practical SDR-to-HDRTV Up-Conversion Using New Dataset and Degradation Models (CVPR 2023)
- Repository: https://github.com/AndreGuo/HDRTVDM (MPL-2.0 License)
- Authors: Cheng Guo, Leidong Fan, Ziyu Xue, Xiuhua Jiang

The Restormer transformer blocks included below are adapted under the MIT License:
- Repository: https://github.com/swz30/Restormer (MIT License)
- Copyright (c) 2022 Syed Waqas Zamir and contributors

See THIRD_PARTY_LICENSES.md for full license texts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _which_act(act_type: str) -> nn.Module:
    if act_type == "relu":
        return nn.ReLU()
    elif act_type == "relu6":
        return nn.ReLU6()
    elif act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1)
    elif act_type == "elu":
        return nn.ELU()
    raise AttributeError(f"Unsupported activation_type: {act_type}")


class Cond_Trunk(nn.Module):
    def __init__(self, in_nc: int = 3, nf: int = 32, act_type: str = "leakyrelu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_nc, nf // 2, 3, 2, 1)
        self.conv2 = nn.Conv2d(nf // 2, nf // 2, 3, 2, 1, groups=4)
        self.conv3 = nn.Conv2d(nf // 2, nf, 3, 2, 1)
        self.act = _which_act(act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))
        return torch.mean(out, dim=[2, 3], keepdim=False)


class PWConv_Cond(nn.Module):
    def __init__(self, in_nc: int = 3, out_nc: int = 3, nf: int = 32, act_type: str = "leakyrelu"):
        super().__init__()
        self.nf = nf
        self.out_nc = out_nc
        self.conv1 = nn.Conv2d(in_nc, nf, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf * 2, 1, 1)
        self.conv3 = nn.Conv2d(nf * 2, nf * 2, 1, 1)
        self.conv4 = nn.Conv2d(nf * 2, nf, 1, 1)
        self.conv5 = nn.Conv2d(nf, out_nc, 1, 1)
        self.act = _which_act(act_type)

        self.cond_net = Cond_Trunk(in_nc=in_nc, nf=nf, act_type=act_type)
        self.cond_scale1 = nn.Linear(nf, nf, bias=True)
        self.cond_scale2 = nn.Linear(nf, nf * 2, bias=True)
        self.cond_scale3 = nn.Linear(nf, nf * 2, bias=True)
        self.cond_scale4 = nn.Linear(nf, nf, bias=True)
        self.cond_scale5 = nn.Linear(nf, out_nc, bias=True)

        self.cond_shift1 = nn.Linear(nf, nf, bias=True)
        self.cond_shift2 = nn.Linear(nf, nf * 2, bias=True)
        self.cond_shift3 = nn.Linear(nf, nf * 2, bias=True)
        self.cond_shift4 = nn.Linear(nf, nf, bias=True)
        self.cond_shift5 = nn.Linear(nf, out_nc, bias=True)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        cond = self.cond_net(prior)
        scale1 = self.cond_scale1(cond).view(-1, self.nf, 1, 1)
        scale2 = self.cond_scale2(cond).view(-1, self.nf * 2, 1, 1)
        scale3 = self.cond_scale3(cond).view(-1, self.nf * 2, 1, 1)
        scale4 = self.cond_scale4(cond).view(-1, self.nf, 1, 1)
        scale5 = self.cond_scale5(cond).view(-1, self.out_nc, 1, 1)

        shift1 = self.cond_shift1(cond).view(-1, self.nf, 1, 1)
        shift2 = self.cond_shift2(cond).view(-1, self.nf * 2, 1, 1)
        shift3 = self.cond_shift3(cond).view(-1, self.nf * 2, 1, 1)
        shift4 = self.cond_shift4(cond).view(-1, self.nf, 1, 1)
        shift5 = self.cond_shift5(cond).view(-1, self.out_nc, 1, 1)

        out = self.conv1(x)
        out = self.act(out * scale1 + shift1 + out)
        out = self.conv2(out)
        out = self.act(out * scale2 + shift2 + out)
        out = self.conv3(out)
        out = self.act(out * scale3 + shift3 + out)
        out = self.conv4(out)
        out = self.act(out * scale4 + shift4 + out)
        out = self.conv5(out)
        out = self.act(out * scale5 + shift5 + out)
        return out


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class RestormerLayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str = "WithBias"):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # b c h w -> b (h w) c -> norm -> b c h w
        out = x.permute(0, 2, 3, 1).flatten(1, 2)
        out = self.body(out)
        return out.view(x.shape[0], h, w, -1).permute(0, 3, 1, 2)


class RestormerFeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class RestormerAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head = self.num_heads
        head_c = c // head

        q = q.view(b, head, head_c, h * w)
        k = k.view(b, head, head_c, h * w)
        v = v.view(b, head, head_c, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.view(b, head * head_c, h, w)
        return self.project_out(out)


class RestormerTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float, bias: bool, layer_norm_type: str):
        super().__init__()
        self.norm1 = RestormerLayerNorm(dim, layer_norm_type)
        self.attn = RestormerAttention(dim, num_heads, bias)
        self.norm2 = RestormerLayerNorm(dim, layer_norm_type)
        self.ffn = RestormerFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RestormerOverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RestormerDownsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RestormerUpsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 16,
        ffn_expansion_factor: float = 2.0,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
    ):
        super().__init__()
        num_blocks = [1, 1, 1]
        heads = [1, 2, 4]

        self.patch_embed = RestormerOverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[
            RestormerTransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = RestormerDownsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            RestormerTransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = RestormerDownsample(int(dim * 2))
        self.latent = nn.Sequential(*[
            RestormerTransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = RestormerUpsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            RestormerTransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = RestormerUpsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            RestormerTransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        self.output = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        latent = self.latent(inp_enc_level3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return self.output(out_dec_level1)


class BrightIntensityLum(nn.Module):
    def __init__(self, t_b: float = 0.7611):
        super().__init__()
        self.t_b = t_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lum = 0.2627 * x[:, 0, :, :] + 0.6780 * x[:, 1, :, :] + 0.0593 * x[:, 2, :, :]
        lum = torch.clamp(lum / self.t_b, min=0.0, max=1.0)
        return lum.unsqueeze(1).repeat(1, 3, 1, 1)


class DarkIntensityLum(nn.Module):
    def __init__(self, t_d: float = 0.2539):
        super().__init__()
        self.t_d = t_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lum = 0.2627 * x[:, 0, :, :] + 0.6780 * x[:, 1, :, :] + 0.0593 * x[:, 2, :, :]
        lum = torch.clamp((lum - 1.0) / (self.t_d - 1.0), min=0.0, max=1.0)
        return lum.unsqueeze(1).repeat(1, 3, 1, 1)


class TriSegNet(nn.Module):
    """
    HDRTVDM Luminance Segmented Network (CVPR 2023).
    Segments luminance into bright/dark/mid components, applying Restormer to bright/dark
    and conditioned pointwise conv to mid frequencies.
    """

    def __init__(self, in_nc: int = 3, nf: int = 32, out_nc: int = 3, act_type: str = "leakyrelu"):
        super().__init__()
        self.prior_downscale = nn.AvgPool2d(kernel_size=8, stride=8)
        self.mid_branch = PWConv_Cond(in_nc=in_nc, out_nc=out_nc, nf=nf, act_type=act_type)
        self.bright_branch = Restormer(inp_channels=3, out_channels=3, dim=nf // 2, ffn_expansion_factor=2.0)
        self.dark_branch = Restormer(inp_channels=3, out_channels=3, dim=nf // 2, ffn_expansion_factor=2.0)
        self.t_b = 0.7611
        self.t_d = 0.7611
        self.bright_intensity = BrightIntensityLum(t_b=self.t_b)
        self.dark_intensity = DarkIntensityLum(t_d=self.t_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial dimensions must be multiple of 8 for 3 levels of 2x down/up sampling and AvgPool2d(8)
        h, w = x.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        prior = self.prior_downscale(x)
        bright_part = x * self.bright_intensity(x)
        dark_part = x * self.dark_intensity(x)
        mid_part = x

        bright_out = self.bright_branch(bright_part)
        dark_out = self.dark_branch(dark_part)
        mid_out = self.mid_branch(mid_part, prior)

        out = mid_out + bright_out + dark_out
        if pad_h > 0 or pad_w > 0:
            out = out[..., :h, :w]
        return out
