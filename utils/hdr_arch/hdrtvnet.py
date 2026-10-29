"""
HDRTVNet and HDRTVNet++ model architectures.

Adapted from:
- HDRTVNet (ICCV 2021): https://github.com/chxy95/HDRTVNet (MIT License, Copyright 2021 XyChen)
- HDRTVNet++ (TPAMI 2023): https://github.com/xiaom233/HDRTVNet-plus (MIT License, Copyright 2023 Hoven Li)

See THIRD_PARTY_LICENSES.md for full license texts.
"""
from __future__ import annotations

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# HDRTVNet AGCM (ConditionNet)
# --------------------------------------------------------------------------------------
def _color_block(in_filters: int, out_filters: int, normalization: bool = False) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0),
        nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True),
        nn.LeakyReLU(0.2),
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class _ColorCondition(nn.Module):
    def __init__(self, out_c: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            *_color_block(3, 16, normalization=True),
            *_color_block(16, 32, normalization=True),
            *_color_block(32, 64, normalization=True),
            *_color_block(64, 128, normalization=True),
            *_color_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AGCMNet(nn.Module):
    """HDRTVNet Adaptive Global Color Mapping network (state_dict compatible with ConditionNet)."""

    def __init__(self, nf: int = 64, cond_c: int = 6):
        super().__init__()
        self.nf = nf
        self.classifier = _ColorCondition(out_c=cond_c)

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def compute_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return self.classifier(condition).flatten(1)

    def forward_with_condition(self, content: torch.Tensor, fea: torch.Tensor) -> torch.Tensor:
        nf = self.nf
        if content.shape[0] == 1:
            # Fuse channel-wise affine modulation (scale + 1.0, shift) directly into 1x1 conv weights & biases
            s1 = (self.cond_scale_first(fea) + 1.0).view(nf, 1, 1, 1)
            b1 = self.conv_first.bias * s1.view(-1) + self.cond_shift_first(fea).view(-1)
            out = self.act(F.conv2d(content, self.conv_first.weight * s1, b1))

            s2 = (self.cond_scale_HR(fea) + 1.0).view(nf, 1, 1, 1)
            b2 = self.HRconv.bias * s2.view(-1) + self.cond_shift_HR(fea).view(-1)
            out = self.act(F.conv2d(out, self.HRconv.weight * s2, b2))

            s3 = (self.cond_scale_last(fea) + 1.0).view(3, 1, 1, 1)
            b3 = self.conv_last.bias * s3.view(-1) + self.cond_shift_last(fea).view(-1)
            return F.conv2d(out, self.conv_last.weight * s3, b3)

        out = self.conv_first(content)
        out = out.mul_((self.cond_scale_first(fea) + 1.0).view(-1, nf, 1, 1)).add_(self.cond_shift_first(fea).view(-1, nf, 1, 1))
        out = self.act(out)

        out = self.HRconv(out)
        out = out.mul_((self.cond_scale_HR(fea) + 1.0).view(-1, nf, 1, 1)).add_(self.cond_shift_HR(fea).view(-1, nf, 1, 1))
        out = self.act(out)

        out = self.conv_last(out)
        out = out.mul_((self.cond_scale_last(fea) + 1.0).view(-1, 3, 1, 1)).add_(self.cond_shift_last(fea).view(-1, 3, 1, 1))
        return out

    def forward(self, content: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.forward_with_condition(content, self.compute_condition(condition))


# --------------------------------------------------------------------------------------
# HDRTVNet-LE (初代 modified SRResNet)
# --------------------------------------------------------------------------------------
class _ResidualBlockNoBN(nn.Module):
    def __init__(self, nf: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.relu(self.conv1(x), inplace=True))


class LENet(nn.Module):
    """HDRTVNet Local Enhancement network (state_dict compatible with Base_arch.SRResNet)."""

    def __init__(self, in_nc: int = 3, out_nc: int = 3, nf: int = 64, nb: int = 16, act_type: str = "relu"):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 2, 1, bias=True)
        block = functools.partial(_ResidualBlockNoBN, nf=nf)
        self.recon_trunk = nn.Sequential(*[block() for _ in range(nb)])
        self.upconv = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upsampler = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        if act_type == "leakyrelu":
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stride-2 conv + PixelShuffle(2) require even spatial dims
        h, w = x.shape[-2:]
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        fea = self.act(self.conv_first(x))
        out = self.recon_trunk(fea)
        out = self.act(self.upsampler(self.upconv(out)))
        out = self.conv_last(self.act(self.HRconv(out)))
        return out[..., :h, :w]


# --------------------------------------------------------------------------------------
# HDRTVNet++ HDRUNet3T1 (SFT-based Local Enhancement)
# --------------------------------------------------------------------------------------
class SFTLayer(nn.Module):
    def __init__(self, in_nc: int = 16, out_nc: int = 32, nf: int = 16):
        super().__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        fea, cond = x
        # Fuse parallel 1x1 convolutions on the shared `cond` tensor and pre-add +1.0 to scale bias
        w0 = torch.cat([self.SFT_scale_conv0.weight, self.SFT_shift_conv0.weight], dim=0)
        b0 = torch.cat([self.SFT_scale_conv0.bias, self.SFT_shift_conv0.bias], dim=0)
        w1 = torch.cat([self.SFT_scale_conv1.weight, self.SFT_shift_conv1.weight], dim=0)
        b1 = torch.cat([self.SFT_scale_conv1.bias + 1.0, self.SFT_shift_conv1.bias], dim=0)
        h0 = F.leaky_relu(F.conv2d(cond, w0, b0), 0.1, inplace=True)
        h1 = F.conv2d(h0, w1, b1, groups=2)
        scale_p1, shift = h1.chunk(2, dim=1)
        return torch.addcmul(shift, fea, scale_p1)


class ResBlockWithSFT(nn.Module):
    def __init__(self, nf: int = 32):
        super().__init__()
        self.sft1 = SFTLayer(in_nc=nf // 2, out_nc=nf, nf=nf // 2)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.sft2 = SFTLayer(in_nc=nf // 2, out_nc=nf, nf=nf // 2)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        fea, cond = x
        out = self.sft1((fea, cond))
        out = F.relu(self.conv1(out), inplace=True)
        out = self.sft2((out, cond))
        out = self.conv2(out)
        out.add_(fea)
        return (out, cond)


def _make_res_layers(nf: int, n_layers: int) -> nn.Sequential:
    return nn.Sequential(*[ResBlockWithSFT(nf=nf) for _ in range(n_layers)])


class HDRUNet3T1(nn.Module):
    """
    HDRTVNet++ Local Enhancement Network with Spatial Feature Transform.
    Downsamples 3 times (factor of 8), with automatic spatial padding for arbitrary resolutions.
    """

    def __init__(self, in_nc: int = 3, out_nc: int = 3, nf: int = 32, act_type: str = "relu", weighting_network: bool = False):
        super().__init__()
        self.nf = nf
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.SFT_layer1 = SFTLayer(in_nc=nf // 2, out_nc=nf, nf=nf // 2)
        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv3 = nn.Conv2d(nf, nf, 3, 2, 1)

        self.recon_trunk1 = _make_res_layers(nf, 1)
        self.recon_trunk2 = _make_res_layers(nf, 1)
        self.recon_trunk3 = _make_res_layers(nf, 4)
        self.recon_trunk4 = _make_res_layers(nf, 1)
        self.recon_trunk5 = _make_res_layers(nf, 1)

        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv3 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))

        self.SFT_layer2 = SFTLayer(in_nc=nf // 2, out_nc=nf, nf=nf // 2)
        self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        cond_in_nc = 3
        cond_nf = 64
        self.cond_first = nn.Sequential(
            nn.Conv2d(cond_in_nc, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
        )
        self.CondNet1 = nn.Sequential(
            nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_nf, nf // 2, 1),
        )
        self.CondNet2 = nn.Sequential(
            nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_nf, nf // 2, 1),
        )
        self.CondNet3 = nn.Sequential(
            nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_nf, nf // 2, 1),
        )
        self.CondNet4 = nn.Sequential(
            nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_nf, nf // 2, 3, 2, 1),
        )

        self.weighting_network = weighting_network
        if weighting_network:
            self.mask_est = nn.Sequential(
                nn.Conv2d(in_nc, nf, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, 1), nn.ReLU(inplace=True),
                nn.Conv2d(nf, out_nc, 1),
            )
        self.act = nn.LeakyReLU(0.1, inplace=True) if act_type == "leakyrelu" else nn.ReLU(inplace=True)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            img, cond_img = x[0], x[1]
        else:
            img, cond_img = x, x

        # Pad to multiple of 8 for 3 levels of 2x down/up sampling
        h, w = img.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            same_input = cond_img is img
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
            cond_img = img if same_input else F.pad(cond_img, (0, pad_w, 0, pad_h), mode="replicate")

        mask_out = self.mask_est(img) * img if self.weighting_network else img

        cond = self.cond_first(cond_img)
        cond1 = self.CondNet1(cond)
        # Fuse shared stride-2 3x3 convolutions across CondNet2, CondNet3, CondNet4
        w_s1 = torch.cat([self.CondNet2[0].weight, self.CondNet3[0].weight, self.CondNet4[0].weight], dim=0)
        b_s1 = torch.cat([self.CondNet2[0].bias, self.CondNet3[0].bias, self.CondNet4[0].bias], dim=0)
        w_s2 = torch.cat([self.CondNet3[2].weight, self.CondNet4[2].weight], dim=0)
        b_s2 = torch.cat([self.CondNet3[2].bias, self.CondNet4[2].bias], dim=0)
        h234 = F.leaky_relu(F.conv2d(cond, w_s1, b_s1, stride=2, padding=1), 0.1, inplace=True)
        c_nf = cond.shape[1]
        h2 = h234[:, :c_nf]
        h34 = F.leaky_relu(F.conv2d(h234[:, c_nf:], w_s2, b_s2, stride=2, padding=1, groups=2), 0.1, inplace=True)
        h3, h4 = h34.chunk(2, dim=1)
        cond2 = self.CondNet2[4](F.leaky_relu(self.CondNet2[2](h2), 0.1, inplace=True))
        cond3 = self.CondNet3[4](h3)
        cond4 = self.CondNet4[4](h4)

        fea0 = self.act(self.conv_first(img))
        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))
        fea2, _ = self.recon_trunk2((fea2, cond3))

        fea3 = self.act(self.down_conv3(fea2))
        out, _ = self.recon_trunk3((fea3, cond4))
        out.add_(fea3)

        out = self.act(self.up_conv1(out)).add_(fea2)
        out, _ = self.recon_trunk4((out, cond3))

        out = self.act(self.up_conv2(out)).add_(fea1)
        out, _ = self.recon_trunk5((out, cond2))

        out = self.act(self.up_conv3(out)).add_(fea0)
        out = self.SFT_layer2((out, cond1))
        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out.add_(mask_out)

        # Slice back to original resolution
        return out[..., :h, :w]


# --------------------------------------------------------------------------------------
# HDRTVNet++ Integrated Model (AGCM + LE, e.g. HR.pth)
# --------------------------------------------------------------------------------------
class EnsembleAGCMLE(nn.Module):
    """Integrated AGCM + HDRUNet3T1 network from HDRTVNet++."""

    def __init__(self, cond_c: int = 6, nf: int = 32):
        super().__init__()
        self.AGCM = AGCMNet(nf=64, cond_c=cond_c)
        self.LE = HDRUNet3T1(in_nc=3, out_nc=3, nf=nf, act_type="relu", weighting_network=False)

    def forward(self, content: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        agcm_out = self.AGCM(content, condition)
        le_out = self.LE((agcm_out, agcm_out))
        return le_out, agcm_out


# --------------------------------------------------------------------------------------
# HDRTVNet Highlight Generation (HG / Hallucination Generator from ICCV 2021)
# --------------------------------------------------------------------------------------
def generate_highlight_mask(sdr_bchw: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
    """
    Computes soft saturation highlight mask (B, 1, H, W) in [0, 1] from input SDR image.
    Pixels with max RGB > threshold ramp linearly from 0.0 to 1.0.
    """
    max_c, _ = torch.max(sdr_bchw, dim=1, keepdim=True)
    mask = torch.clamp((max_c - threshold) / max(1e-5, 1.0 - threshold), 0.0, 1.0)
    return mask


def _hg_conv_bn_relu(in_nc: int, out_nc: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_nc, out_nc, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_nc, affine=True),
        nn.ReLU(inplace=True),
    )


def _hg_up_pixel_shuffle(in_nc: int, out_nc: int, sf: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_nc, out_nc * (sf ** 2), kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(sf),
        nn.ReLU(inplace=True),
    )


class HallucinationGenerator(nn.Module):
    """
    Highlight Generation (HG) network from ICCV 2021 HDRTVNet.
    Takes (LE_output_image, highlight_mask) and synthesizes realistic radiance/texture in saturated areas.
    """

    def __init__(self, in_nc: int = 3, out_nc: int = 3, nf: int = 64, sf: int = 2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(sf)

        self.conv1 = _hg_conv_bn_relu(in_nc, nf)
        self.conv2 = _hg_conv_bn_relu(nf, 2 * nf)

        self.conv3_1 = _hg_conv_bn_relu(2 * nf, 4 * nf)
        self.conv3_2 = _hg_conv_bn_relu(4 * nf, 4 * nf)

        self.conv4_1 = _hg_conv_bn_relu(4 * nf, 8 * nf)
        self.conv4_2 = _hg_conv_bn_relu(8 * nf, 8 * nf)

        self.conv5_1 = _hg_conv_bn_relu(8 * nf, 8 * nf)
        self.conv5_2 = _hg_conv_bn_relu(8 * nf, 8 * nf)

        self.conv_code1 = _hg_conv_bn_relu(8 * nf, 8 * nf)
        self.conv_code2 = _hg_conv_bn_relu(8 * nf, 8 * nf)

        self.Up_conv1 = _hg_up_pixel_shuffle(8 * nf, 8 * nf, sf=2)
        self.conv6 = nn.Conv2d(16 * nf, 8 * nf, 1, 1)

        self.Up_conv2 = _hg_up_pixel_shuffle(8 * nf, 8 * nf, sf=2)
        self.conv7 = nn.Conv2d(16 * nf, 4 * nf, 1, 1)

        self.Up_conv3 = _hg_up_pixel_shuffle(4 * nf, 4 * nf, sf=2)
        self.conv8 = nn.Conv2d(8 * nf, 2 * nf, 1, 1)

        self.Up_conv4 = _hg_up_pixel_shuffle(2 * nf, 2 * nf, sf=2)
        self.conv9 = nn.Conv2d(4 * nf, nf, 1, 1)

        self.Up_conv5 = _hg_up_pixel_shuffle(nf, nf, sf=2)
        self.conv10 = nn.Conv2d(2 * nf, out_nc, 1, 1)

        self.conv_last = nn.Conv2d(2 * out_nc, out_nc, 1, 1)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Spatial dimensions must be divisible by 32 for the 5-stage poolings
        h, w = img.shape[-2:]
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        if pad_h > 0 or pad_w > 0:
            img_in = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            img_in = img

        conv1_out = self.conv1(img_in)
        conv2_out = self.conv2(self.maxpool(conv1_out))

        conv3_out = self.conv3_2(self.maxpool(self.conv3_1(conv2_out)))
        conv4_out = self.conv4_2(self.maxpool(self.conv4_1(conv3_out)))
        conv5_out = self.conv5_2(self.maxpool(self.conv5_1(conv4_out)))

        conv_code = self.conv_code2(self.maxpool(self.conv_code1(conv5_out)))

        conv6_out = self.conv6(torch.cat((self.Up_conv1(conv_code), conv5_out), dim=1))
        conv7_out = self.conv7(torch.cat((self.Up_conv2(conv6_out), conv4_out), dim=1))
        conv8_out = self.conv8(torch.cat((self.Up_conv3(conv7_out), conv3_out), dim=1))
        conv9_out = self.conv9(torch.cat((self.Up_conv4(conv8_out), conv2_out), dim=1))
        conv10_out = self.conv10(torch.cat((self.Up_conv5(conv9_out), conv1_out), dim=1))

        out = self.conv_last(torch.cat((conv10_out, img_in), dim=1))

        if pad_h > 0 or pad_w > 0:
            out = out[..., :h, :w]

        # In HDRTVNet formulation: out = mask * out + img
        return mask * out + img
