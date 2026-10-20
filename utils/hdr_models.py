"""
AI-based Inverse Tone Mapping (SDR -> HDR) model architectures and loading helpers.

Supported architectures (auto-detected from the state_dict):
  1. HDRTVNet-AGCM:
     - Adaptive Global Color Mapping (ConditionNet) from ICCV 2021 (MIT License, Copyright 2021 XyChen)
     - E.g. AGCM.pth / AGCM_G.pth
  2. HDRTVNet-Plus-Ensemble:
     - End-to-end AGCM + Local Enhancement (HDRUNet3T1) from HDRTVNet++ (MIT License, Copyright 2023 Hoven Li)
     - E.g. Ensemble_AGCM_LE.pth / HR.pth (single file with both global & local models)
  3. HDRTVNet-Plus-LE:
     - Standalone Local Enhancement (HDRUNet3T1 with Spatial Feature Transform)
     - E.g. LE.pth
  4. HDRTVNet-LE:
     - Original modified SRResNet Local Enhancement from ICCV 2021
     - E.g. LE_weights.pth
  5. HDRTVDM:
     - Luminance Segmented Network (LSN w. Restormer) from CVPR 2023 (MPL-2.0, Copyright 2023 Cheng Guo et al.)
     - E.g. HDRTVDM_params.pth / HDRTVDM_params_3DM.pth
  6. ITMLUT:
     - Adaptive Interval 3D-LUT Inverse Tone Mapping from ACM CVMP 2023 (MPL-2.0, Copyright 2023 Cheng Guo et al.)
     - E.g. ITMLUT_params.pth / ITMLUT_params_TV1K.pth

Both static images and arbitrary-resolution video frames are supported.
Spatial dimensions for LE are automatically padded to multiples of 8 and unpadded on output.

See THIRD_PARTY_LICENSES.md for full license texts and credits.
"""
from __future__ import annotations

import functools
import logging
import os
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MODEL_FOLDER_NAME = "sdr_to_hdr"
_MIN_COND_SIZE = 32


# --------------------------------------------------------------------------------------
# Model folder registration (ComfyUI/models/sdr_to_hdr)
# --------------------------------------------------------------------------------------
def register_model_folder() -> None:
    import folder_paths

    default_dir = os.path.join(folder_paths.models_dir, MODEL_FOLDER_NAME)
    if MODEL_FOLDER_NAME in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, default_dir)
    else:
        folder_paths.folder_names_and_paths[MODEL_FOLDER_NAME] = (
            [default_dir],
            set(folder_paths.supported_pt_extensions),
        )


# --------------------------------------------------------------------------------------
# Model auto-download support
# --------------------------------------------------------------------------------------
MODEL_DOWNLOAD_URLS: dict[str, str] = {
    "Ensemble_AGCM_LE.pth": (
        "https://github.com/xiaom233/HDRTVNet-plus/raw/main/pretrained_models/Ensemble_AGCM_LE.pth"
    ),
    "HR.pth": (
        "https://github.com/xiaom233/HDRTVNet-plus/raw/main/pretrained_models/HR.pth"
    ),
    "HDRTVDM_params_3DM.pth": (
        "https://raw.githubusercontent.com/AndreGuo/HDRTVDM/main/method/params_3DM.pth"
    ),
    "HDRTVDM_params.pth": (
        "https://raw.githubusercontent.com/AndreGuo/HDRTVDM/main/method/params.pth"
    ),
    "HDRTVDM_params_DaVinci.pth": (
        "https://raw.githubusercontent.com/AndreGuo/HDRTVDM/main/method/params_DaVinci.pth"
    ),
    "ITMLUT_params.pth": (
        "https://raw.githubusercontent.com/AndreGuo/ITMLUT/main/params.pth"
    ),
    "ITMLUT_params_TV1K.pth": (
        "https://raw.githubusercontent.com/AndreGuo/ITMLUT/main/params_TV1K.pth"
    ),
    "AGCM.pth": (
        "https://github.com/xiaom233/HDRTVNet-plus/raw/main/pretrained_models/AGCM.pth"
    ),
    "LE.pth": (
        "https://github.com/xiaom233/HDRTVNet-plus/raw/main/pretrained_models/LE.pth"
    ),
}

DEFAULT_MODELS: list[str] = [
    "HR.pth",
    "HDRTVDM_params_3DM.pth",
    "ITMLUT_params.pth",
]


def download_model_file(url: str, dest_path: str) -> None:
    """Downloads a model file with User-Agent and progress reporting."""
    import shutil
    import urllib.request

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".downloading"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ComfyUI-XENodes/1.0)"},
    )

    try:
        filename = os.path.basename(dest_path)
        print(f"[XENodes] Starting automatic download of '{filename}'...")
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = resp.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else 0
            downloaded = 0
            chunk_size = 64 * 1024

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        pct = (downloaded / total_bytes) * 100.0
                        print(
                            f"\r[XENodes] Downloading '{filename}': {pct:.1f}% ({downloaded // 1024}/{total_bytes // 1024} KB)",
                            end="",
                            flush=True,
                        )

            if total_bytes > 0 and downloaded != total_bytes:
                raise IOError(
                    f"incomplete download ({downloaded}/{total_bytes} bytes)"
                )
            print(f"\n[XENodes] Successfully saved to: {dest_path}")
        shutil.move(tmp_path, dest_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"[XENodes] Failed to download model from {url}: {e}"
        ) from e


def get_available_models() -> list[str]:
    """Returns available model filenames, filtering out non-standalone sub-models (e.g. partial LE.pth, AGCM.pth)."""
    import folder_paths

    # Standalone full models only. DaVinci and sub-models should not clutter the UI.
    excluded = {
        "LE.pth",
        "AGCM.pth",
        "HDRTVDM_params_DaVinci.pth",
        "ITMLUT_params_TV1K.pth",
    }

    existing = folder_paths.get_filename_list(MODEL_FOLDER_NAME)
    models: list[str] = list(DEFAULT_MODELS)
    for name in existing:
        if name in excluded or name.endswith(":Zone.Identifier"):
            continue
        if name not in models:
            models.append(name)
    return models


def resolve_or_download_model(model_name: str) -> str:
    """Resolves local model file path or downloads it automatically if missing."""
    import folder_paths

    # 1. Check if folder_paths already finds it
    path = folder_paths.get_full_path(MODEL_FOLDER_NAME, model_name)
    if path is not None and os.path.exists(path):
        return path

    # 2. Check in the default directory
    dirs = folder_paths.get_folder_paths(MODEL_FOLDER_NAME)
    default_dir = dirs[0] if dirs else os.path.join(folder_paths.models_dir, MODEL_FOLDER_NAME)
    dest_path = os.path.join(default_dir, model_name)
    if os.path.exists(dest_path):
        return dest_path

    # 3. Download if URL is available
    url = MODEL_DOWNLOAD_URLS.get(model_name)
    if not url:
        raise FileNotFoundError(
            f"[XENodes.LoadSDRtoHDRModel] Model '{model_name}' not found in 'models/{MODEL_FOLDER_NAME}', "
            f"and no download URL is known."
        )

    download_model_file(url, dest_path)
    return dest_path


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
        out = self.conv_first(content)
        out = out * self.cond_scale_first(fea).view(-1, nf, 1, 1) + self.cond_shift_first(fea).view(-1, nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * self.cond_scale_HR(fea).view(-1, nf, 1, 1) + self.cond_shift_HR(fea).view(-1, nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * self.cond_scale_last(fea).view(-1, 3, 1, 1) + self.cond_shift_last(fea).view(-1, 3, 1, 1) + out
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
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        return fea * (scale + 1.0) + shift


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
        return (fea + out, cond)


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
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
            cond_img = F.pad(cond_img, (0, pad_w, 0, pad_h), mode="replicate")

        mask_out = self.mask_est(img) * img if self.weighting_network else img

        cond = self.cond_first(cond_img)
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)
        cond4 = self.CondNet4(cond)

        fea0 = self.act(self.conv_first(img))
        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))
        fea2, _ = self.recon_trunk2((fea2, cond3))

        fea3 = self.act(self.down_conv3(fea2))
        out, _ = self.recon_trunk3((fea3, cond4))
        out = out + fea3

        out = self.act(self.up_conv1(out)) + fea2
        out, _ = self.recon_trunk4((out, cond3))

        out = self.act(self.up_conv2(out)) + fea1
        out, _ = self.recon_trunk5((out, cond2))

        out = self.act(self.up_conv3(out)) + fea0
        out = self.SFT_layer2((out, cond1))
        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out = mask_out + out

        # Slice back to original resolution
        return out[..., :h, :w]


# --------------------------------------------------------------------------------------
# HDRTVNet++ Ensemble_AGCM_LE (AGCM + LE integrated model)
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


# --------------------------------------------------------------------------------------
# HDRTVDM (CVPR 2023): Luminance Segmented Network (TriSegNet w. Restormer)
# Adapted from https://github.com/AndreGuo/HDRTVDM (MPL-2.0 License)
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# ITMLUT (Adaptive Interval 3D-LUT / Pure PyTorch)
# Adapted from https://github.com/AndreGuo/ITMLUT (MPL-2.0 License)
# --------------------------------------------------------------------------------------
def pure_ailut_transform(img: torch.Tensor, lut: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch implementation of AiLUT-Transform (Trilinear interpolation on 3D LUT with non-uniform vertices).
    Runs efficiently on CPU and CUDA without custom C++/CUDA extension compilation.

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

    img = img.clamp(0.0, 1.0)
    out = []

    for b in range(B):
        img_b = img[b]
        lut_b = lut[b]
        vert_b = vertices[b]

        r = img_b[0].contiguous().view(-1)
        g = img_b[1].contiguous().view(-1)
        b_ch = img_b[2].contiguous().view(-1)

        v_r = vert_b[0].contiguous().view(-1)
        v_g = vert_b[1].contiguous().view(-1)
        v_b = vert_b[2].contiguous().view(-1)

        rid = (torch.bucketize(r, v_r) - 1).clamp(0, D - 2)
        gid = (torch.bucketize(g, v_g) - 1).clamp(0, D - 2)
        bid = (torch.bucketize(b_ch, v_b) - 1).clamp(0, D - 2)

        r0 = v_r[rid]
        r1 = v_r[rid + 1]
        g0 = v_g[gid]
        g1 = v_g[gid + 1]
        b0 = v_b[bid]
        b1 = v_b[bid + 1]

        rd = (r - r0) / (r1 - r0 + eps)
        gd = (g - g0) / (g1 - g0 + eps)
        bd = (b_ch - b0) / (b1 - b0 + eps)

        w000 = (1.0 - rd) * (1.0 - gd) * (1.0 - bd)
        w100 = rd * (1.0 - gd) * (1.0 - bd)
        w010 = (1.0 - rd) * gd * (1.0 - bd)
        w110 = rd * gd * (1.0 - bd)
        w001 = (1.0 - rd) * (1.0 - gd) * bd
        w101 = rd * (1.0 - gd) * bd
        w011 = (1.0 - rd) * gd * bd
        w111 = rd * gd * bd

        lut_flat = lut_b.view(3, -1)
        stride_g = D
        stride_b = D * D

        idx000 = rid + gid * stride_g + bid * stride_b
        idx100 = (rid + 1) + gid * stride_g + bid * stride_b
        idx010 = rid + (gid + 1) * stride_g + bid * stride_b
        idx110 = (rid + 1) + (gid + 1) * stride_g + bid * stride_b
        idx001 = rid + gid * stride_g + (bid + 1) * stride_b
        idx101 = (rid + 1) + gid * stride_g + (bid + 1) * stride_b
        idx011 = rid + (gid + 1) * stride_g + (bid + 1) * stride_b
        idx111 = (rid + 1) + (gid + 1) * stride_g + (bid + 1) * stride_b

        c000 = lut_flat[:, idx000]
        c100 = lut_flat[:, idx100]
        c010 = lut_flat[:, idx010]
        c110 = lut_flat[:, idx110]
        c001 = lut_flat[:, idx001]
        c101 = lut_flat[:, idx101]
        c011 = lut_flat[:, idx011]
        c111 = lut_flat[:, idx111]

        res = (
            c000 * w000 + c100 * w100 + c010 * w010 + c110 * w110 +
            c001 * w001 + c101 * w101 + c011 * w011 + c111 * w111
        )
        out.append(res.view(3, H, W))

    return torch.stack(out, dim=0)


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

        u_vertices = torch.linspace(0.0, 1.0, self.n_dim, device=device).unsqueeze(0).unsqueeze(0).repeat(b, 1, 1)

        avg_r = torch.mean(img[:, 0, :, :], dim=(-2, -1), keepdim=True)
        avg_g = torch.mean(img[:, 1, :, :], dim=(-2, -1), keepdim=True)
        avg_b = torch.mean(img[:, 2, :, :], dim=(-2, -1), keepdim=True)

        exp_r_b = 1.4 + 0.8 * avg_r
        exp_g_b = 1.4 + 0.8 * avg_g
        exp_b_b = 1.4 + 0.8 * avg_b
        vertices_b = torch.cat([
            u_vertices ** (1.0 / exp_r_b.unsqueeze(-1)),
            u_vertices ** (1.0 / exp_g_b.unsqueeze(-1)),
            u_vertices ** (1.0 / exp_b_b.unsqueeze(-1)),
        ], dim=1)

        exp_r_d = 2.2 - 0.8 * avg_r
        exp_g_d = 2.2 - 0.8 * avg_g
        exp_b_d = 2.2 - 0.8 * avg_b
        vertices_d = torch.cat([
            u_vertices ** exp_r_d.unsqueeze(-1),
            u_vertices ** exp_g_d.unsqueeze(-1),
            u_vertices ** exp_b_d.unsqueeze(-1),
        ], dim=1)

        vertices_m = _itmlut_mid_curve(u_vertices.repeat(1, self.nc, 1))

        out = (
            pure_ailut_transform(img, LUTs_b, vertices_b) * self.bright_prob(img) +
            pure_ailut_transform(img, LUTs_m, vertices_m) * self.mid_prob(img) +
            pure_ailut_transform(img, LUTs_d, vertices_d) * self.dark_prob(img)
        )
        return out


# --------------------------------------------------------------------------------------
# Loading / Architecture detection
# --------------------------------------------------------------------------------------
ARCH_AGCM = "HDRTVNet-AGCM"
ARCH_ENSEMBLE = "HDRTVNet-Plus-Ensemble"
ARCH_LE_PLUS = "HDRTVNet-Plus-LE"
ARCH_LE = "HDRTVNet-LE"
ARCH_HG = "HDRTVNet-HG"
ARCH_HDRTVDM = "HDRTVDM-LSN"
ARCH_ITMLUT = "ITMLUT-AiLUT"


def _clean_state_dict(sd: dict) -> dict:
    for key in ("params_ema", "params", "state_dict", "model"):
        if key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def detect_architecture(sd: dict) -> str:
    # 0. ITMLUT (Adaptive Interval 3D-LUT)
    if "encoder_img_b.enocoder.0.weight" in sd and "lut_gen_b.basic_luts" in sd:
        return ARCH_ITMLUT
    # 1. HDRTVDM LSN (TriSegNet w. Restormer)
    if ("bright_branch.patch_embed.proj.weight" in sd or "bright_branch.output.weight" in sd) and "mid_branch.conv1.weight" in sd:
        return ARCH_HDRTVDM
    # 2. Ensemble model (AGCM + LE)
    if any(k.startswith("AGCM.") for k in sd) and any(k.startswith("LE.") for k in sd):
        return ARCH_ENSEMBLE
    # 3. Standalone AGCM
    if "classifier.model.0.weight" in sd and "cond_scale_first.weight" in sd:
        return ARCH_AGCM
    # 4. HDRTVNet++ standalone LE (HDRUNet3T1 with SFT)
    if "SFT_layer1.SFT_scale_conv0.weight" in sd and "conv_first.weight" in sd:
        return ARCH_LE_PLUS
    # 5. HDRTVNet original LE (SRResNet)
    if "recon_trunk.0.conv1.weight" in sd and "upconv.weight" in sd:
        return ARCH_LE
    # 6. HDRTVNet Highlight Generation (Hallucination Generator)
    if "conv_code1.0.weight" in sd and "conv10.weight" in sd:
        return ARCH_HG

    sample = ", ".join(list(sd.keys())[:8])
    raise ValueError(f"Unsupported SDR-to-HDR model architecture. First keys: {sample}")


def build_model(sd: dict) -> tuple[nn.Module, str]:
    sd = _clean_state_dict(sd)
    arch = detect_architecture(sd)

    if arch == ARCH_ITMLUT:
        model: nn.Module = ITMLUTNet()
    elif arch == ARCH_HDRTVDM:
        model = TriSegNet()
    elif arch == ARCH_ENSEMBLE:
        cond_c = sd["AGCM.cond_scale_first.weight"].shape[1]
        nf = sd["LE.conv_first.weight"].shape[0]
        model = EnsembleAGCMLE(cond_c=cond_c, nf=nf)
    elif arch == ARCH_AGCM:
        nf = sd["conv_first.weight"].shape[0]
        cond_c = sd["cond_scale_first.weight"].shape[1]
        model = AGCMNet(nf=nf, cond_c=cond_c)
    elif arch == ARCH_LE_PLUS:
        nf = sd["conv_first.weight"].shape[0]
        model = HDRUNet3T1(in_nc=3, out_nc=3, nf=nf, act_type="relu", weighting_network=False)
    elif arch == ARCH_HG:
        nf = sd["conv1.0.weight"].shape[0]
        model = HallucinationGenerator(nf=nf)
    else:  # ARCH_LE
        nf, in_nc = sd["conv_first.weight"].shape[:2]
        out_nc = sd["conv_last.weight"].shape[0]
        nb = 1 + max(
            int(m.group(1)) for k in sd if (m := re.match(r"recon_trunk\.(\d+)\.", k))
        )
        model = LENet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb)

    model.load_state_dict(sd, strict=True)
    model.eval().requires_grad_(False)
    return model, arch


@dataclass
class SDRtoHDRModelBundle:
    """Object passed between the loader node and the inference node."""
    agcm: AGCMNet | None = None
    le: nn.Module | None = None
    model: nn.Module | None = None
    arch: str = ARCH_ENSEMBLE
    names: list[str] = field(default_factory=list)
    is_ensemble: bool = False

    def modules(self) -> list[nn.Module]:
        return [m for m in (self.agcm, self.le, self.model) if m is not None]

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "SDRtoHDRModelBundle":
        for m in self.modules():
            m.to(device=device, dtype=dtype)
        return self


def load_model_file(path: str) -> tuple[nn.Module, str]:
    import comfy.utils

    sd = comfy.utils.load_torch_file(path, safe_load=True)
    model, arch = build_model(sd)
    logger.info("[XENodes] Loaded SDR-to-HDR model %s (%s)", os.path.basename(path), arch)
    return model, arch


# --------------------------------------------------------------------------------------
# Inference (compatible with static images and video frames of any resolution)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def run_sdr_to_hdr_frame(
    bundle: SDRtoHDRModelBundle,
    frame_bchw: torch.Tensor,
    condition_scale: int = 4,
    use_refine: bool = True,
    prev_condition: torch.Tensor | None = None,
    condition_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Runs SDR-to-HDR inference on a batch of SDR frames.
    Supports ITMLUT (Neural 3D-LUT), HDRTVDM (TriSegNet w. Restormer), and HDRTVNet (AGCM + LE).

    Args:
        frame_bchw: SDR BT.709 frames, (B, 3, H, W), [0, 1].
        condition_scale: Downscale factor for the global condition image (HDRTVNet uses 4).
        prev_condition: Condition vector of the previous frame (for temporal smoothing).
        condition_smoothing: EMA factor in [0, 1). 0 = per-frame condition (no smoothing).
    Returns:
        (BT.2020 PQ frames (B, 3, H, W) in [0, 1], condition vector used or None)
    """
    # 0. Standalone Highlight Generation model (HG)
    if bundle.arch == ARCH_HG and bundle.model is not None:
        mask = generate_highlight_mask(frame_bchw)
        out = bundle.model(frame_bchw, mask)
        return out.clamp(0.0, 1.0), None

    # 1. End-to-end models: ITMLUT (Neural 3D-LUT) or HDRTVDM (Restormer Transformer-UNet)
    if bundle.arch in (ARCH_HDRTVDM, ARCH_ITMLUT) and bundle.model is not None:
        out = bundle.model(frame_bchw)
        return out.clamp(0.0, 1.0), None

    # 2. HDRTVNet (AGCM + optional LE refinement)
    assert bundle.agcm is not None, "AGCM module required for HDRTVNet inference"
    h, w = frame_bchw.shape[-2:]
    cond_h = max(1, round(h / condition_scale))
    cond_w = max(1, round(w / condition_scale))
    short = min(cond_h, cond_w)
    if short < _MIN_COND_SIZE:
        k = _MIN_COND_SIZE / short
        cond_h, cond_w = max(_MIN_COND_SIZE, round(cond_h * k)), max(_MIN_COND_SIZE, round(cond_w * k))
    if (cond_h, cond_w) != (h, w):
        cond = F.interpolate(frame_bchw, size=(cond_h, cond_w), mode="bicubic", align_corners=False, antialias=True)
        cond = cond.clamp(0.0, 1.0)
    else:
        cond = frame_bchw

    fea = bundle.agcm.compute_condition(cond)
    if prev_condition is not None and condition_smoothing > 0.0:
        fea = condition_smoothing * prev_condition + (1.0 - condition_smoothing) * fea

    out = bundle.agcm.forward_with_condition(frame_bchw, fea)
    if use_refine and bundle.le is not None:
        if isinstance(bundle.le, HDRUNet3T1):
            out = bundle.le((out, out))
        else:
            out = bundle.le(out)

    return out.clamp(0.0, 1.0), fea
