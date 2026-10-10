"""
AI-based Inverse Tone Mapping (SDR -> HDR) model architectures and loading helpers.

Supported architectures (auto-detected from the state_dict):
  1. HDRTVNet-AGCM:
     - Adaptive Global Color Mapping (ConditionNet) from ICCV 2021
     - E.g. AGCM.pth / AGCM_G.pth
  2. HDRTVNet-Plus-Ensemble:
     - End-to-end AGCM + Local Enhancement (HDRUNet3T1) from HDRTVNet++
     - E.g. Ensemble_AGCM_LE.pth / HR.pth (single file with both global & local models)
  3. HDRTVNet-Plus-LE:
     - Standalone Local Enhancement (HDRUNet3T1 with Spatial Feature Transform)
     - E.g. LE.pth
  4. HDRTVNet-LE:
     - Original modified SRResNet Local Enhancement from ICCV 2021
     - E.g. LE_weights.pth

Both static images and arbitrary-resolution video frames are supported.
Spatial dimensions for LE are automatically padded to multiples of 8 and unpadded on output.
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
    "AGCM.pth": (
        "https://github.com/xiaom233/HDRTVNet-plus/raw/main/pretrained_models/AGCM.pth"
    ),
    "LE.pth": (
        "https://github.com/xiaom233/HDRTVNet-plus/raw/main/pretrained_models/LE.pth"
    ),
}

DEFAULT_MODELS: list[str] = [
    "Ensemble_AGCM_LE.pth",
    "AGCM.pth",
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

            print(f"\n[XENodes] Successfully saved to: {dest_path}")
        shutil.move(tmp_path, dest_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"[XENodes] Failed to download model from {url}: {e}"
        ) from e


def get_available_models() -> list[str]:
    """Returns available model filenames, ensuring default models appear in UI even before download."""
    import folder_paths

    existing = folder_paths.get_filename_list(MODEL_FOLDER_NAME)
    models: list[str] = list(DEFAULT_MODELS)
    for name in existing:
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
# Loading / Architecture detection
# --------------------------------------------------------------------------------------
ARCH_AGCM = "HDRTVNet-AGCM"
ARCH_ENSEMBLE = "HDRTVNet-Plus-Ensemble"
ARCH_LE_PLUS = "HDRTVNet-Plus-LE"
ARCH_LE = "HDRTVNet-LE"


def _clean_state_dict(sd: dict) -> dict:
    for key in ("params_ema", "params", "state_dict", "model"):
        if key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def detect_architecture(sd: dict) -> str:
    # 1. Ensemble model (AGCM + LE)
    if any(k.startswith("AGCM.") for k in sd) and any(k.startswith("LE.") for k in sd):
        return ARCH_ENSEMBLE
    # 2. Standalone AGCM
    if "classifier.model.0.weight" in sd and "cond_scale_first.weight" in sd:
        return ARCH_AGCM
    # 3. HDRTVNet++ standalone LE (HDRUNet3T1 with SFT)
    if "SFT_layer1.SFT_scale_conv0.weight" in sd and "conv_first.weight" in sd:
        return ARCH_LE_PLUS
    # 4. HDRTVNet original LE (SRResNet)
    if "recon_trunk.0.conv1.weight" in sd and "upconv.weight" in sd:
        return ARCH_LE

    sample = ", ".join(list(sd.keys())[:8])
    raise ValueError(f"Unsupported SDR-to-HDR model architecture. First keys: {sample}")


def build_model(sd: dict) -> tuple[nn.Module, str]:
    sd = _clean_state_dict(sd)
    arch = detect_architecture(sd)

    if arch == ARCH_ENSEMBLE:
        cond_c = sd["AGCM.cond_scale_first.weight"].shape[1]
        nf = sd["LE.conv_first.weight"].shape[0]
        model: nn.Module = EnsembleAGCMLE(cond_c=cond_c, nf=nf)
    elif arch == ARCH_AGCM:
        nf = sd["conv_first.weight"].shape[0]
        cond_c = sd["cond_scale_first.weight"].shape[1]
        model = AGCMNet(nf=nf, cond_c=cond_c)
    elif arch == ARCH_LE_PLUS:
        nf = sd["conv_first.weight"].shape[0]
        model = HDRUNet3T1(in_nc=3, out_nc=3, nf=nf, act_type="relu", weighting_network=False)
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
    agcm: AGCMNet
    le: nn.Module | None = None
    names: list[str] = field(default_factory=list)
    is_ensemble: bool = False

    def modules(self) -> list[nn.Module]:
        return [m for m in (self.agcm, self.le) if m is not None]

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs AGCM (+ optional LE refinement) on a batch of SDR frames.

    Args:
        frame_bchw: SDR BT.709 frames, (B, 3, H, W), [0, 1].
        condition_scale: Downscale factor for the global condition image (HDRTVNet uses 4).
        prev_condition: Condition vector of the previous frame (for temporal smoothing).
        condition_smoothing: EMA factor in [0, 1). 0 = per-frame condition (no smoothing).
    Returns:
        (BT.2020 PQ frames (B, 3, H, W) in [0, 1], condition vector used)
    """
    h, w = frame_bchw.shape[-2:]
    cond_h = max(1, round(h / condition_scale))
    cond_w = max(1, round(w / condition_scale))
    # The condition network halves the resolution 4 times before its last InstanceNorm,
    # which needs > 1 spatial element: keep the short side >= _MIN_COND_SIZE.
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
