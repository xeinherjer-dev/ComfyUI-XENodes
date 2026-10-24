"""
AI-based Inverse Tone Mapping (SDR -> HDR) model loading helpers and pipeline orchestration.

Model architectures are organized into dedicated modules under `utils.hdr_arch`:
  - `hdrtvnet.py`: HDRTVNet & HDRTVNet++ (MIT License)
  - `hdrtvdm.py`: HDRTVDM TriSegNet (MPL-2.0 License, Restormer components MIT License)
  - `itmlut.py`: ITMLUT Adaptive Interval 3D-LUT (MPL-2.0 License, AdaInt components Apache-2.0)

All architecture classes and functions are re-exported from this module for backward compatibility.
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
# Model Architectures (re-exported from utils.hdr_arch for backward compatibility)
# --------------------------------------------------------------------------------------
from .hdr_arch import (
    # HDRTVNet / HDRTVNet++ (MIT License)
    AGCMNet,
    EnsembleAGCMLE,
    HDRUNet3T1,
    HallucinationGenerator,
    LENet,
    ResBlockWithSFT,
    SFTLayer,
    _ColorCondition,
    _ResidualBlockNoBN,
    _color_block,
    _hg_conv_bn_relu,
    _hg_up_pixel_shuffle,
    _make_res_layers,
    generate_highlight_mask,
    # HDRTVDM (MPL-2.0 License, Restormer components MIT License)
    BiasFree_LayerNorm,
    BrightIntensityLum,
    Cond_Trunk,
    DarkIntensityLum,
    PWConv_Cond,
    Restormer,
    RestormerAttention,
    RestormerDownsample,
    RestormerFeedForward,
    RestormerLayerNorm,
    RestormerOverlapPatchEmbed,
    RestormerTransformerBlock,
    RestormerUpsample,
    TriSegNet,
    WithBias_LayerNorm,
    _which_act,
    # ITMLUT (MPL-2.0 License, AdaInt components Apache-2.0)
    ITMLUT_BasicResBlock,
    ITMLUT_BrightProb,
    ITMLUT_Classifier,
    ITMLUT_DarkProb,
    ITMLUT_GlobPriorEncoder,
    ITMLUT_LUTGenAndFuse,
    ITMLUT_MidProb,
    ITMLUTNet,
    _itmlut_mid_curve,
    pure_ailut_transform,
)



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
