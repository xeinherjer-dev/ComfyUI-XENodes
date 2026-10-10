from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import torch

from comfy_api.latest import Input, InputImpl, Types


COLOR_SPACE_OPTIONS = [
    "HDR",
    "HDR PQ",
]


def resolve_color_space(cs: str) -> tuple[str, str]:
    """
    Resolves UI color space string to:
      - internal_algo: "HDR" (HLG) or "HDR PQ" (PQ)
      - comfy_color_space: "HDR" (HLG) or "HDR PQ" (PQ) expected by ComfyUI VideoFromComponents
    """
    if "HLG" in cs.upper() or cs == "HDR":
        return "HDR", "HDR"
    return "HDR PQ", "HDR PQ"


@dataclass
class SDRSource:
    """SDR frames plus the stream metadata that must be carried over to the HDR output."""
    images: torch.Tensor
    alpha: Any = None
    audio: Any = None
    frame_rate: Fraction = Fraction(24, 1)


def unpack_sdr_inputs(
    images: Input.Image | None,
    video: Input.Video | None,
    node_name: str,
) -> SDRSource:
    """
    Picks the SDR frames from either `images` or `video`.
    If both are given, `images` is used for pixels and `video` supplies alpha/audio/frame rate.
    """
    if images is None and video is None:
        raise ValueError(f"[{node_name}] Either 'images' or 'video' must be provided.")

    if images is not None:
        src = SDRSource(images=images)
        if video is not None:
            comp = video.get_components()
            src.alpha = comp.alpha
            src.audio = comp.audio
            src.frame_rate = comp.frame_rate
        return src

    comp = video.get_components()
    return SDRSource(
        images=comp.images,
        alpha=comp.alpha,
        audio=comp.audio,
        frame_rate=comp.frame_rate,
    )


def pack_hdr_outputs(
    hdr_images: torch.Tensor,
    src: SDRSource,
    comfy_color_space: str,
) -> tuple[torch.Tensor, Input.Video]:
    """Tags the HDR image tensor and wraps it into a 10-bit HDR video."""
    try:
        hdr_images._color_space = comfy_color_space
    except Exception:
        pass

    hdr_video = InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=hdr_images,
            alpha=src.alpha,
            audio=src.audio,
            frame_rate=src.frame_rate,
        ),
        bit_depth=10,
        color_space=comfy_color_space,
    )
    return hdr_images, hdr_video
