import math
import torch

_BT709_TO_BT2020_MATRIX = [
    [0.6274040786, 0.3292820974, 0.0433137970],
    [0.0690972331, 0.9195403953, 0.0113623716],
    [0.0163914389, 0.0880133075, 0.8955952536],
]

def apply_inverse_tone_mapping(frame_tensor: torch.Tensor, peak_nits: float, itm_knee: float, itm_exponent: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Applies Inverse Tone Mapping (SDR to HDR expansion) to an sRGB tensor.
    
    Args:
        frame_tensor (torch.Tensor): Input sRGB image tensor, expected to be in range [0, 1].
        peak_nits (float): Target peak luminance in nits. SDR white (1.0) maps to this value.
        itm_knee (float): Soft-knee threshold for expansion.
        itm_exponent (float): Expansion curve exponent.
        
    Returns:
        tuple[torch.Tensor, torch.Tensor, float]: 
            - linear_hdr: The tone-mapped linear HDR image tensor (1.0 = 100 nits, max = peak_nits / 100.0).
            - ratio: The per-pixel expansion ratio applied to the luminance.
            - scale: The maximum scale factor (peak_nits / 100.0).
    """
    # Convert sRGB to Linear (IEC 61966-2-1 standard)
    linear = torch.where(frame_tensor <= 0.04045, frame_tensor / 12.92, ((frame_tensor + 0.055) / 1.055) ** 2.4)
    
    # Apply Soft-Knee Inverse Tone Mapping (Power Curve)
    # We calculate expansion based on luminance to preserve color (hue)
    luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=linear.device, dtype=linear.dtype)
    luma = torch.sum(linear * luma_weights, dim=-1, keepdim=True)
    
    scale = peak_nits / 100.0
    if itm_knee < 1.0:
        # y = x + a * max(0, x - knee)^exponent
        # a = (scale - 1.0) / (1.0 - knee)^exponent
        a = (scale - 1.0) / ((1.0 - itm_knee) ** itm_exponent)
        luma_diff = torch.clamp(luma - itm_knee, min=0.0)
        luma_hdr = luma + a * (luma_diff ** itm_exponent)
        
        # Apply the same expansion ratio to all channels to preserve color
        ratio = (luma_hdr + 1e-6) / (luma + 1e-6)
        linear_hdr = linear * ratio
    else:
        linear_hdr = linear
        ratio = torch.ones_like(luma)
        
    return linear_hdr, ratio, scale

def convert_bt709_to_bt2020(linear_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts linear RGB from BT.709 primaries to BT.2020 primaries (D65 white point).
    """
    matrix = torch.tensor(_BT709_TO_BT2020_MATRIX, device=linear_tensor.device, dtype=linear_tensor.dtype)
    converted = torch.matmul(linear_tensor, matrix.t())
    return torch.clamp(converted, min=0.0)

def linear_to_pq(linear_hdr: torch.Tensor, nits_scale: float = 100.0) -> torch.Tensor:
    """
    Applies SMPTE ST 2084 (Perceptual Quantizer / PQ) OETF to a linear HDR tensor.
    
    Args:
        linear_hdr (torch.Tensor): Linear HDR values where 1.0 corresponds to nits_scale (default: 100 nits).
        nits_scale (float): Luminance corresponding to 1.0 in linear_hdr (default 100.0 nits).
    Returns:
        torch.Tensor: PQ encoded tensor normalized to [0.0, 1.0] (1.0 = 10,000 nits).
    """
    # SMPTE ST 2084 constants
    m1 = 0.1593017578125       # 2610 / 16384
    m2 = 78.84375              # (2523 / 4096) * 128
    c1 = 0.8359375             # 3424 / 4096
    c2 = 18.8515625            # (2413 / 4096) * 32
    c3 = 18.6875               # (2392 / 4096) * 32

    # Normalize to 0.0 - 1.0 representing 0 - 10,000 nits
    Y = (linear_hdr * nits_scale) / 10000.0
    Y_pos = torch.clamp(Y, min=0.0)
    
    Y_m1 = torch.pow(Y_pos, m1)
    num = c1 + c2 * Y_m1
    den = 1.0 + c3 * Y_m1
    pq = torch.pow(num / den, m2)
    
    # Clean zero mapping
    pq = torch.where(Y <= 0.0, torch.zeros_like(Y), pq)
    return torch.clamp(pq, 0.0, 1.0)

def linear_to_hlg(linear_hdr: torch.Tensor, peak_nits: float = 400.0) -> torch.Tensor:
    """
    Applies ARIB STD-B67 (Hybrid Log-Gamma / HLG) OETF to a linear HDR tensor.
    
    Args:
        linear_hdr (torch.Tensor): Linear HDR values where 1.0 corresponds to 100 nits, max to peak_nits/100.
        peak_nits (float): Peak brightness in nits used to normalize the scene light [0, 1].
    Returns:
        torch.Tensor: HLG encoded tensor normalized to [0.0, 1.0].
    """
    # ARIB STD-B67 constants
    a = 0.17883277
    b = 1.0 - 4.0 * a          # 0.28466892
    c = 0.5 - a * math.log(4.0 * a)  # 0.55991073

    # Normalize scene light E so that peak luminance maps to 1.0 in HLG
    scale = max(1.0, peak_nits / 100.0)
    E = torch.clamp(linear_hdr / scale, min=0.0, max=1.0)
    
    hlg = torch.where(
        E <= 1.0 / 12.0,
        torch.sqrt(3.0 * E),
        a * torch.log(torch.clamp(12.0 * E - b, min=1e-7)) + c
    )
    return torch.clamp(hlg, 0.0, 1.0)

def pq_to_linear(pq: torch.Tensor, nits_scale: float = 100.0) -> torch.Tensor:
    """
    Applies SMPTE ST 2084 (PQ) EOTF, the inverse of linear_to_pq.

    Args:
        pq (torch.Tensor): PQ encoded tensor in [0, 1] (1.0 = 10,000 nits).
        nits_scale (float): Luminance that 1.0 in the returned linear tensor represents (default 100 nits).
    Returns:
        torch.Tensor: Linear HDR tensor where 1.0 corresponds to nits_scale.
    """
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    e = torch.clamp(pq, 0.0, 1.0)
    e_p = torch.pow(e, 1.0 / m2)
    num = torch.clamp(e_p - c1, min=0.0)
    den = c2 - c3 * e_p
    y = torch.pow(num / den, 1.0 / m1)
    return y * (10000.0 / nits_scale)

def is_hlg_color_space(color_space: str) -> bool:
    """Returns True if the given color space string denotes HLG ("HDR" or contains "HLG")."""
    return "HLG" in color_space.upper() or color_space == "HDR"

def encode_linear_bt2020(linear_2020: torch.Tensor, color_space: str, peak_nits: float) -> torch.Tensor:
    """
    Encodes linear BT.2020 HDR (1.0 = 100 nits) with the target transfer function (PQ or HLG).
    Shared by the algorithmic (ITM) and AI-based SDR-to-HDR nodes.
    """
    if is_hlg_color_space(color_space):
        return linear_to_hlg(linear_2020, peak_nits=peak_nits)
    return linear_to_pq(linear_2020, nits_scale=100.0)

_BT2020_LUMA_WEIGHTS = [0.2627, 0.6780, 0.0593]

def adjust_hdr_luminance(
    linear_2020: torch.Tensor,
    exposure: float = 0.0,
    lift: float = 0.0,
) -> torch.Tensor:
    """
    Adjusts linear BT.2020 HDR luminance (exposure and shadow lift) while strictly preserving color/hue.

    Args:
        linear_2020 (torch.Tensor): Linear BT.2020 HDR tensor (where 1.0 = 100 nits).
        exposure (float): Exposure compensation in EV stops. +1.0 doubles luminance (+1 stop), -1.0 halves it.
                          0.0 = unchanged.
        lift (float): Shadow and midtone brightness lift (floor boost). 0.0 = off.
                      Smoothly raises darker areas while tapering off as luminance reaches highlight levels.

    Returns:
        torch.Tensor: Adjusted linear BT.2020 tensor.
    """
    if exposure == 0.0 and lift == 0.0:
        return linear_2020

    luma_weights = torch.tensor(_BT2020_LUMA_WEIGHTS, device=linear_2020.device, dtype=linear_2020.dtype)
    luma = torch.sum(linear_2020 * luma_weights, dim=-1, keepdim=True)

    new_luma = luma
    if exposure != 0.0:
        new_luma = new_luma * (2.0 ** exposure)

    if lift > 0.0:
        boost = lift * torch.exp(-torch.clamp(new_luma, min=0.0) / 2.0)
        new_luma = new_luma + boost

    ratio = torch.clamp((new_luma + 1e-6) / (luma + 1e-6), min=0.0)
    return linear_2020 * ratio

def pq_to_hdr_output(
    pq_2020: torch.Tensor,
    color_space: str,
    peak_nits: float,
    exposure: float = 0.0,
    lift: float = 0.0,
) -> torch.Tensor:
    """
    Converts a BT.2020 PQ tensor (e.g. the output of an AI ITM model) to the target HDR encoding.
    Optionally applies linear HDR luminance adjustments (exposure, shadow lift).
    Luminance is clipped to peak_nits so that the result respects the requested display peak.
    """
    linear_2020 = pq_to_linear(pq_2020, nits_scale=100.0)
    if exposure != 0.0 or lift != 0.0:
        linear_2020 = adjust_hdr_luminance(linear_2020, exposure=exposure, lift=lift)
    linear_2020 = torch.clamp(linear_2020, 0.0, peak_nits / 100.0)
    return encode_linear_bt2020(linear_2020, color_space, peak_nits)

def sdr_to_hdr_tensor(
    images: torch.Tensor,
    peak_nits: float = 400.0,
    itm_knee: float = 0.0,
    itm_exponent: float = 1.0,
    color_space: str = "HDR PQ",
    chunk_size: int = 16,
) -> torch.Tensor:
    """
    Full pipeline to expand SDR sRGB images [0, 1] to HDR BT.2020 (PQ or HLG) images [0, 1].
    
    Args:
        images (torch.Tensor): SDR sRGB images tensor (N, H, W, C) or (H, W, C), range [0, 1].
        peak_nits (float): Peak brightness in nits.
        itm_knee (float): Inverse Tone Mapping soft-knee threshold [0, 1].
        itm_exponent (float): Expansion curve exponent.
        color_space (str): Target HDR color space ("HDR PQ", "HDR HLG", etc.).
        chunk_size (int): Number of frames to process at once for memory efficiency.
        
    Returns:
        torch.Tensor: Tone-mapped and color-transformed HDR tensor in range [0, 1].
    """
    is_single_image = images.ndim == 3
    if is_single_image:
        images = images.unsqueeze(0)

    # Process RGB channels only (keep alpha untouched if present)
    has_alpha = images.shape[-1] == 4
    rgb_images = images[..., :3].contiguous()
    alpha = images[..., 3:] if has_alpha else None

    output_chunks = []
    total_frames = rgb_images.shape[0]

    for start in range(0, total_frames, chunk_size):
        chunk = rgb_images[start : start + chunk_size]
        
        # 1. sRGB -> Linear & Soft-Knee ITM (1.0 = 100 nits)
        linear_hdr, _, _ = apply_inverse_tone_mapping(chunk, peak_nits, itm_knee, itm_exponent)
        
        # 2. BT.709 Linear -> BT.2020 Linear
        linear_2020 = convert_bt709_to_bt2020(linear_hdr)
        
        # 3. Linear -> PQ or HLG OETF
        hdr_chunk = encode_linear_bt2020(linear_2020, color_space, peak_nits)

        output_chunks.append(hdr_chunk)

    hdr_rgb = torch.cat(output_chunks, dim=0)

    if has_alpha and alpha is not None:
        result = torch.cat([hdr_rgb, alpha], dim=-1)
    else:
        result = hdr_rgb

    if is_single_image:
        result = result.squeeze(0)

    return result
