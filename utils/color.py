import functools
import math
import torch

_BT709_TO_BT2020_MATRIX = [
    [0.6274040786, 0.3292820974, 0.0433137970],
    [0.0690972331, 0.9195403953, 0.0113623716],
    [0.0163914389, 0.0880133075, 0.8955952536],
]

_BT709_LUMA_WEIGHTS = [0.2126, 0.7152, 0.0722]
_BT2020_LUMA_WEIGHTS = [0.2627, 0.6780, 0.0593]

_PINNED_STAGE_CACHE: dict[tuple[tuple[int, ...], torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def get_pinned_stage_pair(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a cached pair of pinned CPU staging buffers of at least `shape[0]` frames
    with matching spatial/channel dimensions `(shape[1], shape[2], shape[3])` and `dtype`.
    Avoids repeated 400+ ms `cudaHostAlloc` overhead across video conversions.
    """
    key = (shape[1:], dtype)
    pair = _PINNED_STAGE_CACHE.get(key)
    if pair is None or pair[0].shape[0] < shape[0]:
        # Keep only the most recent resolution/dtype in cache to bound pinned RAM usage
        _PINNED_STAGE_CACHE.clear()
        pair = (
            torch.empty(shape, dtype=dtype, pin_memory=True),
            torch.empty(shape, dtype=dtype, pin_memory=True),
        )
        _PINNED_STAGE_CACHE[key] = pair
    return pair[0][: shape[0]], pair[1][: shape[0]]


@functools.lru_cache(maxsize=16)
def _get_bt709_luma_weights(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_BT709_LUMA_WEIGHTS, device=device, dtype=dtype)


@functools.lru_cache(maxsize=16)
def _get_bt2020_luma_weights(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_BT2020_LUMA_WEIGHTS, device=device, dtype=dtype)


@functools.lru_cache(maxsize=16)
def _get_bt709_to_bt2020_matrix_t(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_BT709_TO_BT2020_MATRIX, device=device, dtype=dtype).t().contiguous()

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
    luma_weights = _get_bt709_luma_weights(linear.device, linear.dtype)
    luma = torch.sum(linear * luma_weights, dim=-1, keepdim=True)
    
    scale = peak_nits / 100.0
    if itm_knee < 1.0:
        # y = x + a * max(0, x - knee)^exponent
        # a = (scale - 1.0) / (1.0 - knee)^exponent
        a = (scale - 1.0) / ((1.0 - itm_knee) ** itm_exponent)
        luma_diff = torch.clamp(luma - itm_knee, min=0.0)
        if itm_exponent == 1.0:
            luma_hdr = luma + a * luma_diff
        elif itm_exponent == 2.0:
            luma_hdr = luma + a * (luma_diff * luma_diff)
        else:
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
    matrix_t = _get_bt709_to_bt2020_matrix_t(linear_tensor.device, linear_tensor.dtype)
    converted = torch.matmul(linear_tensor, matrix_t)
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

def adjust_hdr_luminance(
    linear_2020: torch.Tensor,
    exposure: float = 0.0,
    lift: float = 0.0,
) -> torch.Tensor:
    """
    Adjusts linear BT.2020 HDR luminance (exposure compensation and shadow lift).

    Args:
        linear_2020 (torch.Tensor): Linear BT.2020 HDR tensor (where 1.0 = 100 nits).
        exposure (float): Exposure compensation in EV stops. +1.0 doubles luminance (+1 stop), -1.0 halves it.
                          0.0 = unchanged.
        lift (float): Shadow and midtone brightness lift (floor boost). 0.0 = off.
                      Smoothly raises darker areas while tapering off as luminance reaches highlight levels.
                      Applied as an achromatic pedestal to avoid amplifying chroma noise in near-black pixels.

    Returns:
        torch.Tensor: Adjusted linear BT.2020 tensor.
    """
    if exposure == 0.0 and lift == 0.0:
        return linear_2020

    out = linear_2020
    if exposure != 0.0:
        out = out * (2.0 ** exposure)

    if lift > 0.0:
        luma_weights = _get_bt2020_luma_weights(linear_2020.device, linear_2020.dtype)
        luma = torch.sum(out * luma_weights, dim=-1, keepdim=True)
        boost = lift * torch.exp(-torch.clamp(luma, min=0.0) / 2.0)
        out = out + boost

    return out

def _scalar_linear_to_pq(nits: float) -> float:
    """Fast scalar SMPTE ST 2084 (PQ) OETF for peak luminance clamping."""
    if nits <= 0.0:
        return 0.0
    if nits >= 10000.0:
        return 1.0
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    y_m1 = (nits / 10000.0) ** m1
    return min(1.0, max(0.0, ((c1 + c2 * y_m1) / (1.0 + c3 * y_m1)) ** m2))

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
    # Fast path: when target is PQ and no linear luminance adjustments are requested,
    # pq_to_linear -> clamp -> linear_to_pq reduces to a direct clamp in the monotonic PQ domain.
    if exposure == 0.0 and lift == 0.0 and not is_hlg_color_space(color_space):
        max_pq = _scalar_linear_to_pq(peak_nits)
        return torch.clamp(pq_2020, 0.0, max_pq)

    linear_2020 = pq_to_linear(pq_2020, nits_scale=100.0)
    if exposure != 0.0 or lift != 0.0:
        linear_2020 = adjust_hdr_luminance(linear_2020, exposure=exposure, lift=lift)
    linear_2020 = torch.clamp(linear_2020, 0.0, peak_nits / 100.0)
    return encode_linear_bt2020(linear_2020, color_space, peak_nits)

def _sdr_to_hdr_chunk(
    chunk: torch.Tensor,
    luma_weights: torch.Tensor,
    bt709_to_bt2020_t: torch.Tensor,
    peak_nits: float,
    itm_knee: float,
    itm_exponent: float,
    is_hlg: bool,
) -> torch.Tensor:
    """Optimized kernel for sRGB -> Linear -> Soft-Knee ITM -> BT.2020 -> PQ/HLG."""
    linear = torch.where(chunk <= 0.04045, chunk / 12.92, torch.pow((chunk + 0.055) / 1.055, 2.4))
    if itm_knee < 1.0:
        luma = torch.sum(linear * luma_weights, dim=-1, keepdim=True)
        scale = peak_nits / 100.0
        a = (scale - 1.0) / ((1.0 - itm_knee) ** itm_exponent)
        luma_diff = torch.clamp(luma - itm_knee, min=0.0)
        if itm_exponent == 1.0:
            luma_hdr = luma + a * luma_diff
        elif itm_exponent == 2.0:
            luma_hdr = luma + a * (luma_diff * luma_diff)
        else:
            luma_hdr = luma + a * torch.pow(luma_diff, itm_exponent)
        ratio = (luma_hdr + 1e-6) / (luma + 1e-6)
        linear.mul_(ratio)
    linear_2020 = torch.matmul(linear, bt709_to_bt2020_t).clamp_(min=0.0)
    if is_hlg:
        a_h = 0.17883277
        b_h = 1.0 - 4.0 * a_h
        c_h = 0.5 - a_h * math.log(4.0 * a_h)
        scale_h = max(1.0, peak_nits / 100.0)
        E = linear_2020.div_(scale_h).clamp_(min=0.0, max=1.0)
        hlg = torch.where(
            E <= (1.0 / 12.0),
            torch.sqrt(3.0 * E),
            a_h * torch.log(torch.clamp(12.0 * E - b_h, min=1e-7)) + c_h,
        )
        return hlg.clamp_(0.0, 1.0)
    else:
        Y = (linear_2020 * 100.0) / 10000.0
        Y_pos = torch.clamp(Y, min=0.0)
        Y_m1 = torch.pow(Y_pos, 0.1593017578125)
        num = 0.8359375 + 18.8515625 * Y_m1
        den = 1.0 + 18.6875 * Y_m1
        pq = num.div_(den).pow_(78.84375)
        pq = torch.where(Y <= 0.0, torch.zeros_like(Y), pq)
        return pq.clamp_(0.0, 1.0)


def sdr_to_hdr_tensor(
    images: torch.Tensor,
    peak_nits: float = 400.0,
    itm_knee: float = 0.0,
    itm_exponent: float = 1.0,
    color_space: str = "HDR PQ",
    chunk_size: int = 8,
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

    has_alpha = images.shape[-1] == 4
    total_frames = images.shape[0]
    out_device = images.device
    compute_device = torch.device("cuda") if torch.cuda.is_available() else out_device
    is_hlg = is_hlg_color_space(color_space)

    # Pre-allocate output tensor on the original device to avoid double-memory peak and torch.cat copy
    result = torch.empty_like(images)
    if has_alpha:
        result[..., 3:].copy_(images[..., 3:])

    luma_weights = _get_bt709_luma_weights(compute_device, torch.float32)
    matrix_t = _get_bt709_to_bt2020_matrix_t(compute_device, torch.float32)
    peak_f = float(peak_nits)
    knee_f = float(itm_knee)
    exp_f = float(itm_exponent)

    use_pinned_stage = (
        out_device.type == "cpu"
        and compute_device.type == "cuda"
        and images.numel() >= 4 * 1024 * 1024
    )
    if not use_pinned_stage:
        with torch.no_grad():
            for start in range(0, total_frames, chunk_size):
                end = min(start + chunk_size, total_frames)
                chunk = images[start:end, ..., :3].to(device=compute_device, dtype=torch.float32, non_blocking=True)
                hdr_chunk = _sdr_to_hdr_chunk(chunk, luma_weights, matrix_t, peak_f, knee_f, exp_f, is_hlg)
                result[start:end, ..., :3].copy_(hdr_chunk, non_blocking=False)
        if is_single_image:
            result = result.squeeze(0)
        return result

    stage_len = min(chunk_size, total_frames)
    pinned_pair = get_pinned_stage_pair((stage_len, images.shape[1], images.shape[2], 3), result.dtype)
    events = (torch.cuda.Event(), torch.cuda.Event())
    pending: tuple[int, int, int] | None = None

    with torch.no_grad():
        for idx, start in enumerate(range(0, total_frames, chunk_size)):
            end = min(start + chunk_size, total_frames)
            cur_len = end - start
            slot = idx & 1

            chunk = images[start:end, ..., :3].to(device=compute_device, dtype=torch.float32, non_blocking=True)
            hdr_chunk = _sdr_to_hdr_chunk(chunk, luma_weights, matrix_t, peak_f, knee_f, exp_f, is_hlg)
            st = pinned_pair[slot][:cur_len]
            st.copy_(hdr_chunk, non_blocking=True)
            events[slot].record()

            # Overlap GPU execution & DMA of current chunk with CPU write (and OS page faults) of previous chunk
            if pending is not None:
                p_slot, p_start, p_end = pending
                events[p_slot].synchronize()
                result[p_start:p_end, ..., :3].copy_(pinned_pair[p_slot][: p_end - p_start])

            pending = (slot, start, end)

        if pending is not None:
            p_slot, p_start, p_end = pending
            events[p_slot].synchronize()
            result[p_start:p_end, ..., :3].copy_(pinned_pair[p_slot][: p_end - p_start])

    if is_single_image:
        result = result.squeeze(0)

    return result
