import torch

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
            - linear_hdr: The tone-mapped linear HDR image tensor.
            - ratio: The per-pixel expansion ratio applied to the luminance.
            - scale: The maximum scale factor (peak_nits / 100.0).
    """
    # Convert sRGB to Linear (IEC 61966-2-1 standard)
    linear = torch.where(frame_tensor <= 0.04045, frame_tensor / 12.92, ((frame_tensor + 0.055) / 1.055) ** 2.4)
    
    # Apply Soft-Knee Inverse Tone Mapping (Power Curve)
    # We calculate expansion based on luminance to preserve color (hue)
    luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=linear.device)
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
