import os
import sys
import unittest
import torch

# Ensure ComfyUI and xenodes directories are in sys.path
XENODES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_NODES_ROOT = os.path.dirname(XENODES_ROOT)
COMFYUI_ROOT = os.path.dirname(CUSTOM_NODES_ROOT)

for p in [COMFYUI_ROOT, CUSTOM_NODES_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def create_dummy_image(batch: int = 1, height: int = 32, width: int = 32, channels: int = 3) -> torch.Tensor:
    """Creates a dummy float32 image tensor in [0, 1] range."""
    return torch.rand((batch, height, width, channels), dtype=torch.float32)


def create_dummy_audio(channels: int = 2, samples: int = 44100, sample_rate: int = 44100) -> dict:
    """Creates a dummy audio dict matching ComfyUI Audio input."""
    waveform = torch.zeros((channels, samples), dtype=torch.float32)
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }
