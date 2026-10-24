"""
Neural inverse tone mapping (SDR to HDR) model architectures.

Modules:
- hdrtvnet: HDRTVNet & HDRTVNet++ architectures (MIT License)
- hdrtvdm: HDRTVDM TriSegNet architecture (MPL-2.0 License, Restormer components MIT License)
- itmlut: ITMLUT Adaptive Interval 3D-LUT architecture (MPL-2.0 License, AdaInt components Apache-2.0)
"""
from __future__ import annotations

from .hdrtvnet import (
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
)
from .hdrtvdm import (
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
)
from .itmlut import (
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

__all__ = [
    # HDRTVNet / HDRTVNet++
    "AGCMNet",
    "EnsembleAGCMLE",
    "HDRUNet3T1",
    "HallucinationGenerator",
    "LENet",
    "ResBlockWithSFT",
    "SFTLayer",
    "_ColorCondition",
    "_ResidualBlockNoBN",
    "_color_block",
    "_hg_conv_bn_relu",
    "_hg_up_pixel_shuffle",
    "_make_res_layers",
    "generate_highlight_mask",
    # HDRTVDM
    "BiasFree_LayerNorm",
    "BrightIntensityLum",
    "Cond_Trunk",
    "DarkIntensityLum",
    "PWConv_Cond",
    "Restormer",
    "RestormerAttention",
    "RestormerDownsample",
    "RestormerFeedForward",
    "RestormerLayerNorm",
    "RestormerOverlapPatchEmbed",
    "RestormerTransformerBlock",
    "RestormerUpsample",
    "TriSegNet",
    "WithBias_LayerNorm",
    "_which_act",
    # ITMLUT
    "ITMLUT_BasicResBlock",
    "ITMLUT_BrightProb",
    "ITMLUT_Classifier",
    "ITMLUT_DarkProb",
    "ITMLUT_GlobPriorEncoder",
    "ITMLUT_LUTGenAndFuse",
    "ITMLUT_MidProb",
    "ITMLUTNet",
    "_itmlut_mid_curve",
    "pure_ailut_transform",
]
