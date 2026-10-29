from __future__ import annotations

from typing_extensions import override
import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from comfy_api.latest import ComfyExtension, io, Input
from ..utils.color import pq_to_hdr_output
from ..utils.hdr_io import (
    COLOR_SPACE_OPTIONS,
    resolve_color_space,
    unpack_sdr_inputs,
    pack_hdr_outputs,
)
from ..utils.hdr_models import (
    ARCH_AGCM,
    ARCH_ENSEMBLE,
    ARCH_HDRTVDM,
    ARCH_HG,
    ARCH_ITMLUT,
    ARCH_LE,
    ARCH_LE_PLUS,
    MODEL_FOLDER_NAME,
    EnsembleAGCMLE,
    SDRtoHDRModelBundle,
    get_available_models,
    load_model_file,
    register_model_folder,
    resolve_or_download_model,
    run_sdr_to_hdr_frame,
)

register_model_folder()

SDRtoHDRModel = io.Custom("XE_SDR2HDR_MODEL")
_NONE = "None"


class LoadSDRtoHDRModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = get_available_models()
        return io.Schema(
            node_id="XENodes.LoadSDRtoHDRModel",
            display_name="Load SDR to HDR Model [Beta]",
            category="xenodes/color",
            is_experimental=True,
            description=(
                f"[BETA] Loads an inverse tone mapping model from 'models/{MODEL_FOLDER_NAME}'. "
                "Default recommended models (downloaded automatically if missing):\n"
                "- HR.pth: Highest visual quality (HDRTVNet++, MIT License). GAN-refined highlights & vivid BT.2020 color (best for anime/CG/art).\n"
                "- HDRTVDM_params_3DM.pth: Robust Transformer-UNet (HDRTVDM, MPL-2.0 License). Suppresses real-world compression artifacts (best for live-action/video).\n"
                "- ITMLUT_params.pth: Ultra-fast 3D-LUT (ITMLUT, MPL-2.0 License). Real-time inference (best for long videos/low VRAM)."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=files,
                    default="HR.pth" if "HR.pth" in files else files[0],
                    tooltip=(
                        "Model file: 'HR.pth' (highest quality / anime / CG), "
                        "'HDRTVDM_params_3DM.pth' (live-action / video / artifact restoration), "
                        "'ITMLUT_params.pth' (ultra-fast real-time 3D-LUT). Automatically downloaded if missing."
                    ),
                ),
            ],
            outputs=[SDRtoHDRModel.Output("model")],
        )

    @classmethod
    def execute(cls, model_name: str) -> io.NodeOutput:
        path = resolve_or_download_model(model_name)
        model, arch = load_model_file(path)

        if arch == ARCH_ENSEMBLE:
            assert isinstance(model, EnsembleAGCMLE)
            agcm = model.AGCM
            le = model.LE
            is_ensemble = True
            bundle = SDRtoHDRModelBundle(agcm=agcm, le=le, arch=arch, names=[model_name], is_ensemble=is_ensemble)
        elif arch == ARCH_AGCM:
            agcm = model
            le = None
            is_ensemble = False
            bundle = SDRtoHDRModelBundle(agcm=agcm, le=le, arch=arch, names=[model_name], is_ensemble=is_ensemble)
        elif arch in (ARCH_HDRTVDM, ARCH_ITMLUT, ARCH_HG):
            bundle = SDRtoHDRModelBundle(model=model, arch=arch, names=[model_name])
        else:
            raise ValueError(
                f"[XENodes.LoadSDRtoHDRModel] '{model_name}' is {arch}, "
                f"expected {ARCH_AGCM}, {ARCH_ENSEMBLE}, {ARCH_HDRTVDM}, {ARCH_ITMLUT}, or {ARCH_HG}."
            )

        bundle.to(mm.unet_offload_device())
        return io.NodeOutput(bundle)


class SDRtoHDRUsingModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SDRtoHDRUsingModel",
            display_name="SDR to HDR (using Model) [Beta]",
            category="xenodes/color",
            is_experimental=True,
            search_aliases=["XENodes.SDRtoHDRAI", "SDR to HDR (AI)"],
            description=(
                "[BETA] Model-based SDR to HDR conversion (BT.709 SDR -> BT.2020 PQ/HLG) using a learned inverse tone "
                "mapping model. Outputs are compatible with the algorithmic 'SDR to HDR' node."
            ),
            inputs=[
                SDRtoHDRModel.Input("model", tooltip="Model from 'Load SDR to HDR Model [Beta]'."),
                io.Image.Input("images", optional=True, tooltip="The SDR images or video frames to expand."),
                io.Video.Input("video", optional=True, tooltip="Optional: The SDR video to expand."),
                io.Float.Input(
                    "peak_nits", default=1000.0, min=100.0, max=10000.0, step=1.0,
                    tooltip="Output luminance is clipped to this peak. HDRTVNet was trained on 1000-nit HDR10 content.",
                ),
                io.Float.Input(
                    "exposure", default=0.0, min=-3.0, max=3.0, step=0.1,
                    tooltip="Exposure adjustment in EV stops applied in linear HDR space. +1.0 = twice as bright (+1 stop), -1.0 = half as bright. Default 0.0 = pure AI output.",
                ),
                io.Float.Input(
                    "lift", default=0.0, min=0.0, max=0.5, step=0.01,
                    tooltip="Shadow & midtone brightness lift (floor boost). 0.0 = off. Useful to raise dark or muddy scenes.",
                ),
                io.Combo.Input(
                    "color_space",
                    options=COLOR_SPACE_OPTIONS,
                    default="HDR PQ",
                    tooltip="Target HDR color space. The model natively predicts PQ; HLG is derived from it.",
                ),
                io.Int.Input(
                    "condition_scale", default=4, min=1, max=16, step=1, advanced=True,
                    tooltip="Downscale factor of the global condition image. HDRTVNet was trained with 4.",
                ),
                io.Float.Input(
                    "condition_smoothing", default=0.0, min=0.0, max=0.99, step=0.01, advanced=True,
                    tooltip="Temporal EMA of the global condition across frames to reduce brightness flicker in video. 0 = off.",
                ),
                io.Boolean.Input(
                    "use_refine", default=True, advanced=True,
                    tooltip="Apply the LE refinement model if one was loaded.",
                ),
            ],
            outputs=[
                io.Image.Output("images", tooltip="The expanded HDR images/frames [0, 1]."),
                io.Video.Output("video", tooltip="The expanded HDR video (10-bit, compatible with SaveVideo)."),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: SDRtoHDRModelBundle,
        images: Input.Image | None = None,
        video: Input.Video | None = None,
        peak_nits: float = 1000.0,
        exposure: float = 0.0,
        lift: float = 0.0,
        color_space: str = "HDR PQ",
        condition_scale: int = 4,
        condition_smoothing: float = 0.0,
        use_refine: bool = True,
    ) -> io.NodeOutput:
        src = unpack_sdr_inputs(images, video, "XENodes.SDRtoHDRUsingModel")
        internal_algo, comfy_color_space = resolve_color_space(color_space)

        frames = src.images
        is_single_image = frames.ndim == 3
        if is_single_image:
            frames = frames.unsqueeze(0)

        has_alpha = frames.shape[-1] == 4
        out_device = frames.device
        device = mm.get_torch_device()

        # Pre-allocate output tensor on the source device to avoid 2x RAM peak and torch.cat copy
        hdr_images = torch.empty_like(frames)
        if has_alpha:
            hdr_images[..., 3:].copy_(frames[..., 3:])

        use_pinned_stage = (out_device.type == "cpu" and device.type == "cuda")
        pinned_stage = (
            torch.empty((1, frames.shape[1], frames.shape[2], 3), dtype=hdr_images.dtype, pin_memory=True)
            if use_pinned_stage else None
        )

        model.to(device, dtype=torch.float32)
        pbar = comfy.utils.ProgressBar(frames.shape[0])
        prev_fea = None
        try:
            with torch.no_grad():
                for i in range(frames.shape[0]):
                    mm.throw_exception_if_processing_interrupted()
                    rgb = frames[i : i + 1, ..., :3].to(device=device, dtype=torch.float32, non_blocking=True)
                    pq, prev_fea = run_sdr_to_hdr_frame(
                        model,
                        rgb.permute(0, 3, 1, 2).contiguous(),
                        condition_scale=condition_scale,
                        use_refine=use_refine,
                        prev_condition=prev_fea,
                        condition_smoothing=condition_smoothing,
                    )
                    hdr = pq_to_hdr_output(
                        pq.permute(0, 2, 3, 1),
                        internal_algo,
                        peak_nits,
                        exposure=exposure,
                        lift=lift,
                    )
                    if pinned_stage is not None:
                        pinned_stage.copy_(hdr, non_blocking=False)
                        hdr_images[i : i + 1, ..., :3].copy_(pinned_stage)
                    else:
                        hdr_images[i : i + 1, ..., :3].copy_(hdr, non_blocking=False)
                    pbar.update(1)
        finally:
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        if is_single_image:
            hdr_images = hdr_images.squeeze(0)

        hdr_images, hdr_video = pack_hdr_outputs(hdr_images, src, comfy_color_space)
        return io.NodeOutput(hdr_images, hdr_video)


class SDRtoHDRUsingModelExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LoadSDRtoHDRModel, SDRtoHDRUsingModel]


async def comfy_entrypoint() -> SDRtoHDRUsingModelExtension:
    return SDRtoHDRUsingModelExtension()
