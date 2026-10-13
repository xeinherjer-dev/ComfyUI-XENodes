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
                "Supports Ensemble_AGCM_LE.pth (recommended for anime/video), "
                "HDRTVDM_params.pth (CVPR 2023 Restormer/Transformer standard baseline), "
                "HDRTVDM_params_3DM.pth (CVPR 2023 Restormer for heavily compressed real-world SDR), or "
                "ITMLUT_params.pth (ultra-fast Neural 3D-LUT for real-time video). "
                "Default models are downloaded automatically on first run if not present."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=files,
                    default="Ensemble_AGCM_LE.pth" if "Ensemble_AGCM_LE.pth" in files else files[0],
                    tooltip=(
                        "Model file (Ensemble_AGCM_LE.pth, HDRTVDM_params.pth, HDRTVDM_params_3DM.pth, ITMLUT_params.pth). "
                        "Automatically downloaded if missing."
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
        elif arch in (ARCH_HDRTVDM, ARCH_ITMLUT):
            bundle = SDRtoHDRModelBundle(model=model, arch=arch, names=[model_name])
        else:
            raise ValueError(
                f"[XENodes.LoadSDRtoHDRModel] '{model_name}' is {arch}, "
                f"expected {ARCH_AGCM}, {ARCH_ENSEMBLE}, {ARCH_HDRTVDM}, or {ARCH_ITMLUT}."
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

        model.to(device, dtype=torch.float32)
        pbar = comfy.utils.ProgressBar(frames.shape[0])
        outputs: list[torch.Tensor] = []
        prev_fea = None
        try:
            for i in range(frames.shape[0]):
                mm.throw_exception_if_processing_interrupted()
                rgb = frames[i : i + 1, ..., :3].to(device=device, dtype=torch.float32)
                pq, prev_fea = run_sdr_to_hdr_frame(
                    model,
                    rgb.permute(0, 3, 1, 2).contiguous(),
                    condition_scale=condition_scale,
                    use_refine=use_refine,
                    prev_condition=prev_fea,
                    condition_smoothing=condition_smoothing,
                )
                hdr = pq_to_hdr_output(pq.permute(0, 2, 3, 1), internal_algo, peak_nits)
                outputs.append(hdr.to(out_device))
                pbar.update(1)
        finally:
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        hdr_images = torch.cat(outputs, dim=0)
        if has_alpha:
            hdr_images = torch.cat([hdr_images, frames[..., 3:].to(hdr_images)], dim=-1)
        if is_single_image:
            hdr_images = hdr_images.squeeze(0)

        hdr_images, hdr_video = pack_hdr_outputs(hdr_images, src, comfy_color_space)
        return io.NodeOutput(hdr_images, hdr_video)


class SDRtoHDRAI(SDRtoHDRUsingModel):
    """Backwards-compatibility alias for previously saved workflows."""
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "XENodes.SDRtoHDRAI"
        schema.display_name = "SDR to HDR (AI) [Deprecated]"
        schema.is_deprecated = True
        schema.category = "xenodes/deprecated"
        return schema


class SDRtoHDRUsingModelExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LoadSDRtoHDRModel, SDRtoHDRUsingModel, SDRtoHDRAI]


async def comfy_entrypoint() -> SDRtoHDRUsingModelExtension:
    return SDRtoHDRUsingModelExtension()
