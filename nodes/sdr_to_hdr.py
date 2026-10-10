from __future__ import annotations

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, Input
from ..utils.color import sdr_to_hdr_tensor
from ..utils.hdr_io import (
    COLOR_SPACE_OPTIONS,
    resolve_color_space as _resolve_color_space,
    unpack_sdr_inputs,
    pack_hdr_outputs,
)


class SDRtoHDR(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SDRtoHDR",
            display_name="SDR to HDR",
            category="xenodes/color",
            description="Expands SDR images, video frames, or video to HDR (Inverse Tone Mapping + BT.2020/PQ or HLG). Supports both static images and video.",
            inputs=[
                io.Image.Input("images", optional=True, tooltip="The SDR images or video frames to expand."),
                io.Video.Input("video", optional=True, tooltip="Optional: The SDR video to expand."),
                io.Float.Input("peak_nits", default=400.0, min=100.0, max=10000.0, step=1.0, tooltip="Peak brightness in nits. SDR white (100 nits) will be mapped to this target luminance in HDR."),
                io.Float.Input("itm_knee", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Inverse Tone Mapping (Soft-Knee) threshold. 0.0 starts expansion from black. 0.8 preserves SDR midtones and applies expansion to highlights."),
                io.Float.Input("itm_exponent", default=1.0, min=1.0, max=10.0, step=0.01, tooltip="Expansion curve exponent. 1.0 = Linear (punchy/bright), 2.0 = Quadratic (soft/natural), >2.0 = even softer transition."),
                io.Combo.Input(
                    "color_space",
                    options=COLOR_SPACE_OPTIONS,
                    default="HDR PQ",
                    tooltip="Target HDR color space. 'HDR PQ' (HDR10) is recommended for PC, Mobile, Web, and YouTube. 'HDR' (HLG) is designed for 4K TV broadcast.",
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
        images: Input.Image | None = None,
        video: Input.Video | None = None,
        peak_nits: float = 400.0,
        itm_knee: float = 0.0,
        itm_exponent: float = 1.0,
        color_space: str = "HDR PQ",
    ) -> io.NodeOutput:
        src = unpack_sdr_inputs(images, video, "XENodes.SDRtoHDR")
        internal_algo, comfy_color_space = _resolve_color_space(color_space)

        hdr_images = sdr_to_hdr_tensor(
            src.images,
            peak_nits=peak_nits,
            itm_knee=itm_knee,
            itm_exponent=itm_exponent,
            color_space=internal_algo,
        )

        hdr_images, hdr_video = pack_hdr_outputs(hdr_images, src, comfy_color_space)
        return io.NodeOutput(hdr_images, hdr_video)


class SDRtoHDRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SDRtoHDR]


async def comfy_entrypoint() -> SDRtoHDRExtension:
    return SDRtoHDRExtension()
