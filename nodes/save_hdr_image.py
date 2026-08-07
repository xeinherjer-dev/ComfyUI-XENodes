from __future__ import annotations

import os
import subprocess
import json
import tempfile
import math
from typing_extensions import override

import torch
import numpy as np

from comfy_api.latest import ComfyExtension, io, Input, ui
from comfy_api.latest._ui import SavedImages, SavedResult
import folder_paths

from ..utils.color import apply_inverse_tone_mapping
from ..utils.metadata import get_saved_metadata


def _parse_hdr_image_options(codec_input: dict | str, crf_input: float, kwargs: dict) -> tuple[str, float]:
    sources = []
    if isinstance(codec_input, dict):
        sources.append(codec_input)
    elif isinstance(codec_input, str):
        sources.append({"codec": codec_input})
    if kwargs:
        sources.append(kwargs)

    codec_str = "av1"
    crf = crf_input

    for src in sources:
        if "codec" in src:
            c_val = src["codec"]
            if isinstance(c_val, dict):
                if "codec" in c_val and isinstance(c_val["codec"], str):
                    codec_str = c_val["codec"]
                if "crf" in c_val and c_val["crf"] is not None:
                    try:
                        crf = float(c_val["crf"])
                    except (ValueError, TypeError):
                        pass
            elif isinstance(c_val, str):
                codec_str = c_val

        if "crf" in src and src["crf"] is not None:
            try:
                crf = float(src["crf"])
            except (ValueError, TypeError):
                pass

    return codec_str, crf


class SaveHDRImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveHDRImage",
            display_name="Save HDR Image",
            category="xenodes/experimental",
            is_experimental=True,
            description="Saves the input image natively as HDR AVIF using ffmpeg.",
            inputs=[
                io.Image.Input("images", tooltip="The images to save."),
                io.String.Input("filename_prefix", default="image/ComfyUI", tooltip="The prefix for the file to save."),
                io.Combo.Input("format", options=["avif"], default="avif", tooltip="The image format to save."),
                io.DynamicCombo.Input(
                    "codec",
                    options=[
                        io.DynamicCombo.Option("av1", [io.Float.Input("crf", default=2.0, min=0.0, max=63.0, step=1.0, optional=True, tooltip="CRF for AVIF AV1 (lower = higher quality, default 2).")]),
                        io.DynamicCombo.Option("av1_nvenc", [io.Float.Input("crf", default=2.0, min=0.0, max=63.0, step=1.0, optional=True, tooltip="CQ for AVIF NVENC AV1.")]),
                    ],
                    tooltip="The codec to use for AVIF encoding.",
                ),
                io.Float.Input("peak_nits", default=400.0, min=100.0, max=10000.0, step=1.0, tooltip="Peak brightness in nits. SDR white (100 nits) will be mapped to this target luminance in HDR."),
                io.Float.Input("itm_knee", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Inverse Tone Mapping (Soft-Knee) threshold. 0.0 starts expansion from black. 0.8 preserves SDR midtones and applies expansion to highlights."),
                io.Float.Input("itm_exponent", default=1.0, min=1.0, max=10.0, step=0.01, tooltip="Expansion curve exponent. 1.0 = Linear (punchy/bright), 2.0 = Quadratic (soft/natural), >2.0 = even softer transition."),
            ],
            outputs=[io.Image.Output("gainmap")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: Input.Image, filename_prefix: str, format: str, codec: dict | str = "av1", crf: float = 2.0, peak_nits: float = 400.0, itm_knee: float = 0.0, itm_exponent: float = 1.0, **kwargs) -> io.NodeOutput:
        codec_str, crf = _parse_hdr_image_options(codec, crf, kwargs)
        codec = codec_str

        from ..utils.text import apply_text_replacements
        filename_prefix = apply_text_replacements(filename_prefix, cls.hidden.prompt, cls.hidden.extra_pnginfo)

        width, height = images[0].shape[1], images[0].shape[0]
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )

        saved_metadata = get_saved_metadata(cls)

        results = []
        gainmaps = []
        for i in range(images.shape[0]):
            frame_tensor = images[i]
            current_file_name = f"{filename}_{counter + i:05}_.{format}"
            current_file_path = os.path.join(full_output_folder, current_file_name)

            cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-s", f"{width}x{height}", "-r", "25", "-i", "-"]
            codec_config = {
                "av1": {"codec": "libsvtav1", "options": {"preset": "6"}},
                "av1_nvenc": {"codec": "av1_nvenc", "options": {"preset": "p7"}}
            }
            config = codec_config.get(codec, codec_config["av1"])
            av_codec = config["codec"]
            cmd += ["-c:v", av_codec]

            for key, value in config.get("options", {}).items():
                cmd += [f"-{key}", str(value)]
            cmd += ["-pix_fmt", "yuv420p10le"]
            cmd += ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"]
            
            cmd += ["-vf", "setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709,zscale=p=bt2020:t=smpte2084:m=bt2020nc:npl=100:dither=error_diffusion"]

            if "av1" in av_codec or "svtav1" in av_codec:
                if "nvenc" in av_codec and crf > 0:
                    cmd += ["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"]
                elif crf > 0:
                    cmd += ["-crf", str(int(crf))]

            cmd += ["-frames:v", "1", current_file_path]

            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            try:
                linear_hdr, ratio, scale = apply_inverse_tone_mapping(frame_tensor, peak_nits, itm_knee, itm_exponent)

                if scale > 1.0:
                    gainmap_luma = torch.log2(ratio) / math.log2(scale)
                else:
                    gainmap_luma = torch.zeros_like(ratio)
                
                gainmap_luma = torch.clamp(gainmap_luma, 0.0, 1.0)
                
                gainmap_rgb = gainmap_luma.repeat(1, 1, 3)
                gainmaps.append(gainmap_rgb.clone())

                gbr_planar = linear_hdr[..., [1, 2, 0]].permute(2, 0, 1).contiguous()
                img_bytes = gbr_planar.cpu().numpy().astype(np.float32).tobytes()
                
                proc.stdin.write(img_bytes)
                proc.stdin.close()
                return_code = proc.wait()
                if return_code != 0:
                    stderr_output = proc.stderr.read().decode('utf-8')
                    print(f"[XENodes] FFmpeg failed with return code {return_code}")
                    if stderr_output:
                        print(f"[XENodes] FFmpeg error output:\n{stderr_output}")
            except Exception as e:
                print(f"[XENodes] Error encoding HDR AVIF for frame {i}: {e}")
            finally:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
                proc.wait()

            if saved_metadata:
                try:
                    exif_cmd = ["exiftool", "-overwrite_original"]
                    temp_files = []
                    
                    try:
                        if "workflow" in saved_metadata:
                            workflow_json = json.dumps(saved_metadata["workflow"])
                            fd, path = tempfile.mkstemp(suffix=".txt")
                            temp_files.append(path)
                            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                                f.write(f"workflow:{workflow_json}")
                            exif_cmd.append(f"-Make<={path}")
                            
                        if "prompt" in saved_metadata:
                            prompt_json = json.dumps(saved_metadata["prompt"])
                            fd, path = tempfile.mkstemp(suffix=".txt")
                            temp_files.append(path)
                            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                                f.write(f"prompt:{prompt_json}")
                            exif_cmd.append(f"-Model<={path}")
                            
                        exif_cmd.append(current_file_path)
                        subprocess.run(exif_cmd, check=True, capture_output=True)
                    finally:
                        for f_path in temp_files:
                            try:
                                if os.path.exists(f_path):
                                    os.remove(f_path)
                            except:
                                pass
                except Exception as e:
                    print(f"[XENodes] Warning: Failed to write metadata with exiftool for frame {i}: {e}")

            results.append(SavedResult(current_file_name, subfolder, io.FolderType.output))

        output_gainmaps = torch.stack(gainmaps, dim=0)
        return io.NodeOutput(output_gainmaps, ui=SavedImages(results))

class SaveHDRImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveHDRImage]

async def comfy_entrypoint() -> SaveHDRImageExtension:
    return SaveHDRImageExtension()
