from __future__ import annotations

import os
import subprocess
import json
import tempfile
from typing_extensions import override

import torch
import numpy as np

from comfy_api.latest import ComfyExtension, io, Input, ui
from comfy_api.latest._ui import SavedImages, SavedResult
from comfy.cli_args import args
import folder_paths

class SaveHDRImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveHDRImage",
            display_name="Save HDR Image",
            category="XENodes",
            description="Saves the input image natively as HDR AVIF using ffmpeg.",
            inputs=[
                io.Image.Input("images", tooltip="The images to save."),
                io.String.Input("filename_prefix", default="image/ComfyUI", tooltip="The prefix for the file to save."),
                io.Combo.Input("format", options=["avif"], default="avif", tooltip="The image format to save."),
                io.Combo.Input("codec", options=["av1", "av1_nvenc"], default="av1", tooltip="The codec to use for AVIF encoding."),
                io.Float.Input("crf", default=10.0, min=0.0, max=63.0, step=1.0, tooltip="Specific CRF value used for encoding (maps to CQ for NVENC). Set to 0 to use encoder defaults."),
                io.Float.Input("peak_nits", default=400.0, min=100.0, max=10000.0, step=1.0, tooltip="Peak brightness in nits. SDR white (100 nits) will be mapped to this target luminance in HDR."),
                io.Float.Input("itm_knee", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Inverse Tone Mapping (Soft-Knee) threshold. 0.0 starts expansion from black. 0.8 preserves SDR midtones and applies expansion to highlights."),
                io.Float.Input("itm_exponent", default=1.0, min=1.0, max=10.0, step=0.01, tooltip="Expansion curve exponent. 1.0 = Linear (punchy/bright), 2.0 = Quadratic (soft/natural), >2.0 = even softer transition."),
            ],
            outputs=[io.Image.Output("images")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: Input.Image, filename_prefix: str, format: str, codec: str, crf: float, peak_nits: float, itm_knee: float, itm_exponent: float) -> io.NodeOutput:
        width, height = images[0].shape[1], images[0].shape[0]
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )

        saved_metadata = {}
        if not args.disable_metadata:
            if cls.hidden.extra_pnginfo is not None:
                saved_metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                saved_metadata["prompt"] = cls.hidden.prompt

        results = []
        for i in range(images.shape[0]):
            frame_tensor = images[i]
            current_file_name = f"{filename}_{counter + i:05}_.{format}"
            current_file_path = os.path.join(full_output_folder, current_file_name)

            cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-s", f"{width}x{height}", "-r", "25", "-i", "-"]
            av_codec = "libsvtav1" if codec == "av1" else codec
            cmd += ["-c:v", av_codec]
            cmd += ["-pix_fmt", "yuv420p10le"]
            cmd += ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"]
            
            # Since input is already linear float32 (where 1.0 = 100 nits), we just convert to PQ
            cmd += ["-vf", "setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709,zscale=p=bt2020:t=smpte2084:m=bt2020nc:npl=100"]

            if "av1" in av_codec or "svtav1" in av_codec:
                if "nvenc" in av_codec and crf > 0:
                    cmd += ["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"]
                elif crf > 0:
                    cmd += ["-crf", str(int(crf))]

            cmd += ["-frames:v", "1", current_file_path]

            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            try:
                # Convert sRGB to Linear (IEC 61966-2-1 standard)
                # frame_tensor is expected to be in [0, 1]
                linear = torch.where(frame_tensor <= 0.04045, frame_tensor / 12.92, ((frame_tensor + 0.055) / 1.055) ** 2.4)
                
                # Apply Soft-Knee Inverse Tone Mapping (Power Curve)
                scale = peak_nits / 100.0
                if itm_knee < 1.0:
                    # y = x + a * max(0, x - knee)^exponent
                    # a = (scale - 1.0) / (1.0 - knee)^exponent
                    a = (scale - 1.0) / ((1.0 - itm_knee) ** itm_exponent)
                    diff = torch.clamp(linear - itm_knee, min=0.0)
                    linear = linear + a * (diff ** itm_exponent)

                # Convert to GBR planar format (gbrpf32le)
                gbr_planar = linear[..., [1, 2, 0]].permute(2, 0, 1).contiguous()
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
                    # FFmpeg's default metadata mapping (-map_metadata) is incompatible with ComfyUI's AVIF parser, 
                    # as it doesn't write to the specific Exif boxes the frontend expects.
                    # We use exiftool to write metadata to ASCII tags that ComfyUI's avif.ts can actually parse.
                    # avif.ts looks for tags with type 2 (ASCII) and checks for "workflow:" or "prompt:" prefix.
                    # Note: UserComment is type 7 and the current frontend parser fails to read it.
                    # Use the most robust method for Windows: write values to temp files and use -TAG<=FILE
                    # This avoids command line length limits AND issues with newlines in JSON.
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

        return io.NodeOutput(images, ui=SavedImages(results))

class SaveHDRImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveHDRImage]

async def comfy_entrypoint() -> SaveHDRImageExtension:
    return SaveHDRImageExtension()
