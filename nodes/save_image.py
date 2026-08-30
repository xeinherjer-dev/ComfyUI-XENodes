from __future__ import annotations

import os
import json
import shutil
import subprocess
import tempfile
import numpy as np
import torch
from PIL import Image as PILImage
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, Input, ui
from comfy_api.latest._ui import ImageSaveHelper, SavedImages, SavedResult, FolderType
from comfy.cli_args import args
import folder_paths
from ..utils.metadata import get_saved_metadata


def find_ffmpeg() -> str | None:
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _parse_image_options(format_input: dict | str, kwargs: dict) -> tuple[str, bool, int, int, float, str]:
    sources = []
    if isinstance(format_input, dict):
        sources.append(format_input)
    elif isinstance(format_input, str):
        sources.append({"format": format_input})
    if kwargs:
        sources.append(kwargs)

    format_str = "png"
    lossless = False
    quality = 90
    compression = 6
    crf = 2.0
    color_space = "auto"

    for src in sources:
        if "format" in src and isinstance(src["format"], str):
            format_str = src["format"]
        
        if "lossless" in src and src["lossless"] is not None:
            lossless = bool(src["lossless"])
            
        if "quality" in src and src["quality"] is not None:
            try:
                quality = int(src["quality"])
            except (ValueError, TypeError):
                pass
                
        if "compression" in src and src["compression"] is not None:
            try:
                compression = int(src["compression"])
            except (ValueError, TypeError):
                pass

        if "crf" in src and src["crf"] is not None:
            try:
                crf = float(src["crf"])
            except (ValueError, TypeError):
                pass

        if "color_space" in src and isinstance(src["color_space"], str):
            color_space = src["color_space"]

    return format_str, lossless, quality, compression, crf, color_space


def _save_avif_image(
    image_tensor: torch.Tensor,
    file_path: str,
    crf: float,
    color_space: str,
    saved_metadata: dict | None,
):
    ffmpeg_exe = find_ffmpeg()
    if not ffmpeg_exe:
        raise RuntimeError("FFmpeg executable not found. Please install ffmpeg or imageio-ffmpeg.")

    height, width = image_tensor.shape[0], image_tensor.shape[1]
    is_hdr = color_space in ("auto", "HDR PQ (HDR10)", "HDR PQ", "HDR HLG", "HDR")
    
    input_pix_fmt = "gbrpf32le" if is_hdr else "rgb24"
    out_pix_fmt = "yuv420p10le" if is_hdr else "yuv420p"

    cmd = [
        ffmpeg_exe, "-y", "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", input_pix_fmt,
        "-s", f"{width}x{height}",
        "-r", "25",
        "-i", "-",
        "-c:v", "libsvtav1",
        "-preset", "6",
        "-svtav1-params", "lookahead=0:tune=0",
        "-pix_fmt", out_pix_fmt,
        "-crf", str(int(crf)),
    ]

    if is_hdr:
        if "HLG" in color_space.upper() or color_space == "HDR":
            cmd.extend(["-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc"])
        else:
            cmd.extend(["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"])

    cmd.extend(["-frames:v", "1", file_path])

    stderr_tmp = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_tmp)

    try:
        rgb_tensor = image_tensor[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
        if is_hdr:
            gbr_planar = rgb_tensor[..., [1, 2, 0]].permute(2, 0, 1).contiguous()
            img_bytes = gbr_planar.numpy().astype(np.float32).tobytes()
        else:
            img_bytes = (rgb_tensor * 255.0).round().to(torch.uint8).contiguous().numpy().tobytes()

        proc.stdin.write(img_bytes)
        proc.stdin.close()
        return_code = proc.wait()
        if return_code != 0:
            stderr_tmp.seek(0)
            stderr_output = stderr_tmp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg AVIF encode failed with code {return_code}: {stderr_output}")
    except Exception as e:
        proc.kill()
        proc.wait()
        raise e

    if saved_metadata and shutil.which("exiftool"):
        try:
            exif_cmd = ["exiftool", "-overwrite_original"]
            temp_files = []
            try:
                if "workflow" in saved_metadata:
                    workflow_json = json.dumps(saved_metadata["workflow"])
                    fd, path = tempfile.mkstemp(suffix=".txt")
                    temp_files.append(path)
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(f"workflow:{workflow_json}")
                    exif_cmd.append(f"-Make<={path}")
                    
                if "prompt" in saved_metadata:
                    prompt_json = json.dumps(saved_metadata["prompt"])
                    fd, path = tempfile.mkstemp(suffix=".txt")
                    temp_files.append(path)
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(f"prompt:{prompt_json}")
                    exif_cmd.append(f"-Model<={path}")
                    
                exif_cmd.append(file_path)
                subprocess.run(exif_cmd, check=True, capture_output=True, timeout=10)
            finally:
                for f_path in temp_files:
                    if os.path.exists(f_path):
                        try:
                            os.remove(f_path)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[XENodes] Warning: Failed to write AVIF metadata with exiftool: {e}")


class SaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveImage",
            display_name="Save Image",
            category="xenodes/image",
            description="Saves the input images as PNG, WebP, or 10-bit HDR AVIF.",
            inputs=[
                io.Image.Input("images", tooltip="The images to save (supports SDR and 10-bit HDR images)."),
                io.String.Input(
                    "filename_prefix",
                    default="image/ComfyUI",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                ),
                io.DynamicCombo.Input(
                    "format",
                    options=[
                        io.DynamicCombo.Option(
                            "png",
                            [
                                io.Int.Input("compression", default=6, min=0, max=9, optional=True, tooltip="PNG compression level (0-9)."),
                            ]
                        ),
                        io.DynamicCombo.Option(
                            "webp",
                            [
                                io.Boolean.Input("lossless", default=False, optional=True, tooltip="Enables WebP lossless encoding."),
                                io.Int.Input("quality", default=90, min=0, max=100, optional=True, tooltip="WebP quality (0-100)."),
                            ]
                        ),
                        io.DynamicCombo.Option(
                            "avif",
                            [
                                io.Float.Input("crf", default=2.0, min=0.0, max=63.0, step=1.0, optional=True, tooltip="CRF for AVIF AV1 (lower = higher quality, default 2)."),
                                io.Combo.Input(
                                    "color_space",
                                    options=["auto", "HDR(PQ)", "HDR(HLG)", "sRGB"],
                                    default="auto",
                                    optional=True,
                                    tooltip="Color space / EOTF metadata for AVIF. 'auto' selects HDR(PQ).",
                                ),
                            ]
                        ),
                    ],
                    tooltip="The image format to save as.",
                ),
            ],
            outputs=[io.Image.Output("images")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: Input.Image, filename_prefix: str, format: dict | str = "png", **kwargs) -> io.NodeOutput:
        format_str, lossless, quality, compression, crf, color_space = _parse_image_options(format, kwargs)

        from ..utils.text import apply_text_replacements
        filename_prefix = apply_text_replacements(filename_prefix, cls.hidden.prompt, cls.hidden.extra_pnginfo)

        if images is None or len(images) == 0:
            return io.NodeOutput(images, ui=SavedImages([]))

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            images[0].shape[1],
            images[0].shape[0],
        )

        saved_metadata = get_saved_metadata(cls)
        results = []

        for batch_number, image_tensor in enumerate(images):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

            if format_str == "avif":
                file = f"{filename_with_batch_num}_{counter:05}_.avif"
                file_path = os.path.join(full_output_folder, file)
                _save_avif_image(
                    image_tensor=image_tensor,
                    file_path=file_path,
                    crf=crf,
                    color_space=color_space,
                    saved_metadata=saved_metadata,
                )
            elif format_str == "webp":
                img = PILImage.fromarray(
                    np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
                )
                file = f"{filename_with_batch_num}_{counter:05}_.webp"
                file_path = os.path.join(full_output_folder, file)

                exif_data = ImageSaveHelper._create_webp_metadata(img, cls)
                img.save(
                    file_path,
                    format="webp",
                    lossless=lossless,
                    quality=quality,
                    exif=exif_data,
                )
            else:  # png
                img = PILImage.fromarray(
                    np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
                )
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                file_path = os.path.join(full_output_folder, file)

                compress_level = max(0, min(9, compression))

                metadata = ImageSaveHelper._create_png_metadata(cls)
                img.save(
                    file_path,
                    pnginfo=metadata,
                    compress_level=compress_level,
                )

            results.append(SavedResult(file, subfolder, FolderType.output))
            counter += 1

        return io.NodeOutput(images, ui=SavedImages(results))


class SaveImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveImage]


async def comfy_entrypoint() -> SaveImageExtension:
    return SaveImageExtension()
