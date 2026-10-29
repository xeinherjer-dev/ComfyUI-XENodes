from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import functools
import json
import os
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


@functools.lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


@functools.lru_cache(maxsize=1)
def _find_exiftool() -> str | None:
    return shutil.which("exiftool")


def _parse_image_options(format_input: dict | str, kwargs: dict) -> tuple[str, bool, int, int, float]:
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

    return format_str, lossless, quality, compression, crf


def _resolve_avif_color_config(resolved_cs: str) -> tuple[int, int, int, str, str, str, str, str, str, bool]:
    is_hlg = resolved_cs == "HDR" or "HLG" in str(resolved_cs).upper()
    is_srgb = resolved_cs == "sRGB" or "SRGB" in str(resolved_cs).upper()

    if is_srgb:
        cp_num, tc_num, mc_num = 1, 13, 1
        primaries, trc, csp, matrix = "bt709", "iec61966-2-1", "bt709", "bt709"
        out_pix_fmt = "yuv420p"
        hdr_flag = "0"
        is_bt2020 = False
    elif is_hlg:
        cp_num, tc_num, mc_num = 9, 18, 9
        primaries, trc, csp, matrix = "bt2020", "arib-std-b67", "bt2020nc", "bt2020"
        out_pix_fmt = "yuv420p10le"
        hdr_flag = "1"
        is_bt2020 = True
    else:  # HDR PQ
        cp_num, tc_num, mc_num = 9, 16, 9
        primaries, trc, csp, matrix = "bt2020", "smpte2084", "bt2020nc", "bt2020"
        out_pix_fmt = "yuv420p10le"
        hdr_flag = "1"
        is_bt2020 = True

    return cp_num, tc_num, mc_num, primaries, trc, csp, matrix, out_pix_fmt, hdr_flag, is_bt2020


def _rgb_batch_to_full_range_yuv420(
    images: torch.Tensor,
    is_10bit: bool,
    is_bt2020: bool,
) -> np.ndarray:
    """
    Directly converts RGB float32 [0, 1] batch (B, H, W, C) to Full-Range (PC range)
    ITU-R BT.709 or BT.2020 YUV 4:2:0 (yuv420p or yuv420p10le) on GPU/CPU.
    """
    compute_device = torch.device("cuda") if torch.cuda.is_available() else images.device
    if is_bt2020:
        kr, kg, kb = 0.2627, 0.6780, 0.0593
    else:
        kr, kg, kb = 0.2126, 0.7152, 0.0722

    inv_cb_scale = 1.0 / (2.0 * (1.0 - kb))
    inv_cr_scale = 1.0 / (2.0 * (1.0 - kr))
    scale_max = 1023.0 if is_10bit else 255.0
    out_dtype = torch.uint16 if is_10bit else torch.uint8

    B, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    with torch.no_grad():
        x = images[..., :3].detach().to(device=compute_device, dtype=torch.float32, non_blocking=True).clamp_(0.0, 1.0)
        y_f = kr * x[..., 0] + kg * x[..., 1] + kb * x[..., 2]

        x_sub = x.view(B, H // 2, 2, W // 2, 2, 3).mean(dim=(2, 4))
        y_s = kr * x_sub[..., 0] + kg * x_sub[..., 1] + kb * x_sub[..., 2]
        cb_f = (x_sub[..., 2] - y_s) * inv_cb_scale + 0.5
        cr_f = (x_sub[..., 0] - y_s) * inv_cr_scale + 0.5

        y_q = (y_f * scale_max).round_().clamp_(0.0, scale_max).to(out_dtype).reshape(B, -1)
        cb_q = (cb_f * scale_max).round_().clamp_(0.0, scale_max).to(out_dtype).reshape(B, -1)
        cr_q = (cr_f * scale_max).round_().clamp_(0.0, scale_max).to(out_dtype).reshape(B, -1)
        packed = torch.cat((y_q, cb_q, cr_q), dim=1)
        return packed.cpu().numpy()


def _write_avif_metadata_batch(file_paths: list[str], saved_metadata: dict | None) -> None:
    if not file_paths or not saved_metadata:
        return
    exiftool_exe = _find_exiftool()
    if not exiftool_exe:
        return

    try:
        exif_cmd = [exiftool_exe, "-overwrite_original"]
        temp_files: list[str] = []
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

            exif_cmd.extend(file_paths)
            subprocess.run(exif_cmd, check=True, capture_output=True, timeout=max(10, 5 * len(file_paths)))
        finally:
            for f_path in temp_files:
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except Exception:
                        pass
    except Exception as e:
        print(f"[XENodes] Warning: Failed to write AVIF metadata with exiftool: {e}")


def _encode_avif_bytes_with_ffmpeg(
    raw_bytes: bytes | memoryview,
    file_path: str,
    width: int,
    height: int,
    in_pix_fmt: str,
    out_pix_fmt: str,
    primaries: str,
    trc: str,
    csp: str,
    matrix: str,
    svtav1_params: str,
    crf: float,
    needs_scale_filter: bool,
) -> None:
    ffmpeg_exe = find_ffmpeg()
    if not ffmpeg_exe:
        raise RuntimeError("FFmpeg executable not found. Please install ffmpeg or imageio-ffmpeg.")

    cmd = [
        ffmpeg_exe, "-y", "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", in_pix_fmt,
        "-s", f"{width}x{height}",
        "-r", "25",
        "-color_primaries", primaries,
        "-color_trc", trc,
        "-colorspace", csp,
        "-color_range", "pc",
        "-i", "-",
    ]
    if needs_scale_filter:
        cmd += [
            "-vf",
            f"scale=w=max(64\\,trunc(iw/2)*2):h=max(64\\,trunc(ih/2)*2):"
            f"in_color_matrix={matrix}:out_color_matrix={matrix}:in_range=full:out_range=full:flags=bicubic",
        ]
    cmd += [
        "-c:v", "libsvtav1",
        "-preset", "6",
        "-svtav1-params", svtav1_params,
        "-pix_fmt", out_pix_fmt,
        "-color_primaries", primaries,
        "-color_trc", trc,
        "-colorspace", csp,
        "-color_range", "pc",
        "-crf", str(int(round(crf))),
        "-frames:v", "1",
        file_path,
    ]

    res = subprocess.run(cmd, input=raw_bytes, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        stderr_output = res.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg AVIF encode failed with code {res.returncode}: {stderr_output}")


def _save_avif_image(
    image_tensor: torch.Tensor,
    file_path: str,
    crf: float,
    parent_images: torch.Tensor | None = None,
    saved_metadata: dict | None = None,
    write_metadata: bool = True,
):
    height, width = int(image_tensor.shape[0]), int(image_tensor.shape[1])

    # Auto-detect color space from tensor metadata (set by SDR to HDR node) or fallback to HDR PQ
    resolved_cs = getattr(image_tensor, "_color_space", getattr(parent_images, "_color_space", "HDR PQ"))
    cp_num, tc_num, mc_num, primaries, trc, csp, matrix, out_pix_fmt, hdr_flag, is_bt2020 = _resolve_avif_color_config(resolved_cs)

    # lp=4 caps SVT-AV1 picture control set pool (PPCS 107 -> 41) for 1-frame AVIF while keeping bitstream 100% identical
    svtav1_params = (
        f"lp=4:lookahead=0:tune=0:enable-hdr={hdr_flag}:"
        f"color-primaries={cp_num}:transfer-characteristics={tc_num}:matrix-coefficients={mc_num}:color-range=1"
    )

    can_direct_yuv420 = height >= 64 and width >= 64 and (height % 2 == 0) and (width % 2 == 0)
    if can_direct_yuv420:
        is_10bit = out_pix_fmt == "yuv420p10le"
        yuv_batch = _rgb_batch_to_full_range_yuv420(image_tensor.unsqueeze(0), is_10bit=is_10bit, is_bt2020=is_bt2020)
        _encode_avif_bytes_with_ffmpeg(
            raw_bytes=memoryview(yuv_batch[0]).cast("B"),
            file_path=file_path,
            width=width,
            height=height,
            in_pix_fmt=out_pix_fmt,
            out_pix_fmt=out_pix_fmt,
            primaries=primaries,
            trc=trc,
            csp=csp,
            matrix=matrix,
            svtav1_params=svtav1_params,
            crf=crf,
            needs_scale_filter=False,
        )
    else:
        with torch.no_grad():
            rgb_u16 = (
                image_tensor[..., :3]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .clamp_(0.0, 1.0)
                .mul_(65535.0)
                .round_()
                .to(torch.uint16)
                .contiguous()
            )
        _encode_avif_bytes_with_ffmpeg(
            raw_bytes=memoryview(rgb_u16.numpy()).cast("B"),
            file_path=file_path,
            width=width,
            height=height,
            in_pix_fmt="rgb48le",
            out_pix_fmt=out_pix_fmt,
            primaries=primaries,
            trc=trc,
            csp=csp,
            matrix=matrix,
            svtav1_params=svtav1_params,
            crf=crf,
            needs_scale_filter=True,
        )

    if write_metadata and saved_metadata:
        _write_avif_metadata_batch([file_path], saved_metadata)


def _tensor_batch_to_uint8_numpy(images: torch.Tensor) -> np.ndarray:
    """Vectorized float32 [0, 1] -> uint8 [0, 255] conversion for an entire image batch."""
    with torch.no_grad():
        return (
            images.detach()
            .to(device="cpu", dtype=torch.float32)
            .mul(255.0)
            .clamp_(0.0, 255.0)
            .to(torch.uint8)
            .contiguous()
            .numpy()
        )


class SaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveImage",
            display_name="Save Image",
            category="xenodes/image",
            description="Saves the input images as PNG, WebP, or 10-bit HDR AVIF. AVIF automatically detects HDR color space from SDR to HDR.",
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
        format_str, lossless, quality, compression, crf = _parse_image_options(format, kwargs)

        from ..utils.text import apply_text_replacements
        filename_prefix = apply_text_replacements(filename_prefix, cls.hidden.prompt, cls.hidden.extra_pnginfo)

        if images is None or len(images) == 0:
            return io.NodeOutput(images, ui=SavedImages([]))

        height, width = int(images[0].shape[0]), int(images[0].shape[1])
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )

        batch_size = len(images)
        file_names: list[str] = []
        file_paths: list[str] = []
        for batch_number in range(batch_size):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            ext = "avif" if format_str == "avif" else ("webp" if format_str == "webp" else "png")
            file = f"{filename_with_batch_num}_{counter + batch_number:05}_.{ext}"
            file_names.append(file)
            file_paths.append(os.path.join(full_output_folder, file))

        if format_str == "avif":
            saved_metadata = get_saved_metadata(cls)
            resolved_cs = getattr(images, "_color_space", getattr(images[0], "_color_space", "HDR PQ"))
            cp_num, tc_num, mc_num, primaries, trc, csp, matrix, out_pix_fmt, hdr_flag, is_bt2020 = _resolve_avif_color_config(resolved_cs)
            svtav1_params = (
                f"lp=4:lookahead=0:tune=0:enable-hdr={hdr_flag}:"
                f"color-primaries={cp_num}:transfer-characteristics={tc_num}:matrix-coefficients={mc_num}:color-range=1"
            )
            can_direct_yuv420 = (
                isinstance(images, torch.Tensor)
                and height >= 64
                and width >= 64
                and (height % 2 == 0)
                and (width % 2 == 0)
            )
            if can_direct_yuv420:
                is_10bit = out_pix_fmt == "yuv420p10le"
                yuv_batch = _rgb_batch_to_full_range_yuv420(images, is_10bit=is_10bit, is_bt2020=is_bt2020)

                def _encode_idx(idx: int) -> None:
                    _encode_avif_bytes_with_ffmpeg(
                        raw_bytes=memoryview(yuv_batch[idx]).cast("B"),
                        file_path=file_paths[idx],
                        width=width,
                        height=height,
                        in_pix_fmt=out_pix_fmt,
                        out_pix_fmt=out_pix_fmt,
                        primaries=primaries,
                        trc=trc,
                        csp=csp,
                        matrix=matrix,
                        svtav1_params=svtav1_params,
                        crf=crf,
                        needs_scale_filter=False,
                    )

                if batch_size == 1:
                    _encode_idx(0)
                else:
                    max_workers = min(4, batch_size)
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        list(ex.map(_encode_idx, range(batch_size)))
            else:
                def _save_fallback_idx(idx: int) -> None:
                    _save_avif_image(
                        image_tensor=images[idx],
                        file_path=file_paths[idx],
                        crf=crf,
                        parent_images=images,
                        saved_metadata=None,
                        write_metadata=False,
                    )

                if batch_size == 1:
                    _save_fallback_idx(0)
                else:
                    max_workers = min(4, batch_size)
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        list(ex.map(_save_fallback_idx, range(batch_size)))

            if saved_metadata:
                _write_avif_metadata_batch(file_paths, saved_metadata)

        elif format_str == "webp":
            u8_batch = (
                _tensor_batch_to_uint8_numpy(images)
                if isinstance(images, torch.Tensor)
                else [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in images]
            )
            first_img = PILImage.fromarray(u8_batch[0])
            exif_data = ImageSaveHelper._create_webp_metadata(first_img, cls)

            def _save_webp_idx(idx: int) -> None:
                img = first_img if idx == 0 else PILImage.fromarray(u8_batch[idx])
                img.save(
                    file_paths[idx],
                    format="webp",
                    lossless=lossless,
                    quality=quality,
                    exif=exif_data,
                    thread_level=1,
                )

            if batch_size == 1:
                _save_webp_idx(0)
            else:
                max_workers = min(8, os.cpu_count() or 4, batch_size)
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    list(ex.map(_save_webp_idx, range(batch_size)))

        else:  # png
            u8_batch = (
                _tensor_batch_to_uint8_numpy(images)
                if isinstance(images, torch.Tensor)
                else [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in images]
            )
            compress_level = max(0, min(9, compression))
            metadata = ImageSaveHelper._create_png_metadata(cls)

            def _save_png_idx(idx: int) -> None:
                img = PILImage.fromarray(u8_batch[idx])
                img.save(
                    file_paths[idx],
                    pnginfo=metadata,
                    compress_level=compress_level,
                )

            if batch_size == 1:
                _save_png_idx(0)
            else:
                max_workers = min(8, os.cpu_count() or 4, batch_size)
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    list(ex.map(_save_png_idx, range(batch_size)))

        results = [SavedResult(file, subfolder, FolderType.output) for file in file_names]
        return io.NodeOutput(images, ui=SavedImages(results))


class SaveImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveImage]


async def comfy_entrypoint() -> SaveImageExtension:
    return SaveImageExtension()
