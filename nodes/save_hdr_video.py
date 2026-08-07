from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import json
from fractions import Fraction
from typing_extensions import override

import torch
import numpy as np

from comfy_api.latest import ComfyExtension, io, Input, ui
import folder_paths

from ..utils.audio import expand_audio_waveform
from ..utils.video import generate_frame_indices
from ..utils.color import apply_inverse_tone_mapping
from ..utils.metadata import get_saved_metadata

_MAX_RAW_FRAME_CHUNK_BYTES = 64 * 1024 * 1024

def find_ffmpeg() -> str | None:
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None

def _create_ffmetadata_file(saved_metadata: dict | None) -> str | None:
    if not saved_metadata:
        return None
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".ffmeta", delete=False, encoding="utf-8")
    try:
        handle.write(";FFMETADATA1\n")
        for key, value in saved_metadata.items():
            if isinstance(value, str):
                val_str = value
            else:
                val_str = json.dumps(value)
            escaped = (
                val_str.replace("\\", "\\\\")
                .replace(";", "\\;")
                .replace("#", "\\#")
                .replace("=", "\\=")
                .replace("\n", "\\\n")
            )
            handle.write(f"{key}={escaped}\n")
    finally:
        handle.close()
    return handle.name

def _prepare_audio_file(waveform: torch.Tensor | None, sample_rate: int) -> tuple[str | None, int, int]:
    if waveform is None:
        return None, sample_rate, 2
    
    channels = waveform.shape[0]
    interleaved = waveform.t().contiguous().clamp(-1.0, 1.0).cpu().numpy().astype("float32")
    
    handle = tempfile.NamedTemporaryFile(suffix=".f32le", delete=False)
    try:
        handle.write(interleaved.tobytes())
    finally:
        handle.close()
    return handle.name, sample_rate, channels

def _iter_hdr_frame_byte_chunks(
    images: torch.Tensor,
    frame_indices: list[int],
    total_plays: int,
    peak_nits: float,
    itm_knee: float,
    itm_exponent: float,
    max_chunk_bytes: int = _MAX_RAW_FRAME_CHUNK_BYTES,
):
    bytes_per_frame = images.shape[1] * images.shape[2] * 3 * 4  # float32
    frames_per_chunk = max(1, min(16, max_chunk_bytes // bytes_per_frame))
    
    full_indices = []
    for _ in range(total_plays):
        full_indices.extend(frame_indices)
        
    for start in range(0, len(full_indices), frames_per_chunk):
        chunk_indices = full_indices[start : start + frames_per_chunk]
        chunk_tensors = images[chunk_indices]  # (K, H, W, 3)
        
        linear_hdr, _, _ = apply_inverse_tone_mapping(chunk_tensors, peak_nits, itm_knee, itm_exponent)
        gbr_planar = linear_hdr[..., [1, 2, 0]].permute(0, 3, 1, 2).contiguous()
        chunk_bytes = gbr_planar.cpu().numpy().astype(np.float32).tobytes()
        yield chunk_bytes

def _parse_options(format_input: dict | str, kwargs: dict) -> tuple[str, str, float | None, str, str | None]:
    sources = []
    if isinstance(format_input, dict):
        sources.append(format_input)
    elif isinstance(format_input, str):
        sources.append({"format": format_input})
    if kwargs:
        sources.append(kwargs)

    format_str = "mp4"
    codec_str = "av1"
    crf = None
    audio_codec_str = "aac"
    audio_bitrate = None

    for src in sources:
        if "format" in src and isinstance(src["format"], str):
            format_str = src["format"]
        
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

        if "audio_codec" in src:
            ac_val = src["audio_codec"]
            if isinstance(ac_val, dict):
                if "audio_codec" in ac_val and isinstance(ac_val["audio_codec"], str):
                    audio_codec_str = ac_val["audio_codec"]
                if "audio_bitrate" in ac_val and ac_val["audio_bitrate"] is not None:
                    audio_bitrate = str(ac_val["audio_bitrate"])
            elif isinstance(ac_val, str):
                audio_codec_str = ac_val

        if "audio_bitrate" in src and src["audio_bitrate"] is not None:
            audio_bitrate = str(src["audio_bitrate"])

    return format_str, codec_str, crf, audio_codec_str, audio_bitrate

class SaveHDRVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveHDRVideo",
            display_name="Save HDR Video",
            category="xenodes/experimental",
            is_experimental=True,
            description="Saves the input video natively as HDR using ffmpeg, without AI processing models.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.DynamicCombo.Input(
                    "format",
                    options=[
                        io.DynamicCombo.Option(
                            "mp4",
                            [
                                io.DynamicCombo.Input(
                                    "codec",
                                    options=[
                                        io.DynamicCombo.Option("av1", [io.Float.Input("crf", default=30.0, min=0.0, max=63.0, step=1.0, optional=True, tooltip="CRF for HDR AV1 (lower = higher quality, default 30).")]),
                                        io.DynamicCombo.Option("av1_nvenc", [io.Float.Input("crf", default=30.0, min=0.0, max=51.0, step=1.0, optional=True, tooltip="CQ for HDR NVENC AV1 (0-51, default 30).")]),
                                    ],
                                    tooltip="The video codec.",
                                ),
                                io.DynamicCombo.Input(
                                    "audio_codec",
                                    options=[
                                        io.DynamicCombo.Option(
                                            "aac",
                                            [io.Combo.Input("audio_bitrate", options=["64k", "128k", "192k", "256k", "320k"], default="128k", optional=True, tooltip="Bitrate for AAC audio.")],
                                        ),
                                        io.DynamicCombo.Option(
                                            "opus",
                                            [io.Combo.Input("audio_bitrate", options=["64k", "128k", "192k", "256k", "320k"], default="128k", optional=True, tooltip="Bitrate for Opus audio.")],
                                        ),
                                        io.DynamicCombo.Option("flac", []),
                                    ],
                                    tooltip="The audio codec.",
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "webm",
                            [
                                io.DynamicCombo.Input(
                                    "codec",
                                    options=[
                                        io.DynamicCombo.Option("av1", [io.Float.Input("crf", default=30.0, min=0.0, max=63.0, step=1.0, optional=True, tooltip="CRF for HDR AV1 (lower = higher quality, default 30).")]),
                                        io.DynamicCombo.Option("av1_nvenc", [io.Float.Input("crf", default=30.0, min=0.0, max=51.0, step=1.0, optional=True, tooltip="CQ for HDR NVENC AV1 (0-51, default 30).")]),
                                    ],
                                    tooltip="The video codec.",
                                ),
                                io.DynamicCombo.Input(
                                    "audio_codec",
                                    options=[
                                        io.DynamicCombo.Option(
                                            "opus",
                                            [io.Combo.Input("audio_bitrate", options=["64k", "128k", "192k", "256k", "320k"], default="128k", optional=True, tooltip="Bitrate for Opus audio.")],
                                        ),
                                        io.DynamicCombo.Option("flac", []),
                                    ],
                                    tooltip="The audio codec.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="The format to save the video as.",
                ),
                io.Float.Input("peak_nits", default=400.0, min=100.0, max=10000.0, step=1.0, tooltip="Peak brightness in nits. SDR white (100 nits) will be mapped to this target luminance in HDR."),
                io.Float.Input("itm_knee", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Inverse Tone Mapping (Soft-Knee) threshold. 0.0 starts expansion from black. 0.8 preserves SDR midtones and applies expansion to highlights."),
                io.Float.Input("itm_exponent", default=1.0, min=1.0, max=10.0, step=0.01, tooltip="Expansion curve exponent. 1.0 = Linear (punchy/bright), 2.0 = Quadratic (soft/natural), >2.0 = even softer transition."),
                io.Int.Input("loop_count", default=0, min=0, max=100, step=1, tooltip="Loop count. 0 = play once. For mp4/webm, this physically repeats the frames."),
                io.Boolean.Input("pingpong", default=False, tooltip="Pingpong animation (images only). Plays frames forward then backward."),
            ],
            outputs=[
                io.Video.Output("video", tooltip="The input video."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        filename_prefix: str,
        format: dict | str = "mp4",
        peak_nits: float = 400.0,
        itm_knee: float = 0.0,
        itm_exponent: float = 1.0,
        loop_count: int = 0,
        pingpong: bool = False,
        **kwargs,
    ) -> io.NodeOutput:
        format_str, codec_str, crf, audio_codec_str, audio_bitrate = _parse_options(format, kwargs)

        format = format_str
        codec = codec_str
        audio_codec = audio_codec_str

        from ..utils.text import apply_text_replacements
        filename_prefix = apply_text_replacements(filename_prefix, cls.hidden.prompt, cls.hidden.extra_pnginfo)

        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )

        saved_metadata = get_saved_metadata(cls)

        file_name = f"{filename}_{counter:05}_.{format}"
        file_path = os.path.join(full_output_folder, file_name)

        components = video.get_components()
        frame_rate = Fraction(round(components.frame_rate * 1000), 1000)
        fps = float(frame_rate)

        images = components.images  # shape: (N, H, W, 3)
        num_images = images.shape[0]

        frame_indices = generate_frame_indices(num_images, pingpong)
        n_orig = len(frame_indices)
        total_plays = loop_count + 1

        waveform, audio_sample_rate, layout = expand_audio_waveform(components, fps, n_orig, total_plays)
        output_sample_rate = audio_sample_rate
        if waveform is not None:
            if audio_codec == "opus":
                output_sample_rate = 48000
        else:
            output_sample_rate = 44100

        ffmpeg_exe = find_ffmpeg()
        if not ffmpeg_exe:
            raise RuntimeError("FFmpeg executable not found. Please install ffmpeg or imageio-ffmpeg.")

        if crf is None:
            crf = 30.0

        codec_config = {
            "av1": {"codec": "libsvtav1", "options": ["-preset", "6"]},
            "av1_nvenc": {"codec": "av1_nvenc", "options": ["-preset", "p7"]}
        }
        config = codec_config.get(codec, codec_config["av1"])
        av_codec = config["codec"]
        base_options = config["options"]

        metadata_file = _create_ffmetadata_file(saved_metadata)
        audio_file, audio_sr, audio_ch = _prepare_audio_file(waveform, output_sample_rate)

        cmd = [ffmpeg_exe, "-y", "-v", "error"]

        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", "gbrpf32le",
            "-s", f"{width}x{height}",
            "-r", f"{fps:.06f}",
            "-i", "-",
        ])

        input_idx = 1
        video_map = "0:v:0"
        audio_map = None
        meta_map = None

        if audio_file is not None:
            cmd.extend([
                "-f", "f32le",
                "-ar", str(audio_sr),
                "-ac", str(audio_ch),
                "-i", audio_file,
            ])
            audio_map = f"{input_idx}:a:0"
            input_idx += 1

        if metadata_file is not None:
            cmd.extend([
                "-f", "ffmetadata",
                "-i", metadata_file,
            ])
            meta_map = f"{input_idx}"
            input_idx += 1

        cmd.extend(["-map", video_map])
        if audio_map:
            cmd.extend(["-map", audio_map])
        if meta_map:
            cmd.extend(["-map_metadata", meta_map])

        cmd.extend(["-c:v", av_codec, "-pix_fmt", "yuv420p10le"])
        cmd.extend(base_options)

        if "nvenc" in av_codec:
            cmd.extend(["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"])
        else:
            cmd.extend(["-crf", str(int(crf))])

        if format == "mp4":
            cmd.extend(["-movflags", "+use_metadata_tags+faststart"])

        trc = "smpte2084"
        cmd.extend(["-color_primaries", "bt2020", "-color_trc", trc, "-colorspace", "bt2020nc"])

        zscale_trc = "smpte2084"
        zscale_params = f"p=bt2020:t={zscale_trc}:m=bt2020nc:npl=100:dither=error_diffusion"
        cmd.extend(["-vf", f"setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709,zscale={zscale_params}"])

        if audio_map:
            audio_codec_map = {
                "aac": "aac",
                "opus": "libopus",
                "flac": "flac"
            }
            av_audio_codec = audio_codec_map.get(audio_codec, "aac")
            cmd.extend(["-c:a", av_audio_codec])
            if av_audio_codec == "libopus":
                cmd.extend(["-ar", "48000"])
            if audio_bitrate is not None and av_audio_codec != "flac":
                cmd.extend(["-b:a", audio_bitrate])

        cmd.append(file_path)

        print(f"[XENodes] SaveHDRVideo (FFmpeg): running command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            for chunk in _iter_hdr_frame_byte_chunks(
                images, frame_indices, total_plays, peak_nits, itm_knee, itm_exponent
            ):
                process.stdin.write(chunk)
            process.stdin.close()
            retcode = process.wait()
        except Exception as e:
            process.kill()
            process.wait()
            raise e
        finally:
            if metadata_file and os.path.exists(metadata_file):
                os.unlink(metadata_file)
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)

        if retcode != 0:
            stderr_out = process.stderr.read().decode(errors="replace")
            raise RuntimeError(f"FFmpeg HDR encoding failed (exit code {retcode}): {stderr_out}")

        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)]))

class SaveHDRVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveHDRVideo]

async def comfy_entrypoint() -> SaveHDRVideoExtension:
    return SaveHDRVideoExtension()
