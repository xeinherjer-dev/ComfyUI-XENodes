from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import json
import time
import math
import torch
from fractions import Fraction
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, Input, ui
import folder_paths

from ..utils.audio import expand_audio_waveform
from ..utils.video import generate_frame_indices
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

def _iter_frame_byte_chunks(
    images: torch.Tensor,
    frame_indices: list[int],
    total_plays: int,
    max_chunk_bytes: int = _MAX_RAW_FRAME_CHUNK_BYTES,
):
    bytes_per_frame = images.shape[1] * images.shape[2] * 3
    frames_per_chunk = max(1, min(32, max_chunk_bytes // bytes_per_frame))
    
    full_indices = []
    for _ in range(total_plays):
        full_indices.extend(frame_indices)
        
    for start in range(0, len(full_indices), frames_per_chunk):
        chunk_indices = full_indices[start : start + frames_per_chunk]
        chunk = images[chunk_indices, ..., :3].detach().to(device="cpu", dtype=torch.float32).clamp_(0, 1)
        yield (chunk * 255.0).round_().to(torch.uint8).numpy().tobytes()

class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveVideo",
            display_name="Save Video",
            category="xenodes/video",
            description="Saves the input video natively with AV1/CRF support, independently of core save_to.",
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
                                        io.DynamicCombo.Option("h264", [io.Float.Input("crf", default=23.0, min=0.0, max=51.0, step=1.0, optional=True, tooltip="CRF for H.264 (lower = higher quality).")]),
                                        io.DynamicCombo.Option("h265", [io.Float.Input("crf", default=28.0, min=0.0, max=51.0, step=1.0, optional=True, tooltip="CRF for H.265 (lower = higher quality).")]),
                                        io.DynamicCombo.Option("av1", [io.Float.Input("crf", default=42.0, min=0.0, max=63.0, step=1.0, optional=True, tooltip="CRF for AV1 (lower = higher quality).")]),
                                        io.DynamicCombo.Option("h264_nvenc", [io.Float.Input("crf", default=30.0, min=0.0, max=51.0, step=1.0, optional=True, tooltip="CQ for NVENC H.264.")]),
                                        io.DynamicCombo.Option("hevc_nvenc", [io.Float.Input("crf", default=35.0, min=0.0, max=51.0, step=1.0, optional=True, tooltip="CQ for NVENC HEVC.")]),
                                        io.DynamicCombo.Option("av1_nvenc", [io.Float.Input("crf", default=42.0, min=0.0, max=63.0, step=1.0, optional=True, tooltip="CQ for NVENC AV1.")]),
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
                                        io.DynamicCombo.Option("av1", [io.Float.Input("crf", default=42.0, min=0.0, max=63.0, step=1.0, optional=True, tooltip="CRF for AV1 (lower = higher quality).")]),
                                        io.DynamicCombo.Option("av1_nvenc", [io.Float.Input("crf", default=42.0, min=0.0, max=63.0, step=1.0, optional=True, tooltip="CQ for NVENC AV1.")]),
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
                io.Int.Input("loop_count", default=0, min=0, max=100, step=1, tooltip="Loop count. 0 = play once. For mp4/webm, this physically repeats the frames."),
                io.Boolean.Input("pingpong", default=False, tooltip="Pingpong animation (images only). Plays frames forward then backward."),
            ],
            outputs=[io.Video.Output("video")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        video: Input.Video,
        filename_prefix: str,
        format: dict | str = "mp4",
        loop_count: int = 0,
        pingpong: bool = False,
        **kwargs,
    ) -> io.NodeOutput:
        format_str = "mp4"
        codec_str = "h264"
        audio_codec_str = "aac"
        crf = None
        audio_bitrate = None

        if isinstance(format, dict):
            format_str = format.get("format", "mp4")
            codec_obj = format.get("codec")
            if isinstance(codec_obj, dict):
                codec_str = codec_obj.get("codec", "h264")
                crf = codec_obj.get("crf")
            elif isinstance(codec_obj, str):
                codec_str = codec_obj

            ac_obj = format.get("audio_codec")
            if isinstance(ac_obj, dict):
                audio_codec_str = ac_obj.get("audio_codec", "aac")
                audio_bitrate = ac_obj.get("audio_bitrate")
            elif isinstance(ac_obj, str):
                audio_codec_str = ac_obj
        elif isinstance(format, str):
            format_str = format
            if "codec" in kwargs:
                codec_val = kwargs["codec"]
                if isinstance(codec_val, dict):
                    codec_str = codec_val.get("codec", "h264")
                    crf = codec_val.get("crf")
                elif isinstance(codec_val, str):
                    codec_str = codec_val
            if "crf" in kwargs and crf is None:
                try:
                    crf = float(kwargs["crf"])
                except (ValueError, TypeError):
                    pass
            if "audio_codec" in kwargs:
                ac_val = kwargs["audio_codec"]
                if isinstance(ac_val, dict):
                    audio_codec_str = ac_val.get("audio_codec", "aac")
                    audio_bitrate = ac_val.get("audio_bitrate")
                elif isinstance(ac_val, str):
                    audio_codec_str = ac_val
            if "audio_bitrate" in kwargs and audio_bitrate is None:
                audio_bitrate = kwargs["audio_bitrate"]

        format = format_str
        codec = codec_str
        audio_codec = audio_codec_str

        print(f"[XENodes] SaveVideo parsed: format={format!r}, codec={codec!r}, crf={crf!r}, audio_codec={audio_codec!r}, audio_bitrate={audio_bitrate!r}")

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

        images = components.images  # shape: (N, H, W, 3)
        num_images = images.shape[0]

        frame_indices = generate_frame_indices(num_images, pingpong)
        n_orig = len(frame_indices)
        total_plays = loop_count + 1

        waveform, audio_sample_rate, layout = expand_audio_waveform(components, float(frame_rate), n_orig, total_plays)
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
            crf_defaults = {
                'h264': 23.0, 'h265': 28.0, 'av1': 42.0,
                'h264_nvenc': 30.0, 'hevc_nvenc': 35.0, 'av1_nvenc': 42.0,
            }
            crf = crf_defaults.get(codec, 23.0)

        codec_config = {
            'h264': {'codec': 'libx264', 'pix_fmt': 'yuv420p', 'options': ['-preset', 'slow']},
            'h265': {'codec': 'libx265', 'pix_fmt': 'yuv420p10le', 'options': ['-preset', 'slow']},
            'av1':  {'codec': 'libsvtav1', 'pix_fmt': 'yuv420p10le', 'options': ['-preset', '6']},
            'h264_nvenc': {'codec': 'h264_nvenc', 'pix_fmt': 'yuv420p', 'options': ['-preset', 'p7']},
            'hevc_nvenc': {'codec': 'hevc_nvenc', 'pix_fmt': 'p010le', 'options': ['-preset', 'p7']},
            'av1_nvenc':  {'codec': 'av1_nvenc', 'pix_fmt': 'p010le', 'options': ['-preset', 'p7']}
        }

        config = codec_config.get(codec, codec_config['h264'])
        av_codec = config['codec']
        pix_fmt = config['pix_fmt']
        base_options = config['options']

        metadata_file = _create_ffmetadata_file(saved_metadata)
        audio_file, audio_sr, audio_ch = _prepare_audio_file(waveform, output_sample_rate)

        cmd = [ffmpeg_exe, "-y", "-v", "error"]

        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-framerate", str(float(frame_rate)),
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

        cmd.extend(["-c:v", av_codec, "-pix_fmt", pix_fmt])
        cmd.extend(base_options)

        if "nvenc" in codec:
            cmd.extend(["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"])
        else:
            cmd.extend(["-crf", str(int(crf))])

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

        if format == "mp4":
            cmd.extend(["-movflags", "+use_metadata_tags+faststart"])

        cmd.append(file_path)

        print(f"[XENodes] SaveVideo (FFmpeg): running command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            for chunk in _iter_frame_byte_chunks(images, frame_indices, total_plays):
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
            raise RuntimeError(f"FFmpeg encoding failed (exit code {retcode}): {stderr_out}")

        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)]))

class SaveVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveVideo]

async def comfy_entrypoint() -> SaveVideoExtension:
    return SaveVideoExtension()
