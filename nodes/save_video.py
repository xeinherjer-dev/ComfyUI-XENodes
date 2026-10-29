from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import json
import time
import math
import torch
import av
from fractions import Fraction
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, Input, InputImpl, ui
import folder_paths

from ..utils.audio import expand_audio_waveform
from ..utils.video import generate_frame_indices
from ..utils.metadata import get_saved_metadata
from ..utils.color import pq_to_linear

_MAX_RAW_FRAME_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_LOOP_CACHE_BYTES = 256 * 1024 * 1024
DEFAULT_AUDIO_BITRATE = "128k"

# HDR10 mastering display: P3-D65 primaries (Display P3 gamut inside BT.2020 container).
_MDCV_PRIMARIES_X265 = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)"
_MDCV_PRIMARIES_SVT = "G(0.2650,0.6900)B(0.1500,0.0600)R(0.6800,0.3200)WP(0.3127,0.3290)"
_MDCV_MIN_NITS = 0.0001
_MDCV_DEFAULT_MAX_NITS = 1000
_HDR10_STATS_CHUNK_FRAMES = 16


def _slice_or_index_frames(images: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """
    Returns a zero-copy slice view when `indices` is a contiguous ascending range,
    falling back to list indexing only when non-contiguous (e.g. pingpong reversal boundary).
    """
    n = len(indices)
    if n == 0:
        return images[:0, ..., :3]
    first = indices[0]
    if n == 1 or (indices[-1] == first + n - 1 and all(indices[i] == first + i for i in range(n))):
        return images[first : first + n, ..., :3]
    return images[indices, ..., :3]


def _compute_hdr10_light_levels(images: torch.Tensor, frame_indices: list[int]) -> tuple[int, int]:
    """
    Computes HDR10 content light levels (MaxCLL, MaxFALL) in nits from PQ-encoded BT.2020 frames.
    MaxCLL  = brightest max(R,G,B) of any pixel; MaxFALL = highest frame-average of max(R,G,B).
    """
    unique_indices = sorted(set(frame_indices))
    if not unique_indices:
        return 0, 0

    compute_device = torch.device("cuda") if torch.cuda.is_available() else images.device
    cll_tensors: list[torch.Tensor] = []
    fall_tensors: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(unique_indices), _HDR10_STATS_CHUNK_FRAMES):
            idx = unique_indices[start : start + _HDR10_STATS_CHUNK_FRAMES]
            chunk_view = _slice_or_index_frames(images, idx)
            chunk = chunk_view.detach().to(device=compute_device, dtype=torch.float32, non_blocking=True)
            # pq_to_linear is strictly monotonic on [0, 1], so taking max(R,G,B) in PQ domain first
            # is mathematically identical and reduces expensive power operations by 3x.
            max_rgb_pq = chunk.amax(dim=-1)  # (B, H, W)
            nits = pq_to_linear(max_rgb_pq, nits_scale=1.0)
            cll_tensors.append(nits.amax())
            fall_tensors.append(nits.mean(dim=(1, 2)).amax())

        max_cll = float(torch.stack(cll_tensors).amax().cpu())
        max_fall = float(torch.stack(fall_tensors).amax().cpu())

    max_cll_i = int(min(10000, math.ceil(max_cll)))
    max_fall_i = int(min(max_cll_i, math.ceil(max_fall)))
    return max_cll_i, max_fall_i


def _can_passthrough_video(
    video: Input.Video,
    target_format: str,
    target_audio_codec: str,
    target_audio_bitrate: str | None,
    loop_count: int,
    pingpong: bool,
) -> tuple[bool, str | None, str | None, bool, bool]:
    """
    Checks if the video input can be saved without re-encoding the video track.
    Returns:
        (can_video_passthrough, source_file_path, video_codec_name, has_audio, can_audio_copy)
    """
    if loop_count != 0 or pingpong:
        return False, None, None, False, False

    # Only VideoFromFile can be passed through without re-encoding.
    # VideoFromComponents inherits a fallback get_stream_source() from VideoInput
    # that encodes the entire video into a BytesIO buffer via self.save_to(buffer),
    # which causes an expensive full re-encode (taking 20+ seconds)!
    if not isinstance(video, InputImpl.VideoFromFile) and type(video).__name__ != "VideoFromFile":
        return False, None, None, False, False

    # Check for trim or crop (VideoFromFile attributes)
    if hasattr(video, "get_active_trim_window"):
        start_time, duration = video.get_active_trim_window()
        if start_time != 0 or duration != 0:
            return False, None, None, False, False
    else:
        if getattr(video, "_VideoFromFile__start_time", 0) != 0:
            return False, None, None, False, False
        if getattr(video, "_VideoFromFile__duration", 0) != 0:
            return False, None, None, False, False

    if getattr(video, "_VideoFromFile__crop", None) is not None:
        return False, None, None, False, False

    source = video.get_stream_source()
    if not isinstance(source, (str, os.PathLike)) or not os.path.exists(source):
        return False, None, None, False, False

    source_path = str(source)

    try:
        with av.open(source_path, mode="r") as container:
            if not container.streams.video:
                return False, None, None, False, False
            v_stream = container.streams.video[0]
            v_codec = v_stream.codec.canonical_name if v_stream.codec else ""

            # Video container compatibility check
            if target_format == "mp4":
                if v_codec not in {"h264", "hevc", "av1", "mpeg4"}:
                    return False, None, None, False, False
            elif target_format == "webm":
                if v_codec not in {"av1", "vp9", "vp8"}:
                    return False, None, None, False, False
            else:
                return False, None, None, False, False

            # Check audio streams
            has_audio = len(container.streams.audio) > 0
            can_audio_copy = False
            if has_audio:
                a_stream = container.streams.audio[0]
                a_codec = a_stream.codec.canonical_name if a_stream.codec else ""
                
                # Check if the source audio codec matches the requested format and target codec
                if target_format == "mp4":
                    allowed_codecs = {"aac", "mp3", "opus", "flac", "ac3", "eac3"}
                    if a_codec in allowed_codecs:
                        if target_audio_codec == a_codec and target_audio_bitrate is None:
                            can_audio_copy = True
                        elif target_audio_codec == "aac" and a_codec == "aac":
                            can_audio_copy = (target_audio_bitrate is None or target_audio_bitrate == DEFAULT_AUDIO_BITRATE)
                elif target_format == "webm":
                    allowed_codecs = {"opus", "vorbis"}
                    if a_codec in allowed_codecs:
                        if target_audio_codec == a_codec and target_audio_bitrate is None:
                            can_audio_copy = True
                        elif target_audio_codec == "opus" and a_codec == "opus":
                            can_audio_copy = (target_audio_bitrate is None or target_audio_bitrate == DEFAULT_AUDIO_BITRATE)

            return True, source_path, v_codec, has_audio, can_audio_copy

    except Exception as e:
        print(f"[XENodes] SaveVideo passthrough check failed: {e}")
        return False, None, None, False, False

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
    interleaved = waveform.detach().t().to(device="cpu", dtype=torch.float32).contiguous().clamp_(-1.0, 1.0)
    
    handle = tempfile.NamedTemporaryFile(suffix=".f32le", delete=False)
    try:
        handle.write(memoryview(interleaved.numpy()).cast("B"))
    finally:
        handle.close()
    return handle.name, sample_rate, channels


def _iter_frame_byte_chunks(
    images: torch.Tensor,
    frame_indices: list[int],
    total_plays: int,
    is_10bit: bool = False,
    max_chunk_bytes: int = _MAX_RAW_FRAME_CHUNK_BYTES,
):
    """
    Yields zero-copy memoryviews of raw RGB24 or RGB48LE frame chunks.
    Uses slice views when indices are contiguous and offloads float->uint conversion to GPU when available.
    """
    compute_device = torch.device("cuda") if torch.cuda.is_available() else images.device
    bytes_per_frame = images.shape[1] * images.shape[2] * 3 * (2 if is_10bit else 1)
    max_frames_cap = 16 if is_10bit else 32
    frames_per_chunk = max(1, min(max_frames_cap, max_chunk_bytes // max(1, bytes_per_frame)))

    single_play_bytes = len(frame_indices) * bytes_per_frame
    can_cache_loop = total_plays > 1 and single_play_bytes <= _MAX_LOOP_CACHE_BYTES
    cached_chunks: list[memoryview] = []

    def _convert_single_play():
        with torch.no_grad():
            for start in range(0, len(frame_indices), frames_per_chunk):
                chunk_indices = frame_indices[start : start + frames_per_chunk]
                chunk_view = _slice_or_index_frames(images, chunk_indices)
                chunk = chunk_view.detach().to(device=compute_device, dtype=torch.float32, non_blocking=True).clamp_(0.0, 1.0)
                if is_10bit:
                    out = (chunk * 65535.0).round_().to(torch.uint16).cpu()
                else:
                    out = (chunk * 255.0).round_().to(torch.uint8).cpu()
                yield memoryview(out.numpy()).cast("B")

    if can_cache_loop:
        for mv in _convert_single_play():
            cached_chunks.append(mv)
            yield mv
        for _ in range(total_plays - 1):
            for mv in cached_chunks:
                yield mv
    else:
        for _ in range(total_plays):
            yield from _convert_single_play()


def _iter_yuv420_byte_chunks(
    images: torch.Tensor,
    frame_indices: list[int],
    total_plays: int,
    pix_fmt: str,
    is_hdr: bool,
    max_chunk_bytes: int = _MAX_RAW_FRAME_CHUNK_BYTES,
    use_pinned_ring: bool = False,
):
    """
    Converts RGB float32 [0, 1] frames directly to ITU-R BT.709 / BT.2020 Limited Range YUV 4:2:0
    (yuv420p, yuv420p10le, or p010le) on GPU/CPU.
    Halves pipe transfer bandwidth vs RGB24/RGB48LE and eliminates FFmpeg's CPU swscale bottleneck.
    """
    compute_device = torch.device("cuda") if torch.cuda.is_available() else images.device
    H, W = int(images.shape[1]), int(images.shape[2])
    is_10bit = pix_fmt in ("yuv420p10le", "p010le")
    bytes_per_frame = (H * W * 3) if is_10bit else ((H * W * 3) // 2)
    max_frames_cap = 16 if is_10bit else 32
    frames_per_chunk = max(1, min(max_frames_cap, max_chunk_bytes // max(1, bytes_per_frame)))

    if is_hdr:
        # ITU-R BT.2020 non-constant luminance (bt2020nc)
        kr, kg, kb = 0.2627, 0.6780, 0.0593
    else:
        # ITU-R BT.709
        kr, kg, kb = 0.2126, 0.7152, 0.0722

    inv_cb_scale = 1.0 / (2.0 * (1.0 - kb))
    inv_cr_scale = 1.0 / (2.0 * (1.0 - kr))

    single_play_bytes = len(frame_indices) * bytes_per_frame
    can_cache_loop = total_plays > 1 and single_play_bytes <= _MAX_LOOP_CACHE_BYTES
    cached_chunks: list[memoryview] = []

    out_dtype = torch.uint16 if is_10bit else torch.uint8
    elems_per_frame = (H * W * 3) // 2
    enable_ring = use_pinned_ring and (not can_cache_loop) and compute_device.type == "cuda"
    # 4 slots safely cover the producer + 2 queued items + 1 active consumer write in _write_chunks_pipelined
    pinned_ring = (
        [torch.empty((frames_per_chunk, elems_per_frame), dtype=out_dtype, pin_memory=True) for _ in range(4)]
        if enable_ring else None
    )
    ring_idx = 0

    def _convert_single_play():
        nonlocal ring_idx
        with torch.no_grad():
            for start in range(0, len(frame_indices), frames_per_chunk):
                chunk_indices = frame_indices[start : start + frames_per_chunk]
                chunk_view = _slice_or_index_frames(images, chunk_indices)
                x = chunk_view.detach().to(device=compute_device, dtype=torch.float32, non_blocking=True).clamp_(0.0, 1.0)
                B = x.shape[0]

                # Luma plane Y' in [0, 1] at full resolution (B, H, W)
                y_f = kr * x[..., 0] + kg * x[..., 1] + kb * x[..., 2]

                # Chroma 2x2 subsampling: since matrix transform is linear, averaging 2x2 RGB blocks first
                # is mathematically identical to averaging Y, Cb, Cr and reduces chroma matrix math by 4x.
                x_sub = x.view(B, H // 2, 2, W // 2, 2, 3).mean(dim=(2, 4))
                r_s = x_sub[..., 0]
                g_s = x_sub[..., 1]
                b_s = x_sub[..., 2]
                y_s = kr * r_s + kg * g_s + kb * b_s
                cb_f = (b_s - y_s) * inv_cb_scale
                cr_f = (r_s - y_s) * inv_cr_scale

                if pix_fmt == "yuv420p":
                    y_q = (y_f * 219.0 + 16.5).floor_().clamp_(16.0, 235.0).to(torch.uint8)
                    u_q = (cb_f * 224.0 + 128.5).floor_().clamp_(16.0, 240.0).to(torch.uint8)
                    v_q = (cr_f * 224.0 + 128.5).floor_().clamp_(16.0, 240.0).to(torch.uint8)
                    out_gpu = torch.cat([
                        y_q.view(B, H * W),
                        u_q.view(B, (H // 2) * (W // 2)),
                        v_q.view(B, (H // 2) * (W // 2)),
                    ], dim=1)
                elif pix_fmt == "yuv420p10le":
                    y_q = (y_f * 876.0 + 64.5).floor_().clamp_(64.0, 940.0).to(torch.uint16)
                    u_q = (cb_f * 896.0 + 512.5).floor_().clamp_(64.0, 960.0).to(torch.uint16)
                    v_q = (cr_f * 896.0 + 512.5).floor_().clamp_(64.0, 960.0).to(torch.uint16)
                    out_gpu = torch.cat([
                        y_q.view(B, H * W),
                        u_q.view(B, (H // 2) * (W // 2)),
                        v_q.view(B, (H // 2) * (W // 2)),
                    ], dim=1)
                else:  # p010le (semi-planar, 10-bit values stored in upper 10 bits: * 64)
                    y_10 = (y_f * 876.0 + 64.5).floor_().clamp_(64.0, 940.0)
                    u_10 = (cb_f * 896.0 + 512.5).floor_().clamp_(64.0, 960.0)
                    v_10 = (cr_f * 896.0 + 512.5).floor_().clamp_(64.0, 960.0)
                    uv_10 = torch.stack([u_10, v_10], dim=-1)
                    out_gpu = (torch.cat([
                        y_10.view(B, H * W),
                        uv_10.view(B, (H // 2) * (W // 2) * 2),
                    ], dim=1) * 64.0).to(torch.uint16)

                if pinned_ring is not None:
                    slot = pinned_ring[ring_idx % 4][:B]
                    ring_idx += 1
                    slot.copy_(out_gpu, non_blocking=False)
                    out_cpu = slot
                else:
                    out_cpu = out_gpu.cpu()

                yield memoryview(out_cpu.numpy()).cast("B")

    if can_cache_loop:
        for mv in _convert_single_play():
            cached_chunks.append(mv)
            yield mv
        for _ in range(total_plays - 1):
            for mv in cached_chunks:
                yield mv
    else:
        for _ in range(total_plays):
            yield from _convert_single_play()


def _write_chunks_pipelined(stdin_pipe, chunk_iter) -> None:
    """
    Overlaps GPU/CPU chunk conversion on the main thread with blocking pipe writes to FFmpeg
    on a dedicated I/O worker thread (double-buffered queue).
    """
    q: queue.Queue[memoryview | bytes | None] = queue.Queue(maxsize=2)
    writer_error: list[BaseException] = []

    def _writer():
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                stdin_pipe.write(item)
        except BaseException as exc:
            writer_error.append(exc)
            # Drain remaining items so producer never blocks on q.put()
            while True:
                try:
                    if q.get_nowait() is None:
                        break
                except queue.Empty:
                    break

    t = threading.Thread(target=_writer, daemon=True)
    t.start()
    try:
        for chunk in chunk_iter:
            if writer_error:
                break
            q.put(chunk)
    finally:
        q.put(None)
        t.join()
        try:
            stdin_pipe.close()
        except BrokenPipeError:
            pass
    if writer_error and not isinstance(writer_error[0], BrokenPipeError):
        raise writer_error[0]

def _parse_options(format_input: dict | str, kwargs: dict) -> tuple[str, str, float | None, str, str | None]:
    sources = []
    if isinstance(format_input, dict):
        sources.append(format_input)
    elif isinstance(format_input, str):
        sources.append({"format": format_input})
    if kwargs:
        sources.append(kwargs)

    format_str = "mp4"
    codec_str = "h264"
    crf = None
    audio_codec_str = "aac"
    audio_bitrate = None

    for src in sources:
        if "format" in src and isinstance(src["format"], str):
            format_str = src["format"]
        
        for k in ("codec", "format.codec"):
            if k in src:
                c_val = src[k]
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

        for k in ("crf", "format.codec.crf"):
            if k in src and src[k] is not None:
                try:
                    crf = float(src[k])
                except (ValueError, TypeError):
                    pass

        for k in ("audio_codec", "format.audio_codec"):
            if k in src:
                ac_val = src[k]
                if isinstance(ac_val, dict):
                    if "audio_codec" in ac_val and isinstance(ac_val["audio_codec"], str):
                        audio_codec_str = ac_val["audio_codec"]
                    if "audio_bitrate" in ac_val and ac_val["audio_bitrate"] is not None:
                        audio_bitrate = str(ac_val["audio_bitrate"])
                elif isinstance(ac_val, str):
                    audio_codec_str = ac_val

        for k in ("audio_bitrate", "format.audio_codec.audio_bitrate"):
            if k in src and src[k] is not None:
                audio_bitrate = str(src[k])

    if codec_str == "auto":
        codec_str = "av1" if format_str == "webm" else "h264"

    return format_str, codec_str, crf, audio_codec_str, audio_bitrate

class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveVideo",
            display_name="Save Video",
            category="xenodes/video",
            description="Saves the input video (SDR or HDR) natively with AV1/CRF and NVENC support.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save (supports both SDR and 10-bit HDR videos)."),
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
                                        io.DynamicCombo.Option("av1_nvenc", [io.Float.Input("crf", default=42.0, min=0.0, max=51.0, step=1.0, optional=True, tooltip="CQ for NVENC AV1.")]),
                                    ],
                                    tooltip="The video codec.",
                                ),
                                io.DynamicCombo.Input(
                                    "audio_codec",
                                    options=[
                                        io.DynamicCombo.Option(
                                            "aac",
                                            [io.Combo.Input("audio_bitrate", options=["64k", "128k", "192k", "256k", "320k"], default=DEFAULT_AUDIO_BITRATE, optional=True, tooltip="Bitrate for AAC audio.")],
                                        ),
                                        io.DynamicCombo.Option(
                                            "opus",
                                            [io.Combo.Input("audio_bitrate", options=["64k", "128k", "192k", "256k", "320k"], default=DEFAULT_AUDIO_BITRATE, optional=True, tooltip="Bitrate for Opus audio.")],
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
                                        io.DynamicCombo.Option("av1_nvenc", [io.Float.Input("crf", default=42.0, min=0.0, max=51.0, step=1.0, optional=True, tooltip="CQ for NVENC AV1.")]),
                                    ],
                                    tooltip="The video codec.",
                                ),
                                io.DynamicCombo.Input(
                                    "audio_codec",
                                    options=[
                                        io.DynamicCombo.Option(
                                            "opus",
                                            [io.Combo.Input("audio_bitrate", options=["64k", "128k", "192k", "256k", "320k"], default=DEFAULT_AUDIO_BITRATE, optional=True, tooltip="Bitrate for Opus audio.")],
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
        t0 = time.perf_counter()
        
        format_str, codec_str, crf, audio_codec_str, audio_bitrate = _parse_options(format, kwargs)

        format = format_str
        codec = codec_str
        audio_codec = audio_codec_str

        color_space = video.get_color_space()
        bit_depth = video.get_bit_depth()
        # HDR is determined by the transfer characteristic only. A 10-bit SDR (BT.709) source must not be
        # tagged as PQ/BT.2020.
        is_hdr = color_space in ("HDR PQ", "HDR")

        print(f"[XENodes] SaveVideo parsed: format={format!r}, codec={codec!r}, crf={crf!r}, color_space={color_space!r}, bit_depth={bit_depth!r}, is_hdr={is_hdr}")

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

        ffmpeg_exe = find_ffmpeg()
        if not ffmpeg_exe:
            raise RuntimeError("FFmpeg executable not found. Please install ffmpeg or imageio-ffmpeg.")

        # Check if the input video can be reused without re-encoding (auto passthrough)
        can_passthrough, source_file, source_vcodec, has_audio, can_audio_copy = _can_passthrough_video(
            video, format, audio_codec, audio_bitrate, loop_count, pingpong
        )

        if can_passthrough and source_file:
            passthrough_mode = "video+audio copy" if (has_audio and can_audio_copy) else ("video copy + audio re-encode" if has_audio else "video copy (no audio)")
            print(f"[XENodes] SaveVideo: Input unchanged, reusing original video ({passthrough_mode}).")
            metadata_file = _create_ffmetadata_file(saved_metadata)
            try:
                cmd = [ffmpeg_exe, "-y", "-v", "error", "-i", source_file]

                if metadata_file is not None:
                    cmd.extend(["-f", "ffmetadata", "-i", metadata_file, "-map_metadata", "1"])
                else:
                    cmd.extend(["-map_metadata", "0"])

                cmd.extend(["-map", "0:v:0", "-c:v", "copy"])

                if has_audio:
                    cmd.extend(["-map", "0:a:0"])
                    if can_audio_copy:
                        cmd.extend(["-c:a", "copy"])
                    else:
                        audio_codec_map = {
                            "aac": "aac",
                            "opus": "libopus",
                            "flac": "flac"
                        }
                        av_audio_codec = audio_codec_map.get(audio_codec, "aac")
                        if format == "webm" and av_audio_codec == "aac":
                            av_audio_codec = "libopus"
                        cmd.extend(["-c:a", av_audio_codec])
                        if av_audio_codec == "libopus":
                            cmd.extend(["-ar", "48000"])
                        if audio_bitrate is not None and av_audio_codec != "flac":
                            cmd.extend(["-b:a", audio_bitrate])

                if format == "mp4":
                    cmd.extend(["-movflags", "+use_metadata_tags+faststart"])
                    if source_vcodec in ("hevc", "h265"):
                        cmd.extend(["-tag:v", "hvc1"])

                cmd.append(file_path)

                print(f"[XENodes] SaveVideo (FFmpeg Passthrough): running command: {' '.join(cmd)}")
                t_encode_start = time.perf_counter()
                ret = subprocess.run(cmd, capture_output=True, text=True)
                t_encode_end = time.perf_counter()
            finally:
                if metadata_file and os.path.exists(metadata_file):
                    os.unlink(metadata_file)

            if ret.returncode == 0:
                print(
                    f"[XENodes] SaveVideo breakdown:\n"
                    f"  - passthrough:   {t_encode_end - t_encode_start:.3f}s\n"
                    f"  - TOTAL:         {t_encode_end - t0:.3f}s"
                )
                return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)]))
            else:
                print(f"[XENodes] SaveVideo passthrough failed (code {ret.returncode}: {ret.stderr.strip()}), falling back to re-encode.")

        t_prep_start = time.perf_counter()
        
        components = video.get_components()
        t_get_comp = time.perf_counter()

        frame_rate = Fraction(round(components.frame_rate * 1000), 1000)
        images = components.images  # shape: (N, H, W, 3)
        num_images = images.shape[0]

        frame_indices = generate_frame_indices(num_images, pingpong)
        n_orig = len(frame_indices)
        total_plays = loop_count + 1

        waveform, audio_sample_rate, layout = expand_audio_waveform(components, float(frame_rate), n_orig, total_plays)
        t_expand_audio = time.perf_counter()

        # h264_nvenc does not support 10-bit YUV in hardware; fall back to libx264 for HDR.
        if is_hdr and codec == 'h264_nvenc':
            print("[XENodes] SaveVideo: h264_nvenc does not support 10-bit HDR; falling back to libx264.")
            codec = 'h264'

        if crf is None:
            crf_defaults = {
                'h264': 23.0, 'h265': 28.0, 'av1': 42.0,
                'h264_nvenc': 30.0, 'hevc_nvenc': 35.0, 'av1_nvenc': 42.0,
            }
            crf = crf_defaults.get(codec, 23.0)

        # HEVC/AV1 always use 10-bit (also for SDR) to minimise banding, matching libx265/libsvtav1.
        # H.264 stays 8-bit for SDR because 10-bit H.264 has poor hardware decoder support.
        codec_config = {
            'h264': {'codec': 'libx264', 'pix_fmt': 'yuv420p10le' if is_hdr else 'yuv420p', 'options': ['-preset', 'slow']},
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
        is_10bit_pipe = pix_fmt in ('yuv420p10le', 'p010le')
        # When dimensions are even, convert RGB->YUV420 directly in PyTorch (GPU/CPU) to halve pipe
        # bandwidth and eliminate FFmpeg's single-threaded CPU swscale bottleneck.
        use_direct_yuv420 = (width % 2 == 0 and height % 2 == 0)

        metadata_file = _create_ffmetadata_file(saved_metadata)
        # Pass the dynamic input audio_sample_rate to _prepare_audio_file (same pattern as DaSiWa)
        audio_file, audio_sr, audio_ch = _prepare_audio_file(waveform, audio_sample_rate)
        t_files_prep = time.perf_counter()

        cmd = [ffmpeg_exe, "-y", "-v", "error"]

        if use_direct_yuv420:
            input_pix_fmt = pix_fmt
            cmd.extend([
                "-f", "rawvideo",
                "-pix_fmt", input_pix_fmt,
                "-s", f"{width}x{height}",
                "-framerate", str(float(frame_rate)),
                "-color_range", "tv",
                "-colorspace", "bt2020nc" if is_hdr else "bt709",
                "-color_primaries", "bt2020" if is_hdr else "bt709",
                "-color_trc", ("arib-std-b67" if color_space == "HDR" else "smpte2084") if is_hdr else "bt709",
                "-i", "-",
            ])
        else:
            input_pix_fmt = "rgb48le" if is_10bit_pipe else "rgb24"
            cmd.extend([
                "-f", "rawvideo",
                "-pix_fmt", input_pix_fmt,
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

        # Video filter for color matrix conversion and even-dimension alignment (only needed for odd-dimension RGB fallback)
        if not use_direct_yuv420:
            if is_hdr:
                vf_filter = "scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2:in_color_matrix=bt2020:out_color_matrix=bt2020:in_range=full:out_range=tv:flags=bicubic"
            else:
                vf_filter = "scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2:in_color_matrix=bt709:out_color_matrix=bt709:in_range=full:out_range=tv:flags=bicubic"
            cmd.extend(["-vf", vf_filter])

        cmd.extend(["-c:v", av_codec, "-pix_fmt", pix_fmt])
        cmd.extend(base_options)

        if "nvenc" in codec:
            cmd.extend(["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"])
        else:
            cmd.extend(["-crf", str(int(crf))])

        if format == "mp4":
            cmd.extend(["-movflags", "+use_metadata_tags+faststart"])
            if codec in ("h265", "hevc_nvenc"):
                cmd.extend(["-tag:v", "hvc1"])

        # Color Properties & Metadata
        if is_hdr:
            if color_space == "HDR":  # HLG
                color_primaries = "bt2020"
                color_trc = "arib-std-b67"
                colorspace = "bt2020nc"
                svt_trc = "18"  # ARIB STD-B67
                x264_transfer = "arib-std-b67"
                x265_transfer = "arib-std-b67"
                bsf_trc = "18"
            else:  # HDR PQ
                color_primaries = "bt2020"
                color_trc = "smpte2084"
                colorspace = "bt2020nc"
                svt_trc = "16"  # SMPTE ST 2084
                x264_transfer = "smpte2084"
                x265_transfer = "smpte2084"
                bsf_trc = "16"

            cmd.extend([
                "-color_primaries", color_primaries,
                "-color_trc", color_trc,
                "-colorspace", colorspace,
                "-color_range", "tv",
            ])

            # HDR10 static metadata (PQ only): P3-D65 mastering primaries, content light levels measured
            # from the actual frames so that displays tone-map according to the real content peak.
            is_pq = color_space != "HDR"
            if is_pq:
                max_cll, max_fall = _compute_hdr10_light_levels(images, frame_indices)
                mdcv_max = max(_MDCV_DEFAULT_MAX_NITS, max_cll)
                print(f"[XENodes] SaveVideo: HDR10 MaxCLL={max_cll} MaxFALL={max_fall} mastering_max={mdcv_max} nits")
                # x264/x265 units: chromaticity 0.00002, luminance 0.0001 cd/m2
                xl_lum = f"L({mdcv_max * 10000},{round(_MDCV_MIN_NITS * 10000)})"
                x264_mdcv = f"mastering-display={_MDCV_PRIMARIES_X265}{xl_lum}:cll={max_cll},{max_fall}"
                x265_mdcv = f"master-display={_MDCV_PRIMARIES_X265}{xl_lum}:max-cll={max_cll},{max_fall}"
                svt_mdcv = (
                    f"mastering-display={_MDCV_PRIMARIES_SVT}L({mdcv_max:.1f},{_MDCV_MIN_NITS})"
                    f":content-light={max_cll},{max_fall}"
                )

            if av_codec == "libx264":
                # NOTE: x264 option names differ from x265 (range=tv, mastering-display, cll).
                x264_params = f"colorprim=bt2020:transfer={x264_transfer}:colormatrix=bt2020nc:range=tv"
                if is_pq:
                    x264_params += f":{x264_mdcv}"
                cmd.extend([
                    "-x264-params", x264_params,
                    "-bsf:v",
                    f"h264_metadata=colour_primaries=9:transfer_characteristics={bsf_trc}:matrix_coefficients=9:video_full_range_flag=0",
                ])
            elif av_codec == "libx265":
                x265_params = f"colorprim=bt2020:transfer={x265_transfer}:colormatrix=bt2020nc:range=limited"
                if is_pq:
                    x265_params += f":{x265_mdcv}:hdr10=1"
                cmd.extend([
                    "-x265-params", x265_params,
                    "-bsf:v",
                    f"hevc_metadata=colour_primaries=9:transfer_characteristics={bsf_trc}:matrix_coefficients=9:video_full_range_flag=0",
                ])
            elif av_codec == "libsvtav1":
                svt_params = f"enable-hdr=1:color-primaries=9:transfer-characteristics={svt_trc}:matrix-coefficients=9:color-range=0"
                if is_pq:
                    svt_params += f":{svt_mdcv}"
                cmd.extend([
                    "-svtav1-params", svt_params,
                    "-bsf:v",
                    f"av1_metadata=color_primaries=9:transfer_characteristics={bsf_trc}:matrix_coefficients=9:color_range=tv",
                ])
            elif av_codec == "hevc_nvenc":
                cmd.extend([
                    "-bsf:v",
                    f"hevc_metadata=colour_primaries=9:transfer_characteristics={bsf_trc}:matrix_coefficients=9:video_full_range_flag=0",
                ])
            elif av_codec == "av1_nvenc":
                cmd.extend([
                    "-bsf:v",
                    f"av1_metadata=color_primaries=9:transfer_characteristics={bsf_trc}:matrix_coefficients=9:color_range=tv",
                ])
        else:
            cmd.extend([
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-colorspace", "bt709",
                "-color_range", "tv",
            ])

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

        print(f"[XENodes] SaveVideo (FFmpeg): running command: {' '.join(cmd)}")

        t_encode_start = time.perf_counter()

        with tempfile.TemporaryFile() as stderr_tmp:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_tmp,
            )

            try:
                if use_direct_yuv420:
                    chunk_iter = _iter_yuv420_byte_chunks(
                        images, frame_indices, total_plays, pix_fmt=pix_fmt, is_hdr=is_hdr, use_pinned_ring=True
                    )
                else:
                    chunk_iter = _iter_frame_byte_chunks(
                        images, frame_indices, total_plays, is_10bit=is_10bit_pipe
                    )
                _write_chunks_pipelined(process.stdin, chunk_iter)
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

            t_encode_end = time.perf_counter()

            if retcode != 0:
                stderr_tmp.seek(0)
                stderr_out = stderr_tmp.read().decode(errors="replace")
                raise RuntimeError(f"FFmpeg encoding failed (exit code {retcode}): {stderr_out}")

        print(
            f"[XENodes] SaveVideo breakdown:\n"
            f"  - prepare:       {t_encode_start - t0:.3f}s\n"
            f"  - ffmpeg encode: {t_encode_end - t_encode_start:.3f}s\n"
            f"  - TOTAL:         {t_encode_end - t0:.3f}s"
        )

        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)]))

class SaveVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveVideo]

async def comfy_entrypoint() -> SaveVideoExtension:
    return SaveVideoExtension()
