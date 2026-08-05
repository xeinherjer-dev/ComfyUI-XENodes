from __future__ import annotations

import os
import av
import math
import torch
import json
from fractions import Fraction
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, Input, ui
import folder_paths

from ..utils.audio import expand_audio_waveform, encode_audio_to_stream
from ..utils.video import generate_frame_indices
from ..utils.metadata import get_saved_metadata

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
                                        io.DynamicCombo.Option("av1_nvenc", [io.Float.Input("crf", default=42.0, min=0.0, max=51.0, step=1.0, optional=True, tooltip="CQ for NVENC AV1.")]),
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
                                        io.DynamicCombo.Option("av1_nvenc", [io.Float.Input("crf", default=42.0, min=0.0, max=51.0, step=1.0, optional=True, tooltip="CQ for NVENC AV1.")]),
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
        crf: float = 23.0,
        loop_count: int = 0,
        pingpong: bool = False,
        codec: dict | str = "h264",
        audio_codec: dict | str = "aac",
    ) -> io.NodeOutput:
        audio_bitrate = None
        format_str = "mp4"
        codec_str = "h264"
        audio_codec_str = "aac"

        if isinstance(format, dict):
            format_str = format.get("format", "mp4")
            codec_obj = format.get("codec")
            if isinstance(codec_obj, dict):
                codec_str = codec_obj.get("codec", "h264")
                crf = codec_obj.get("crf", crf)
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

        if isinstance(codec, dict):
            codec_str = codec.get("codec", codec_str)
            crf = codec.get("crf", crf)
        elif isinstance(codec, str):
            codec_str = codec

        if isinstance(audio_codec, dict):
            audio_codec_str = audio_codec.get("audio_codec", audio_codec_str)
            audio_bitrate = audio_codec.get("audio_bitrate")
        elif isinstance(audio_codec, str):
            audio_codec_str = audio_codec

        # Align variables for rest of execute method
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

        # === Frame sequence generation ===
        images = components.images  # shape: (N, H, W, 3)
        num_images = images.shape[0]

        # Generate lightweight index list for streaming frames
        frame_indices = generate_frame_indices(num_images, pingpong)
        n_orig = len(frame_indices) # Audio logic automatically scales to this elongated pingpong loop length

        # loop: 0 = play once, N > 0 = loop N times (play N+1 times total)
        total_plays = loop_count + 1

        # === Audio transformation ===
        waveform, audio_sample_rate, layout = expand_audio_waveform(components, float(frame_rate), n_orig, total_plays)
        output_sample_rate = audio_sample_rate
        if waveform is not None:
            # Resampling prep for specific codecs like Opus (requires 48k)
            if audio_codec == "opus":
                output_sample_rate = 48000
        else:
            output_sample_rate = 44100

        codec_config = {
            'h264': {'codec': 'libx264', 'pix_fmt': 'yuv420p', 'options': {'preset': 'slow'}},
            'h265': {'codec': 'libx265', 'pix_fmt': 'yuv420p10le', 'options': {'preset': 'slow'}},
            'av1':  {'codec': 'libsvtav1', 'pix_fmt': 'yuv420p10le', 'options': {'preset': '6'}},
            'h264_nvenc': {'codec': 'h264_nvenc', 'pix_fmt': 'yuv420p', 'options': {'preset': 'p7'}},
            'hevc_nvenc': {'codec': 'hevc_nvenc', 'pix_fmt': 'p010le', 'options': {'preset': 'p7'}},
            'av1_nvenc':  {'codec': 'av1_nvenc', 'pix_fmt': 'p010le', 'options': {'preset': 'p7'}}
        }

        config = codec_config.get(codec, codec_config['h264'])
        av_codec = config['codec']
        pix_fmt = config['pix_fmt']

        container_options = {}
        if format == 'mp4':
            container_options['movflags'] = 'use_metadata_tags+faststart'

        with av.open(file_path, mode='w', options=container_options) as output:
            if saved_metadata:
                for key, value in saved_metadata.items():
                    if isinstance(value, str):
                        output.metadata[key] = value
                    else:
                        output.metadata[key] = json.dumps(value)

            video_stream = output.add_stream(av_codec, rate=frame_rate)
            video_stream.width = images.shape[2]
            video_stream.height = images.shape[1]
            video_stream.pix_fmt = pix_fmt

            # Quality mapping
            opts = {}
            base_options = config.get('options')
            if isinstance(base_options, dict):
                opts.update(base_options)
            if crf > 0:
                if 'nvenc' in codec:
                    opts['rc'] = 'vbr'
                    opts['cq'] = str(int(crf))
                    opts['b:v'] = '0'
                else:
                    opts['crf'] = str(int(crf))
            if opts:
                video_stream.options = opts

            audio_stream = None
            if waveform is not None:
                try:
                    audio_codec_map = {
                        "aac": "aac",
                        "opus": "libopus",
                        "flac": "flac"
                    }
                    av_audio_codec = audio_codec_map.get(audio_codec, "aac")
                    audio_stream = output.add_stream(av_audio_codec, rate=output_sample_rate, layout=layout)
                    
                    if audio_codec != "flac" and audio_bitrate is not None:
                        audio_stream.bit_rate = int(audio_bitrate.replace("k", "000"))
                except Exception as e:
                    print(f"[XENodes] Warning: Failed to add audio stream: {e}")
                    audio_stream = None

            # Encode modified frames
            for _ in range(total_plays):
                for idx in frame_indices:
                    frame_tensor = images[idx] # Single index creates a view, no memory copy
                    img = (frame_tensor * 255).clamp(0, 255).byte().cpu().numpy()
                    
                    frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                    frame = frame.reformat(format=pix_fmt)
                    for packet in video_stream.encode(frame):
                        output.mux(packet)

            # Flush video encoder
            for packet in video_stream.encode(None):
                output.mux(packet)

            # Encode audio
            if audio_stream is not None and waveform is not None:
                try:
                    encode_audio_to_stream(output, audio_stream, waveform, audio_sample_rate, output_sample_rate, layout)
                except Exception as e:
                    print(f"[XENodes] Error during audio encoding in video: {e}")

        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)]))

class SaveVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveVideo]

async def comfy_entrypoint() -> SaveVideoExtension:
    return SaveVideoExtension()
