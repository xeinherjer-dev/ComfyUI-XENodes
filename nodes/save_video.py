from __future__ import annotations

import os
import av
import math
import torch
import json
from fractions import Fraction
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, Input, ui
from comfy.cli_args import args
import folder_paths

class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveVideo",
            display_name="Save Video",
            category="XENodes",
            description="Saves the input video natively with AV1/CRF support, independently of core save_to.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("format", options=["mp4", "webm"], default="mp4", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=["h264", "h265", "av1", "h264_nvenc", "hevc_nvenc", "av1_nvenc"], default="h264", tooltip="The codec to use for the video."),
                io.Float.Input("crf", default=0.0, min=0.0, max=63.0, step=1.0, tooltip="Specific CRF value used for encoding (maps to CQ for NVENC). Set to 0 to use encoder defaults."),
                io.Int.Input("loop_count", default=0, min=0, max=100, step=1, tooltip="Loop count. 0 = play once. For mp4/webm, this physically repeats the frames."),
                io.Boolean.Input("pingpong", default=False, tooltip="Pingpong animation (images only). Plays frames forward then backward."),
                io.Combo.Input("audio_codec", options=["aac", "opus", "flac"], default="aac", tooltip="The codec to use for the audio."),
                io.Combo.Input("audio_bitrate", options=["64k", "128k", "192k", "256k", "320k"], default="128k", tooltip="The bitrate to use for the audio codec (ignored if flac)."),
            ],
            outputs=[io.Video.Output("video")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix: str, format: str, codec: str, crf: float, loop_count: int, pingpong: bool, audio_codec: str, audio_bitrate: str) -> io.NodeOutput:
        width, height = video.get_dimensions()
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

        file_name = f"{filename}_{counter:05}_.{format}"
        file_path = os.path.join(full_output_folder, file_name)

        components = video.get_components()
        frame_rate = Fraction(round(components.frame_rate * 1000), 1000)

        # === Frame sequence generation ===
        images = components.images  # shape: (N, H, W, 3)
        num_images = images.shape[0]

        # Generate lightweight index list for streaming frames
        if pingpong and num_images > 2:
            frame_indices = list(range(num_images)) + list(range(num_images - 2, 0, -1))
        else:
            frame_indices = list(range(num_images))
            
        n_orig = len(frame_indices) # Audio logic automatically scales to this elongated pingpong loop length

        # loop: 0 = play once, N > 0 = loop N times (play N+1 times total)
        total_plays = loop_count + 1

        # === Audio transformation ===
        audio_sample_rate = 1
        waveform = None
        layout = 'stereo'

        if getattr(components, 'audio', None) is not None:
            try:
                audio_sample_rate = int(components.audio['sample_rate'])
                raw_waveform = components.audio['waveform']  # shape: (batch, channels, samples)
                samples_per_frame = audio_sample_rate / float(frame_rate)

                raw_waveform = raw_waveform[0]  # shape: (channels, samples)
                n_orig_samples = math.ceil(samples_per_frame * n_orig)
                total_samples_needed = math.ceil(samples_per_frame * (n_orig * total_plays))

                if raw_waveform.shape[-1] > n_orig_samples:
                    # Original audio is longer than 1 loop of images. Use the continuous audio.
                    if raw_waveform.shape[-1] >= total_samples_needed:
                        waveform = raw_waveform[:, :total_samples_needed]
                    else:
                        # If audio runs out before all loops finish, repeat the available audio
                        repeats = math.ceil(total_samples_needed / raw_waveform.shape[-1])
                        waveform = torch.cat([raw_waveform] * repeats, dim=-1)[:, :total_samples_needed]
                else:
                    # Regular case: audio is exactly image length (or shorter). Repeat per image loop.
                    waveform = raw_waveform[:, :n_orig_samples]
                    if total_plays > 1:
                        waveform = torch.cat([waveform] * total_plays, dim=-1)

                layout = {1: 'mono', 2: 'stereo', 6: '5.1'}.get(waveform.shape[0], 'stereo')
                
                # Resampling prep for specific codecs like Opus (requires 48k)
                output_sample_rate = audio_sample_rate
                if audio_codec == "opus":
                    output_sample_rate = 48000
            except Exception as e:
                print(f"[XENodes] Warning: Failed to process audio stream: {e}")
                waveform = None
                output_sample_rate = 44100

        codec_config = {
            'h264': {'codec': 'libx264', 'pix_fmt': 'yuv420p'},
            'h265': {'codec': 'libx265', 'pix_fmt': 'yuv420p10le'},
            'av1':  {'codec': 'libsvtav1', 'pix_fmt': 'yuv420p10le', 'options': {'preset': '6'}},
            'h264_nvenc': {'codec': 'h264_nvenc', 'pix_fmt': 'yuv420p'},
            'hevc_nvenc': {'codec': 'hevc_nvenc', 'pix_fmt': 'yuv420p'},
            'av1_nvenc':  {'codec': 'av1_nvenc', 'pix_fmt': 'yuv420p'}
        }

        config = codec_config.get(codec, codec_config['h264'])
        av_codec = config['codec']
        pix_fmt = config['pix_fmt']

        container_options = {}
        if format == 'mp4':
            container_options['movflags'] = 'use_metadata_tags'

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
                    
                    if audio_codec != "flac":
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
                orig_frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format='fltp', layout=layout)
                orig_frame.sample_rate = audio_sample_rate
                orig_frame.pts = 0

                # Actual resampling if needed
                if audio_sample_rate != output_sample_rate:
                    resampler = av.AudioResampler(format='fltp', layout=layout, rate=output_sample_rate)
                    resampled_frames = resampler.resample(orig_frame)
                    # libopus and other encoders might need frames to be a certain size, muxing directly works often
                    for f in resampled_frames:
                        f.pts = None # Let av handle pts for simplicity when resampled
                        output.mux(audio_stream.encode(f))
                else:
                    output.mux(audio_stream.encode(orig_frame))

                # Flush audio encoder
                output.mux(audio_stream.encode(None))

        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)]))

class SaveVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveVideo]

async def comfy_entrypoint() -> SaveVideoExtension:
    return SaveVideoExtension()
