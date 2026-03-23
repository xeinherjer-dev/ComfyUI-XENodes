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

class XESaveVideo(io.ComfyNode):
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
                io.Combo.Input("codec", options=["h264", "h265", "av1"], default="h264", tooltip="The codec to use for the video."),
                io.Float.Input("crf", default=0.0, min=0.0, max=63.0, step=1.0, tooltip="Specific CRF value used for encoding. Set to 0 to use encoder defaults."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix: str, format: str, codec: str, crf: float) -> io.NodeOutput:
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
        
        codec_config = {
            'h264': {'codec': 'libx264', 'pix_fmt': 'yuv420p'},
            'h265': {'codec': 'libx265', 'pix_fmt': 'yuv420p10le'},
            'av1':  {'codec': 'libsvtav1', 'pix_fmt': 'yuv420p10le', 'options': {'preset': '6'}}
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
            video_stream.width = components.images.shape[2]
            video_stream.height = components.images.shape[1]
            video_stream.pix_fmt = pix_fmt

            # Quality mapping
            opts = {}
            base_options = config.get('options')
            if isinstance(base_options, dict):
                opts.update(base_options)
            if crf > 0:
                opts['crf'] = str(int(crf))
            if opts:
                video_stream.options = opts

            audio_sample_rate = 1
            audio_stream = None
            waveform = None
            
            # Handle possible audio input
            if getattr(components, 'audio', None) is not None:
                try:
                    audio_sample_rate = int(components.audio['sample_rate'])
                    waveform = components.audio['waveform']
                    waveform = waveform[0, :, :math.ceil((audio_sample_rate / frame_rate) * components.images.shape[0])]
                    layout = {1: 'mono', 2: 'stereo', 6: '5.1'}.get(waveform.shape[0], 'stereo')
                    audio_stream = output.add_stream('aac', rate=audio_sample_rate, layout=layout)
                except Exception as e:
                    print(f"[XENodes] Warning: Failed to process audio stream: {e}")
                    audio_stream = None
                    waveform = None

            # Encode video
            for frame_idx, frame_tensor in enumerate(components.images):
                img = (frame_tensor * 255).clamp(0, 255).byte().cpu().numpy() # shape: (H, W, 3)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame = frame.reformat(format=pix_fmt)
                packet = video_stream.encode(frame)
                output.mux(packet)

            # Flush video encoder
            packet = video_stream.encode(None)
            output.mux(packet)

            # Encode audio if it was successfully setup
            if audio_stream is not None and waveform is not None:
                frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format='fltp', layout=layout)
                frame.sample_rate = audio_sample_rate
                frame.pts = 0
                output.mux(audio_stream.encode(frame))

                # Flush audio encoder
                output.mux(audio_stream.encode(None))

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)]))

class XESaveVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [XESaveVideo]

async def comfy_entrypoint() -> XESaveVideoExtension:
    return XESaveVideoExtension()
