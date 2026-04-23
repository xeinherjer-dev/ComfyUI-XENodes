from __future__ import annotations

import os
import av
import json
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, Input, ui
from comfy.cli_args import args
import folder_paths

AUDIO_CODEC_CONFIG = {
    "mp3": {
        "av_codec": "libmp3lame",
        "ext": "mp3",
        "sample_rate_override": None,
        "supports_bitrate": True,
        "supported_bitrates": ["V0", "64k", "128k", "192k", "256k", "320k"],
        "default_bitrate": "V0",
    },
    "opus": {
        "av_codec": "libopus",
        "ext": "opus",
        "sample_rate_override": 48000,
        "supports_bitrate": True,
        "supported_bitrates": ["64k", "128k", "192k", "256k", "320k"],
        "default_bitrate": "128k",
    },
    "flac": {
        "av_codec": "flac",
        "ext": "flac",
        "sample_rate_override": None,
        "supports_bitrate": False,
        "supported_bitrates": [],
        "default_bitrate": None,
    }
}

class SaveAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveAudio",
            display_name="Save Audio",
            category="XENodes",
            description="Saves the input audio natively with standard codec support. MP3, Opus, FLAC.",
            inputs=[
                io.Audio.Input("audio", tooltip="The audio to save."),
                io.String.Input("filename_prefix", default="audio/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("audio_codec", options=list(AUDIO_CODEC_CONFIG.keys()), default="mp3", tooltip="The codec to use for the audio. Recommended: mp3 (for workflow support)."),
                io.Combo.Input("audio_bitrate", options=["V0", "64k", "128k", "192k", "256k", "320k"], default="192k", tooltip="The bitrate to use for the audio codec."),
            ],
            outputs=[io.Audio.Output("audio")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio: Input.Audio, filename_prefix: str, audio_codec: str, audio_bitrate: str) -> io.NodeOutput:
        config = AUDIO_CODEC_CONFIG.get(audio_codec, AUDIO_CODEC_CONFIG["mp3"])
        format_ext = config["ext"]

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory()
        )

        saved_metadata = {}
        if not args.disable_metadata:
            if cls.hidden.extra_pnginfo is not None:
                saved_metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                saved_metadata["prompt"] = cls.hidden.prompt

        file_name = f"{filename}_{counter:05}_.{format_ext}"
        file_path = os.path.join(full_output_folder, file_name)

        # Get audio variables
        audio_dict = audio if audio is not None else {}
        waveform = audio_dict.get('waveform')
        audio_sample_rate = int(audio_dict.get('sample_rate', 44100))
        
        # Make sure waveform has expected dims: (channels, samples)
        if waveform is not None:
            if waveform.dim() == 3:
                waveform = waveform[0]
            
            layout = {1: 'mono', 2: 'stereo', 6: '5.1'}.get(waveform.shape[0], 'stereo')
        else:
            layout = 'stereo'

        # Determine output sample rate
        output_sample_rate = audio_sample_rate
        if config.get("sample_rate_override") is not None:
            output_sample_rate = config["sample_rate_override"]

        av_audio_codec = config["av_codec"]

        with av.open(file_path, mode='w') as output:
            if saved_metadata:
                for key, value in saved_metadata.items():
                    if isinstance(value, str):
                        output.metadata[key] = value
                    else:
                        output.metadata[key] = json.dumps(value)

            audio_stream = None
            if waveform is not None:
                try:
                    audio_stream = output.add_stream(av_audio_codec, rate=output_sample_rate, layout=layout)

                    if config["supports_bitrate"]:
                        target_bitrate = audio_bitrate
                        if target_bitrate not in config["supported_bitrates"]:
                            target_bitrate = config["default_bitrate"]
                            
                        if target_bitrate == "V0":
                            audio_stream.codec_context.qscale = 1
                        else:
                            audio_stream.bit_rate = int(target_bitrate.replace("k", "000"))
                except Exception as e:
                    print(f"[XENodes] Warning: Failed to add audio stream: {e}")
                    audio_stream = None

            if audio_stream is not None and waveform is not None:
                orig_frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format='fltp', layout=layout)
                orig_frame.sample_rate = audio_sample_rate
                orig_frame.pts = 0

                # Actual resampling if needed
                if audio_sample_rate != output_sample_rate:
                    resampler = av.AudioResampler(format='fltp', layout=layout, rate=output_sample_rate)
                    resampled_frames = resampler.resample(orig_frame)
                    for f in resampled_frames:
                        f.pts = None
                        output.mux(audio_stream.encode(f))
                else:
                    output.mux(audio_stream.encode(orig_frame))

                # Flush audio encoder
                output.mux(audio_stream.encode(None))

        return io.NodeOutput(
            audio,
            ui=ui.SavedAudios([ui.SavedResult(file_name, subfolder, io.FolderType.output)])
        )

class SaveAudioExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveAudio]

async def comfy_entrypoint() -> SaveAudioExtension:
    return SaveAudioExtension()
