from __future__ import annotations

import os
import math
import subprocess
import json
import tempfile
from fractions import Fraction
from typing_extensions import override

import torch
import numpy as np

from comfy_api.latest import ComfyExtension, io, Input, ui
from comfy.cli_args import args
import folder_paths

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
                io.Combo.Input("format", options=["mp4", "webm"], default="mp4", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=["av1", "av1_nvenc"], default="av1", tooltip="The codec to use for the video. HDR requires AV1."),
                io.Float.Input("crf", default=0.0, min=0.0, max=63.0, step=1.0, tooltip="Specific CRF value used for encoding (maps to CQ for NVENC). Set to 0 to use encoder defaults."),
                io.Float.Input("peak_nits", default=400.0, min=100.0, max=10000.0, step=1.0, tooltip="Peak brightness in nits. SDR white (100 nits) will be mapped to this target luminance in HDR."),
                io.Float.Input("itm_knee", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Inverse Tone Mapping (Soft-Knee) threshold. 0.0 starts expansion from black. 0.8 preserves SDR midtones and applies expansion to highlights."),
                io.Float.Input("itm_exponent", default=1.0, min=1.0, max=10.0, step=0.01, tooltip="Expansion curve exponent. 1.0 = Linear (punchy/bright), 2.0 = Quadratic (soft/natural), >2.0 = even softer transition."),
                io.Int.Input("loop_count", default=0, min=0, max=100, step=1, tooltip="Loop count. 0 = play once. For mp4/webm, this physically repeats the frames."),
                io.Boolean.Input("pingpong", default=False, tooltip="Pingpong animation (images only). Plays frames forward then backward."),
                io.Combo.Input("audio_codec", options=["aac", "opus", "flac"], default="aac", tooltip="The codec to use for the audio."),
                io.Combo.Input("audio_bitrate", options=["64k", "128k", "192k", "256k", "320k"], default="128k", tooltip="The bitrate to use for the audio codec (ignored if flac)."),
            ],
            outputs=None,
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix: str, format: str, codec: str, crf: float, loop_count: int, pingpong: bool, audio_codec: str, audio_bitrate: str, peak_nits: float, itm_knee: float, itm_exponent: float) -> io.NodeOutput:
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
        fps = float(frame_rate)

        # === Frame sequence generation ===
        images = components.images  # shape: (N, H, W, 3)
        num_images = images.shape[0]

        if pingpong and num_images > 2:
            frame_indices = list(range(num_images)) + list(range(num_images - 2, 0, -1))
        else:
            frame_indices = list(range(num_images))
            
        n_orig = len(frame_indices)
        total_plays = loop_count + 1

        # === Audio transformation ===
        audio_sample_rate = 44100
        waveform = None

        if getattr(components, 'audio', None) is not None:
            try:
                audio_sample_rate = int(components.audio['sample_rate'])
                raw_waveform = components.audio['waveform']

                raw_waveform = raw_waveform[0]
                samples_per_frame = audio_sample_rate / fps
                n_orig_samples = math.ceil(samples_per_frame * n_orig)
                total_samples_needed = math.ceil(samples_per_frame * (n_orig * total_plays))

                if raw_waveform.shape[-1] > n_orig_samples:
                    if raw_waveform.shape[-1] >= total_samples_needed:
                        waveform = raw_waveform[:, :total_samples_needed]
                    else:
                        repeats = math.ceil(total_samples_needed / raw_waveform.shape[-1])
                        waveform = torch.cat([raw_waveform] * repeats, dim=-1)[:, :total_samples_needed]
                else:
                    waveform = raw_waveform[:, :n_orig_samples]
                    if total_plays > 1:
                        waveform = torch.cat([waveform] * total_plays, dim=-1)
            except Exception as e:
                print(f"[XENodes] Warning: Failed to process audio stream: {e}")
                waveform = None

        temp_audio_path = None
        if waveform is not None:
            fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                import scipy.io.wavfile
                # waveform: [channels, samples] -> [samples, channels]
                audio_data = waveform.t().cpu().numpy()
                scipy.io.wavfile.write(temp_audio_path, audio_sample_rate, audio_data)
            except Exception as e:
                print(f"[XENodes] Warning: Failed to save temp audio: {e}")
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                temp_audio_path = None

        # Build ffmpeg command with 32-bit float input for maximum precision during SDR->HDR conversion
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-s", f"{width}x{height}", "-r", f"{fps:.06f}", "-i", "-"]
        input_count = 1
        
        if temp_audio_path:
            cmd += ["-i", temp_audio_path]
            input_count += 1

        # Add workflow metadata using a temp file to avoid Windows command line length limits (WinError 206)
        temp_meta_path = None
        if saved_metadata:
            try:
                fd, temp_meta_path = tempfile.mkstemp(suffix=".txt")
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(";FFMETADATA1\n")
                    for key, value in saved_metadata.items():
                        # ComfyUI frontend often looks for 'Workflow' and 'Prompt' (case-insensitive or specific)
                        # We write them as separate entries in the FFMETADATA format.
                        tag_name = key.capitalize() if key in ["workflow", "prompt"] else key
                        json_str = json.dumps(value)
                        # Escape special characters for FFMETADATA format
                        safe_json = json_str.replace('\\', '\\\\').replace('=', '\\=').replace(';', '\\;').replace('\n', ' ')
                        f.write(f"{tag_name}={safe_json}\n")
                
                # Insert metadata file as an additional input
                cmd += ["-i", temp_meta_path]
                metadata_input_index = input_count
                input_count += 1
            except Exception as e:
                print(f"[XENodes] Warning: Failed to prepare metadata file: {e}")
                if temp_meta_path and os.path.exists(temp_meta_path):
                    os.remove(temp_meta_path)
                temp_meta_path = None
        else:
            metadata_input_index = -1

        # Codec setup
        codec_config = {
            "av1": {"codec": "libsvtav1", "options": {}},
            "av1_nvenc": {"codec": "av1_nvenc", "options": {}}
        }
        config = codec_config.get(codec, codec_config["av1"])
        av_codec = config["codec"]
        cmd += ["-c:v", av_codec]

        for key, value in config.get("options", {}).items():
            cmd += [f"-{key}", str(value)]

        # HDR/10-bit setup
        cmd += ["-pix_fmt", "yuv420p10le"]

        if format == "mp4":
            cmd += ["-movflags", "use_metadata_tags"]

        trc = "smpte2084"
        cmd += ["-color_primaries", "bt2020", "-color_trc", trc, "-colorspace", "bt2020nc"]
        
        # Proper SDR to HDR conversion using zscale.
        # Since input is already linear float32 (where 1.0 = 100 nits), we just convert to PQ or HLG
        zscale_trc = "smpte2084"
        zscale_params = f"p=bt2020:t={zscale_trc}:m=bt2020nc:npl=100:dither=error_diffusion"
            
        cmd += ["-vf", f"setparams=color_primaries=bt709:color_trc=linear:colorspace=bt709,zscale={zscale_params}"]

        if "av1" in av_codec or "svtav1" in av_codec:
            if "nvenc" in av_codec and crf > 0:
                cmd += ["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"]
            elif crf > 0:
                cmd += ["-crf", str(int(crf))]

        # Audio setup
        if temp_audio_path:
            acodec = audio_codec
            if acodec == "opus":
                acodec = "libopus"
            cmd += ["-c:a", acodec]
            if audio_codec == "opus":
                cmd += ["-ar", "48000"] # Opus requires 48kHz
            if audio_codec != "flac":
                cmd += ["-b:a", audio_bitrate]

        # Map workflow metadata if it exists
        if metadata_input_index >= 0:
            cmd += ["-map_metadata", str(metadata_input_index)]

        cmd += [file_path]

        # Run ffmpeg
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            for _ in range(total_plays):
                for idx in frame_indices:
                    frame_tensor = images[idx]
                    
                    # Convert sRGB to Linear (IEC 61966-2-1 standard)
                    linear = torch.where(frame_tensor <= 0.04045, frame_tensor / 12.92, ((frame_tensor + 0.055) / 1.055) ** 2.4)
                    
                    # Apply Soft-Knee Inverse Tone Mapping (Power Curve)
                    # We calculate expansion based on luminance to preserve color (hue)
                    luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=linear.device)
                    luma = torch.sum(linear * luma_weights, dim=-1, keepdim=True)
                    
                    scale = peak_nits / 100.0
                    if itm_knee < 1.0:
                        # y = x + a * max(0, x - knee)^exponent
                        # a = (scale - 1.0) / (1.0 - knee)^exponent
                        a = (scale - 1.0) / ((1.0 - itm_knee) ** itm_exponent)
                        luma_diff = torch.clamp(luma - itm_knee, min=0.0)
                        luma_hdr = luma + a * (luma_diff ** itm_exponent)
                        
                        # Apply the same expansion ratio to all channels to preserve color
                        multiplier = (luma_hdr + 1e-6) / (luma + 1e-6)
                        linear = linear * multiplier

                    # Convert to GBR planar format (gbrpf32le)
                    gbr_planar = linear[..., [1, 2, 0]].permute(2, 0, 1).contiguous()
                    img_bytes = gbr_planar.cpu().numpy().astype(np.float32).tobytes()
                    
                    proc.stdin.write(img_bytes)
            
            # Close stdin and check for any immediate errors
            proc.stdin.close()
            return_code = proc.wait()
            if return_code != 0:
                stderr_output = proc.stderr.read().decode('utf-8')
                print(f"[XENodes] FFmpeg failed with return code {return_code}")
                if stderr_output:
                    print(f"[XENodes] FFmpeg error output:\n{stderr_output}")

        except Exception as e:
            # Capture stderr if available when a pipe error or other exception occurs
            stderr_output = ""
            try:
                if proc.stderr:
                    stderr_output = proc.stderr.read().decode('utf-8')
            except:
                pass
            
            print(f"[XENodes] Error encoding HDR video: {e}")
            if stderr_output:
                print(f"[XENodes] FFmpeg error output:\n{stderr_output}")
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            proc.wait()

        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
        
        if temp_meta_path and os.path.exists(temp_meta_path):
            try:
                os.remove(temp_meta_path)
            except:
                pass

        # Return only UI (preview)
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)]))

class SaveHDRVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveHDRVideo]

async def comfy_entrypoint() -> SaveHDRVideoExtension:
    return SaveHDRVideoExtension()
