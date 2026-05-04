from __future__ import annotations

import os
import math
import subprocess
import json
import tempfile
from fractions import Fraction
from typing_extensions import override

import torch

from comfy_api.latest import ComfyExtension, io, Input, ui
from comfy.cli_args import args
import folder_paths

class SaveHDRVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveHDRVideo",
            display_name="Save HDR Video",
            category="XENodes",
            description="Saves the input video natively as HDR using ffmpeg, without AI processing models.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."),
                io.Combo.Input("format", options=["mp4", "webm"], default="mp4", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=["h265", "av1", "hevc_nvenc", "av1_nvenc"], default="h265", tooltip="The codec to use for the video. HDR requires H.265 or AV1."),
                io.Float.Input("crf", default=0.0, min=0.0, max=63.0, step=1.0, tooltip="Specific CRF value used for encoding (maps to CQ for NVENC). Set to 0 to use encoder defaults."),
                io.Int.Input("loop_count", default=0, min=0, max=100, step=1, tooltip="Loop count. 0 = play once. For mp4/webm, this physically repeats the frames."),
                io.Boolean.Input("pingpong", default=False, tooltip="Pingpong animation (images only). Plays frames forward then backward."),
                io.Combo.Input("audio_codec", options=["aac", "opus", "flac"], default="aac", tooltip="The codec to use for the audio."),
                io.Combo.Input("audio_bitrate", options=["64k", "128k", "192k", "256k", "320k"], default="128k", tooltip="The bitrate to use for the audio codec (ignored if flac)."),
                io.Float.Input("peak_nits", default=400.0, min=100.0, max=10000.0, step=10.0, tooltip="Peak brightness in nits. SDR white (100 nits) will be mapped to this target luminance in HDR."),
            ],
            outputs=[io.Video.Output("video")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix: str, format: str, codec: str, crf: float, loop_count: int, pingpong: bool, audio_codec: str, audio_bitrate: str, peak_nits: float) -> io.NodeOutput:
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

        # Build ffmpeg command
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:.06f}", "-i", "-"]
        input_count = 1
        
        if temp_audio_path:
            cmd += ["-i", temp_audio_path]
            input_count += 1

        # Add workflow metadata using a temp file to avoid Windows command line length limits (WinError 206)
        temp_meta_path = None
        if saved_metadata:
            try:
                metadata_json = json.dumps(saved_metadata)
                fd, temp_meta_path = tempfile.mkstemp(suffix=".txt")
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(";FFMETADATA1\n")
                    # Escape special characters for FFMETADATA format
                    safe_json = metadata_json.replace('\\', '\\\\').replace('=', '\\=').replace(';', '\\;').replace('\n', ' ')
                    f.write(f"comment={safe_json}\n")
                
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
        av_codec = codec
        if codec == "h264":
            av_codec = "libx264"
        elif codec == "h265":
            av_codec = "libx265"
        elif codec == "av1":
            av_codec = "libsvtav1"
            
        cmd += ["-c:v", av_codec]

        # HDR/10-bit setup
        cmd += ["-pix_fmt", "yuv420p10le"]

        cmd += ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"]
        
        # Proper SDR to HDR conversion using zscale (matching user reference for quality).
        # We use a 2-step process with gbrpf32le intermediate for maximum precision.
        npl = peak_nits
        cmd += ["-vf", f"setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709,zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt2020:t=smpte2084:m=bt2020nc:npl={npl}"]

        # Add HDR metadata and CRF
        mastering = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
        max_cll = f"{int(peak_nits)},{max(int(peak_nits * 0.4), 1)}"

        if codec == "h265":
            cmd += ["-tag:v", "hvc1"]
            if crf > 0:
                cmd += ["-crf", str(int(crf))]
            cmd += ["-x265-params", f"hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display={mastering}:max-cll={max_cll}"]
        elif codec == "hevc_nvenc":
            cmd += ["-tag:v", "hvc1", "-preset", "p6", "-tune", "hq", "-profile:v", "main10"]
            if crf > 0:
                cmd += ["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"]
            cmd += ["-bsf:v", "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9"]
        elif "h264" in codec:
            if "nvenc" in codec and crf > 0:
                cmd += ["-rc", "vbr", "-cq", str(int(crf)), "-b:v", "0"]
            elif crf > 0:
                cmd += ["-crf", str(int(crf))]
        elif "av1" in codec:
            if "nvenc" in codec and crf > 0:
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
                    img = (frame_tensor * 255.0).clamp(0, 255).byte().cpu().numpy()
                    proc.stdin.write(img.tobytes())
            
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

        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)]))

class SaveHDRVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveHDRVideo]

async def comfy_entrypoint() -> SaveHDRVideoExtension:
    return SaveHDRVideoExtension()
