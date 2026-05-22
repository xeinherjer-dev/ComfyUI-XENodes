import math
import torch

def expand_audio_waveform(components, fps: float, n_orig: int, total_plays: int):
    """
    Expands an audio waveform to match the duration of a looping video.
    Returns: (waveform, audio_sample_rate, layout)
    """
    audio_sample_rate = 1
    waveform = None
    layout = 'stereo'

    if getattr(components, 'audio', None) is not None:
        try:
            audio_sample_rate = int(components.audio['sample_rate'])
            raw_waveform = components.audio['waveform']
            samples_per_frame = audio_sample_rate / fps

            raw_waveform = raw_waveform[0]  # shape: (channels, samples)
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

            layout = {1: 'mono', 2: 'stereo', 6: '5.1'}.get(waveform.shape[0], 'stereo')
        except Exception as e:
            print(f"[XENodes] Warning: Failed to process audio stream: {e}")
            waveform = None
            audio_sample_rate = 44100

    return waveform, audio_sample_rate, layout


def encode_audio_to_stream(output, audio_stream, waveform, audio_sample_rate: int, output_sample_rate: int, layout: str):
    """
    Safely encodes a numpy/torch waveform to a PyAV audio stream and muxes it.
    """
    import av
    # 1. Create the original audio frame
    orig_frame = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format='fltp', layout=layout)
    orig_frame.sample_rate = audio_sample_rate
    orig_frame.pts = 0

    # 2. Get the encoder's required frame size (fallback to None if 0 for variable frame size)
    encoder_frame_size = audio_stream.codec_context.frame_size
    if encoder_frame_size == 0:
        encoder_frame_size = None

    # 3. Always use AudioResampler to handle resampling and splitting into required frame sizes
    resampler = av.AudioResampler(
        format='fltp',
        layout=layout,
        rate=output_sample_rate,
        frame_size=encoder_frame_size
    )

    # 4. Process and encode the main data
    resampled_frames = resampler.resample(orig_frame)
    for f in resampled_frames:
        f.pts = None
        for packet in audio_stream.encode(f):
            output.mux(packet)

    # 5. Flush and collect the remaining data in the resampler buffer
    flush_frames = resampler.resample(None)
    for f in flush_frames:
        f.pts = None
        for packet in audio_stream.encode(f):
            output.mux(packet)

    # 6. Flush the audio encoder itself to output the final packet
    for packet in audio_stream.encode(None):
        output.mux(packet)
