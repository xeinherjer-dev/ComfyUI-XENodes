import unittest
import asyncio
import math
import os
import glob
import tempfile
import time
from fractions import Fraction
from unittest.mock import patch
import torch
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
from custom_nodes.xenodes.nodes.save_video import (
    SaveVideo,
    SaveVideoExtension,
    _parse_options,
    _can_passthrough_video,
    _compute_hdr10_light_levels,
    _slice_or_index_frames,
    _iter_frame_byte_chunks,
    _iter_yuv420_byte_chunks,
    comfy_entrypoint,
)
from custom_nodes.xenodes.utils.color import pq_to_linear


class TestSaveVideo(unittest.TestCase):
    def test_define_schema(self):
        schema = SaveVideo.define_schema()
        self.assertEqual(schema.node_id, "XENodes.SaveVideo")
        self.assertTrue(schema.is_output_node)

    def test_parse_options(self):
        fmt, codec, crf, a_codec, a_bitrate = _parse_options("mp4", {})
        self.assertEqual(fmt, "mp4")
        self.assertEqual(codec, "h264")
        self.assertEqual(a_codec, "aac")

        fmt, codec, crf, a_codec, a_bitrate = _parse_options(
            {"format": "webm", "codec": "vp9", "crf": 30.0, "audio_codec": "opus"}, {}
        )
        self.assertEqual(fmt, "webm")
        self.assertEqual(codec, "vp9")
        self.assertEqual(crf, 30.0)
        self.assertEqual(a_codec, "opus")

        # Flat keys from dynamic combos
        fmt, codec, crf, a_codec, a_bitrate = _parse_options(
            {}, {"format": "mp4", "format.codec.crf": 26.0, "format.audio_codec.audio_bitrate": "192k"}
        )
        self.assertEqual(crf, 26.0)
        self.assertEqual(a_bitrate, "192k")

    def test_compute_hdr10_light_levels(self):
        from custom_nodes.xenodes.nodes.save_video import _compute_hdr10_light_levels
        # Create dummy PQ frames (shape B, H, W, 3)
        dummy = torch.full((2, 8, 8, 3), 0.5, dtype=torch.float32)
        max_cll, max_fall = _compute_hdr10_light_levels(dummy, [0, 1])
        self.assertIsInstance(max_cll, int)
        self.assertIsInstance(max_fall, int)
        self.assertGreater(max_cll, 0)
        self.assertLessEqual(max_fall, max_cll)

    def test_can_passthrough_video_guard(self):
        # When loop_count != 0 or pingpong is True, passthrough must return False
        can_pass, _, _, _, _ = _can_passthrough_video(
            video=None,
            target_format="mp4",
            target_audio_codec="aac",
            target_audio_bitrate=None,
            loop_count=1,
            pingpong=False,
        )
        self.assertFalse(can_pass)

    def test_iter_raw_frame_byte_chunks(self):
        # 4 frames of 16x16 image
        frames = conftest_base.create_dummy_image(batch=4, height=16, width=16)
        chunks = list(_iter_frame_byte_chunks(frames, frame_indices=[0, 1, 2, 3], total_plays=1))
        self.assertGreater(len(chunks), 0)
        # Expected bytes for 4 frames: 4 * 16 * 16 * 3
        total_bytes = sum(len(c) for c in chunks)
        self.assertEqual(total_bytes, 4 * 16 * 16 * 3)

    def test_iter_yuv420_byte_chunks(self):
        import numpy as np
        # 2 frames of 16x16: frame 0 = pure black (0.0), frame 1 = pure white (1.0)
        black = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
        white = torch.ones((1, 16, 16, 3), dtype=torch.float32)
        frames = torch.cat([black, white], dim=0)

        # 8-bit SDR yuv420p: 1.5 bytes per pixel
        chunks_8 = list(_iter_yuv420_byte_chunks(frames, [0, 1], total_plays=1, pix_fmt="yuv420p", is_hdr=False))
        raw_8 = b"".join(bytes(c) for c in chunks_8)
        self.assertEqual(len(raw_8), 2 * (16 * 16 * 3) // 2)
        arr_8 = np.frombuffer(raw_8, dtype=np.uint8).reshape(2, -1)
        self.assertEqual(int(arr_8[0, 0]), 16)    # black Y = 16
        self.assertEqual(int(arr_8[0, 256]), 128) # black U = 128
        self.assertEqual(int(arr_8[1, 0]), 235)   # white Y = 235
        self.assertEqual(int(arr_8[1, 256]), 128) # white U = 128

        # 10-bit HDR yuv420p10le: 3 bytes per pixel
        chunks_10 = list(_iter_yuv420_byte_chunks(frames, [0, 1], total_plays=1, pix_fmt="yuv420p10le", is_hdr=True))
        raw_10 = b"".join(bytes(c) for c in chunks_10)
        self.assertEqual(len(raw_10), 2 * 16 * 16 * 3)
        arr_10 = np.frombuffer(raw_10, dtype=np.uint16).reshape(2, -1)
        self.assertEqual(int(arr_10[0, 0]), 64)    # black Y = 64
        self.assertEqual(int(arr_10[0, 256]), 512) # black U = 512
        self.assertEqual(int(arr_10[1, 0]), 940)   # white Y = 940 (strict ITU-R BT.2020, not 943)
        self.assertEqual(int(arr_10[1, 256]), 512) # white U = 512

        # 10-bit HDR p010le: 3 bytes per pixel, shifted << 6 (* 64)
        chunks_p010 = list(_iter_yuv420_byte_chunks(frames, [0, 1], total_plays=1, pix_fmt="p010le", is_hdr=True))
        raw_p010 = b"".join(bytes(c) for c in chunks_p010)
        self.assertEqual(len(raw_p010), 2 * 16 * 16 * 3)
        arr_p010 = np.frombuffer(raw_p010, dtype=np.uint16).reshape(2, -1)
        self.assertEqual(int(arr_p010[0, 0]), 64 * 64)
        self.assertEqual(int(arr_p010[1, 0]), 940 * 64)

    def test_zero_copy_slice_and_loop_cache_performance(self):
        """Verifies zero-copy slicing on contiguous indices and loop chunk caching."""
        frames = conftest_base.create_dummy_image(batch=8, height=64, width=64)

        # 1. Contiguous indices must return a view sharing storage with `frames`
        view = _slice_or_index_frames(frames, [2, 3, 4, 5])
        self.assertEqual(view.shape, (4, 64, 64, 3))
        self.assertEqual(view.untyped_storage().data_ptr(), frames.untyped_storage().data_ptr())

        # 2. Non-contiguous indices (e.g. pingpong reversal) still return valid tensor
        non_contig = _slice_or_index_frames(frames, [0, 1, 2, 1])
        self.assertEqual(non_contig.shape, (4, 64, 64, 3))

        # 3. Chunks must be zero-copy memoryviews, and loop caching must reuse converted buffers
        chunks_loop = list(
            _iter_yuv420_byte_chunks(frames, list(range(8)), total_plays=4, pix_fmt="yuv420p10le", is_hdr=True)
        )
        self.assertEqual(len(chunks_loop), 4)  # 8 frames fit in 1 chunk, repeated 4 times
        self.assertIsInstance(chunks_loop[0], memoryview)
        # Subsequent plays must reuse the exact same memoryview object (zero re-computation)
        self.assertIs(chunks_loop[0], chunks_loop[1])
        self.assertIs(chunks_loop[0], chunks_loop[2])
        self.assertIs(chunks_loop[0], chunks_loop[3])

    def test_hdr10_light_levels_accuracy_and_speed(self):
        """Verifies that optimized MaxCLL/MaxFALL matches naive 3-channel EOTF exactly and runs fast."""
        torch.manual_seed(123)
        frames = torch.rand((16, 256, 256, 3), dtype=torch.float32)
        indices = list(range(16))

        # Naive reference computation (full 3-channel pq_to_linear on CPU)
        nits_ref = pq_to_linear(frames, nits_scale=1.0).amax(dim=-1)
        ref_cll = int(min(10000, math.ceil(float(nits_ref.amax()))))
        ref_fall = int(min(ref_cll, math.ceil(float(nits_ref.mean(dim=(1, 2)).amax()))))

        t0 = time.perf_counter()
        cll, fall = _compute_hdr10_light_levels(frames, indices)
        elapsed = time.perf_counter() - t0

        self.assertEqual(cll, ref_cll)
        self.assertEqual(fall, ref_fall)
        self.assertLess(elapsed, 1.0, f"_compute_hdr10_light_levels took too long: {elapsed:.3f}s")

    def test_save_video_end_to_end_quality_and_speed(self):
        """Verifies end-to-end SaveVideo execution, decoded PSNR quality, and odd-resolution fallback."""
        import numpy as np
        import av
        from comfy_api.latest import InputImpl, Types

        # Create smooth deterministic gradient video (12 frames, 128x128 even + 127x127 odd)
        y = torch.linspace(0.0, 1.0, 128, dtype=torch.float32).view(1, 128, 1, 1)
        x = torch.linspace(0.0, 1.0, 128, dtype=torch.float32).view(1, 1, 128, 1)
        frames = torch.cat([
            x.expand(12, 128, 128, 1),
            y.expand(12, 128, 128, 1),
            ((x + y) * 0.5).expand(12, 128, 128, 1),
        ], dim=-1).contiguous()

        class DummyHidden:
            prompt = None
            extra_pnginfo = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("folder_paths.get_output_directory", return_value=tmpdir):
                SaveVideo.hidden = DummyHidden()

                # 1. Direct YUV420 10-bit HDR PQ encode (h265)
                vid_hdr = InputImpl.VideoFromComponents(
                    Types.VideoComponents(images=frames, frame_rate=Fraction(24, 1)),
                    bit_depth=10,
                    color_space="HDR PQ",
                )
                t0 = time.perf_counter()
                SaveVideo.execute(
                    video=vid_hdr,
                    filename_prefix="test_hdr",
                    format={"format": "mp4", "codec": "h265", "crf": 18.0},
                )
                elapsed_hdr = time.perf_counter() - t0
                hdr_files = glob.glob(os.path.join(tmpdir, "test_hdr*.mp4"))
                self.assertEqual(len(hdr_files), 1)
                self.assertLess(elapsed_hdr, 5.0)

                # Decode and verify PSNR >= 42 dB against input frames
                with av.open(hdr_files[0]) as container:
                    v_stream = container.streams.video[0]
                    self.assertEqual(v_stream.width, 128)
                    self.assertEqual(v_stream.height, 128)
                    decoded = [f.to_ndarray(format="rgb24").astype(np.float32) / 255.0 for f in container.decode(video=0)]
                self.assertEqual(len(decoded), 12)
                mse = float(np.mean((frames.numpy() - np.stack(decoded, axis=0)) ** 2))
                psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
                self.assertGreaterEqual(psnr, 40.0, f"Decoded video PSNR too low: {psnr:.2f} dB")

                # 2. Odd-dimension (126x126 vs 127x126) fallback to RGB pipe + scale filter
                frames_odd = frames[:, :126, :127, :].contiguous()
                vid_odd = InputImpl.VideoFromComponents(
                    Types.VideoComponents(images=frames_odd, frame_rate=Fraction(24, 1)),
                )
                SaveVideo.execute(
                    video=vid_odd,
                    filename_prefix="test_odd",
                    format={"format": "mp4", "codec": "av1", "crf": 35.0},
                )
                odd_files = glob.glob(os.path.join(tmpdir, "test_odd*.mp4"))
                self.assertEqual(len(odd_files), 1)

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, SaveVideoExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(SaveVideo, nodes)


if __name__ == "__main__":
    unittest.main()
