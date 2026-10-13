import unittest
import asyncio
import os
import glob
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
    _iter_frame_byte_chunks,
    comfy_entrypoint,
)


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

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, SaveVideoExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(SaveVideo, nodes)


if __name__ == "__main__":
    unittest.main()
