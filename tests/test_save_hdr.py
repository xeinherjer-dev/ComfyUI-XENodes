import unittest
import asyncio
import os
import torch
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
from custom_nodes.xenodes.nodes.save_hdr_image import (
    SaveHDRImage,
    SaveHDRImageExtension,
    _parse_hdr_image_options,
    comfy_entrypoint as save_hdr_image_entrypoint,
)
from custom_nodes.xenodes.nodes.save_hdr_video import (
    SaveHDRVideo,
    SaveHDRVideoExtension,
    _parse_options as _parse_hdr_video_options,
    _create_ffmetadata_file,
    _prepare_audio_file,
    comfy_entrypoint as save_hdr_video_entrypoint,
)


class TestSaveHDRNodes(unittest.TestCase):
    def test_save_hdr_image_schema_and_options(self):
        schema = SaveHDRImage.define_schema()
        self.assertEqual(schema.node_id, "XENodes.SaveHDRImage")
        self.assertTrue(schema.is_deprecated)

        codec, crf = _parse_hdr_image_options("av1", crf_input=2.0, kwargs={})
        self.assertEqual(codec, "av1")
        self.assertEqual(crf, 2.0)

        codec, crf = _parse_hdr_image_options({"codec": "av1_nvenc", "crf": 4.0}, crf_input=2.0, kwargs={})
        self.assertEqual(codec, "av1_nvenc")
        self.assertEqual(crf, 4.0)

    def test_save_hdr_video_schema_and_helpers(self):
        schema = SaveHDRVideo.define_schema()
        self.assertEqual(schema.node_id, "XENodes.SaveHDRVideo")
        self.assertTrue(schema.is_output_node)

        # ffmetadata file generation
        meta_file = _create_ffmetadata_file({"title": "Test Title", "author": "User"})
        self.assertIsNotNone(meta_file)
        self.assertTrue(os.path.exists(meta_file))
        os.remove(meta_file)

        # audio prep helper
        dummy_audio = torch.zeros((2, 100), dtype=torch.float32)
        audio_file, rate, ch = _prepare_audio_file(dummy_audio, 44100)
        self.assertIsNotNone(audio_file)
        self.assertEqual(ch, 2)
        self.assertEqual(rate, 44100)
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)

    def test_extensions(self):
        ext_img = asyncio.run(save_hdr_image_entrypoint())
        self.assertIsInstance(ext_img, SaveHDRImageExtension)
        nodes_img = asyncio.run(ext_img.get_node_list())
        self.assertIn(SaveHDRImage, nodes_img)

        ext_vid = asyncio.run(save_hdr_video_entrypoint())
        self.assertIsInstance(ext_vid, SaveHDRVideoExtension)
        nodes_vid = asyncio.run(ext_vid.get_node_list())
        self.assertIn(SaveHDRVideo, nodes_vid)


if __name__ == "__main__":
    unittest.main()
