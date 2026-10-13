import unittest
import asyncio
import os
import glob
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
import folder_paths
from custom_nodes.xenodes.nodes.save_audio import (
    SaveAudio,
    SaveAudioExtension,
    _parse_audio_options,
    comfy_entrypoint,
)


class TestSaveAudio(unittest.TestCase):
    def setUp(self):
        self.created_files = []

    def tearDown(self):
        for path in self.created_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def test_define_schema(self):
        schema = SaveAudio.define_schema()
        self.assertEqual(schema.node_id, "XENodes.SaveAudio")
        self.assertTrue(schema.is_output_node)

    def test_parse_audio_options(self):
        # String format
        codec, bitrate = _parse_audio_options("mp3", {})
        self.assertEqual(codec, "mp3")

        # Dict format with bitrate
        codec_dict, bitrate_dict = _parse_audio_options(
            {"audio_codec": "opus", "audio_bitrate": "192k"}, {}
        )
        self.assertEqual(codec_dict, "opus")
        self.assertEqual(bitrate_dict, "192k")

    def test_execute_save_audio(self):
        audio_data = conftest_base.create_dummy_audio(channels=2, samples=4410, sample_rate=44100)
        prefix = "tests/test_audio_out"

        out = SaveAudio.execute(
            audio=audio_data,
            filename_prefix=prefix,
            audio_codec="mp3",
        )
        self.assertEqual(out.result[0], audio_data)
        self.assertTrue(hasattr(out, "ui"))
        self.assertGreater(len(out.ui.results), 0)

        # Track created file for teardown
        out_dir = folder_paths.get_output_directory()
        matched = glob.glob(os.path.join(out_dir, "tests", "test_audio_out*"))
        self.created_files.extend(matched)

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, SaveAudioExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(SaveAudio, nodes)


if __name__ == "__main__":
    unittest.main()
