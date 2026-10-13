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
from custom_nodes.xenodes.nodes.save_image import (
    SaveImage,
    SaveImageExtension,
    _parse_image_options,
    comfy_entrypoint,
)


class TestSaveImage(unittest.TestCase):
    def setUp(self):
        self.created_files = []
        # Ensure hidden prompt and extra_pnginfo are not None for test execution
        class DummyHidden:
            prompt = None
            extra_pnginfo = None
        SaveImage.hidden = DummyHidden()

    def tearDown(self):
        for path in self.created_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def test_define_schema(self):
        schema = SaveImage.define_schema()
        self.assertEqual(schema.node_id, "XENodes.SaveImage")
        self.assertTrue(schema.is_output_node)

    def test_parse_image_options(self):
        fmt, lossless, quality, compression, crf = _parse_image_options("png", {})
        self.assertEqual(fmt, "png")
        self.assertEqual(compression, 6)

        fmt, lossless, quality, compression, crf = _parse_image_options(
            {"format": "webp", "lossless": True, "quality": 95}, {}
        )
        self.assertEqual(fmt, "webp")
        self.assertTrue(lossless)
        self.assertEqual(quality, 95)

    def test_execute_save_png(self):
        img_tensor = conftest_base.create_dummy_image(batch=1, height=16, width=16)
        prefix = "tests/test_img_png"

        out = SaveImage.execute(
            images=img_tensor,
            filename_prefix=prefix,
            format="png",
        )
        self.assertTrue(torch_equal := (out.result[0] is img_tensor))
        self.assertTrue(hasattr(out, "ui"))
        self.assertGreater(len(out.ui.results), 0)

        out_dir = folder_paths.get_output_directory()
        matched = glob.glob(os.path.join(out_dir, "tests", "test_img_png*"))
        self.created_files.extend(matched)

    def test_execute_empty_images(self):
        out = SaveImage.execute(images=None, filename_prefix="tests/none")
        self.assertIsNone(out.result[0])
        self.assertEqual(len(out.ui.results), 0)

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, SaveImageExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(SaveImage, nodes)


if __name__ == "__main__":
    unittest.main()
