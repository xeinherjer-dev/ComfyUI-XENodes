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
import torch
from custom_nodes.xenodes.nodes.save_image import (
    SaveImage,
    SaveImageExtension,
    _parse_image_options,
    _rgb_batch_to_full_range_yuv420,
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

        fmt, lossless, quality, compression, crf = _parse_image_options(
            {"format": "avif", "crf": 18}, {}
        )
        self.assertEqual(fmt, "avif")
        self.assertEqual(crf, 18.0)

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

    def test_execute_save_batch_webp_and_avif(self):
        y = torch.linspace(0.0, 1.0, 64).view(1, 64, 1, 1)
        x = torch.linspace(0.0, 1.0, 64).view(1, 1, 64, 1)
        batch_img = (0.5 * y + 0.5 * x).expand(3, 64, 64, 3).contiguous()

        for fmt in ("png", "webp", "avif"):
            prefix = f"tests/test_batch_{fmt}"
            out = SaveImage.execute(
                images=batch_img,
                filename_prefix=prefix,
                format=fmt,
            )
            self.assertEqual(len(out.ui.results), 3)
            out_dir = folder_paths.get_output_directory()
            matched = glob.glob(os.path.join(out_dir, "tests", f"test_batch_{fmt}*"))
            self.created_files.extend(matched)
            for p in matched:
                self.assertGreater(os.path.getsize(p), 0)

        # Verify odd resolution AVIF fallback does not crash
        odd_img = torch.full((1, 65, 67, 3), 0.5, dtype=torch.float32)
        out_odd = SaveImage.execute(
            images=odd_img,
            filename_prefix="tests/test_odd_avif",
            format="avif",
        )
        self.assertEqual(len(out_odd.ui.results), 1)
        out_dir = folder_paths.get_output_directory()
        self.created_files.extend(glob.glob(os.path.join(out_dir, "tests", "test_odd_avif*")))

        # Verify full-range YUV420 conversion accuracy on pure white and pure black
        bw = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], dtype=torch.float32)
        yuv10 = _rgb_batch_to_full_range_yuv420(bw, is_10bit=True, is_bt2020=True)
        self.assertEqual(int(yuv10[0, 0]), 0)
        self.assertEqual(int(yuv10[0, 2]), 1023)

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, SaveImageExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(SaveImage, nodes)


if __name__ == "__main__":
    unittest.main()
