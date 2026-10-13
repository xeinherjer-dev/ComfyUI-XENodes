import unittest
import asyncio
import os
import tempfile
import torch
from PIL import Image
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
import folder_paths
from custom_nodes.xenodes.nodes.load_image import (
    LoadImageFromFolder,
    LoadImageExtension,
    _resolve_path,
    _sort_folders,
    get_image_files,
    is_path_safe_and_allowed,
    comfy_entrypoint,
)


class TestLoadImage(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary subfolder in the output directory
        self.output_dir = folder_paths.get_output_directory()
        self.test_subfolder = os.path.join(self.output_dir, "xenodes_test_suite")
        os.makedirs(self.test_subfolder, exist_ok=True)
        self.test_img_path = os.path.join(self.test_subfolder, "sample.png")
        
        img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
        img.save(self.test_img_path)

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_subfolder):
            try:
                shutil.rmtree(self.test_subfolder)
            except Exception:
                pass

    def test_define_schema(self):
        schema = LoadImageFromFolder.define_schema()
        self.assertEqual(schema.node_id, "XENodes.LoadImageFromFolder")
        self.assertEqual(len(schema.outputs), 6)
        self.assertTrue(schema.is_output_node)

    def test_path_safety_and_resolution(self):
        # Output directory is allowed
        self.assertTrue(is_path_safe_and_allowed(self.output_dir))
        # Arbitrary root path is disallowed
        self.assertFalse(is_path_safe_and_allowed("/etc"))

        # _resolve_path with empty or output prefix
        res_output = _resolve_path("output")
        self.assertEqual(res_output, os.path.realpath(self.output_dir))

    def test_sort_folders(self):
        folders = ["output/2026-01-01", "output/2026-10-10", "output/misc"]
        sorted_newest = _sort_folders(folders, "newest_first")
        self.assertEqual(sorted_newest[0], "output/2026-10-10")
        self.assertEqual(sorted_newest[1], "output/2026-01-01")

        sorted_oldest = _sort_folders(folders, "oldest_first")
        self.assertEqual(sorted_oldest[0], "output/2026-01-01")

    def test_validate_and_fingerprint_inputs(self):
        # Valid path
        valid = LoadImageFromFolder.validate_inputs(path="output")
        self.assertTrue(valid)

        # Invalid path
        invalid = LoadImageFromFolder.validate_inputs(path="/etc/passwd")
        self.assertIsInstance(invalid, str)

        # Fingerprint
        fp = LoadImageFromFolder.fingerprint_inputs(path="output", index=0)
        self.assertIn("output", fp)

    def test_execute_load_image(self):
        # Execute loading our created test image from isolated subfolder
        out = LoadImageFromFolder.execute(path="output/xenodes_test_suite", index=0, sort_by="name")
        image_tensor = out.result[0]
        mask_tensor = out.result[1]
        filename = out.result[2]
        total_images = out.result[5]

        self.assertIsInstance(image_tensor, torch.Tensor)
        self.assertEqual(image_tensor.dim(), 4)  # [1, H, W, 3]
        self.assertEqual(image_tensor.shape[1:], (64, 64, 3))

        self.assertIsInstance(mask_tensor, torch.Tensor)
        self.assertEqual(mask_tensor.shape, (1, 64, 64))

        self.assertEqual(total_images, 1)

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, LoadImageExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(LoadImageFromFolder, nodes)


if __name__ == "__main__":
    unittest.main()
