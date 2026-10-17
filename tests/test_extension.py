import unittest
import asyncio
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
import custom_nodes.xenodes as xenodes


class TestXENodesExtensionIntegration(unittest.TestCase):
    def test_extension_entrypoint(self):
        ext = asyncio.run(xenodes.comfy_entrypoint())
        self.assertIsInstance(ext, xenodes.XENodesExtension)

        nodes = asyncio.run(ext.get_node_list())
        # Verify node count and essential node IDs
        self.assertGreaterEqual(len(nodes), 15)

        node_ids = {node.define_schema().node_id for node in nodes}
        expected_ids = {
            "XENodes.MultiSwitch",
            "XENodes.Slider",
            "XENodes.Slider2D",
            "XENodes.SaveVideo",
            "XENodes.SaveAudio",
            "XENodes.SaveImage",
            "XENodes.ComboSelector",
            "XENodes.ShowAny",
            "XENodes.MultiPipeIn",
            "XENodes.MultiPipeOut",
            "XENodes.SaveHDRVideo",
            "XENodes.SaveHDRImage",
            "XENodes.SDRtoHDR",
            "XENodes.LoadSDRtoHDRModel",
            "XENodes.SDRtoHDRUsingModel",
            "XENodes.FastGroupsMuterBypasser",
            "XENodes.LoadImageFromFolder",
        }
        for eid in expected_ids:
            self.assertIn(eid, node_ids, f"Expected node_id '{eid}' not found in registered nodes.")

    def test_web_directory(self):
        self.assertEqual(xenodes.WEB_DIRECTORY, "./web")


if __name__ == "__main__":
    unittest.main()
