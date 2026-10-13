import unittest
import asyncio
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
from custom_nodes.xenodes.nodes.slider import (
    SliderNode,
    SliderExtension,
    comfy_entrypoint as slider_entrypoint,
)
from custom_nodes.xenodes.nodes.slider2d import (
    Slider2DNode,
    Slider2DExtension,
    comfy_entrypoint as slider2d_entrypoint,
)


class TestSliderNodes(unittest.TestCase):
    def test_slider_define_schema(self):
        schema = SliderNode.define_schema()
        self.assertEqual(schema.node_id, "XENodes.Slider")
        self.assertEqual(len(schema.inputs), 1)
        self.assertEqual(len(schema.outputs), 1)
        self.assertEqual(schema.outputs[0].get_io_type(), "INT,FLOAT")

    def test_slider_execute(self):
        # Float representing integer should be converted to int
        out_int = SliderNode.execute(20.0)
        self.assertEqual(out_int.result, (20,))
        self.assertIsInstance(out_int.result[0], int)

        # Non-integer float should remain float
        out_float = SliderNode.execute(20.5)
        self.assertEqual(out_float.result, (20.5,))
        self.assertIsInstance(out_float.result[0], float)

        # Negative and zero integer
        out_neg = SliderNode.execute(-5.0)
        self.assertEqual(out_neg.result, (-5,))
        self.assertIsInstance(out_neg.result[0], int)

    def test_slider2d_define_schema(self):
        schema = Slider2DNode.define_schema()
        self.assertEqual(schema.node_id, "XENodes.Slider2D")
        self.assertEqual(len(schema.inputs), 2)
        self.assertEqual(len(schema.outputs), 2)
        self.assertEqual(schema.outputs[0].get_io_type(), "INT,FLOAT")
        self.assertEqual(schema.outputs[1].get_io_type(), "INT,FLOAT")

    def test_slider2d_execute(self):
        # Both integer floats
        out = Slider2DNode.execute(512.0, 768.0)
        self.assertEqual(out.result, (512, 768))
        self.assertIsInstance(out.result[0], int)
        self.assertIsInstance(out.result[1], int)

        # Mixed integer and fractional floats
        out_mixed = Slider2DNode.execute(100.25, 200.0)
        self.assertEqual(out_mixed.result, (100.25, 200))
        self.assertIsInstance(out_mixed.result[0], float)
        self.assertIsInstance(out_mixed.result[1], int)

    def test_slider_extensions(self):
        ext = asyncio.run(slider_entrypoint())
        self.assertIsInstance(ext, SliderExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(SliderNode, nodes)

        ext2d = asyncio.run(slider2d_entrypoint())
        self.assertIsInstance(ext2d, Slider2DExtension)
        nodes2d = asyncio.run(ext2d.get_node_list())
        self.assertIn(Slider2DNode, nodes2d)


if __name__ == "__main__":
    unittest.main()
