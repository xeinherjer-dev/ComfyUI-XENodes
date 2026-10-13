import unittest
import asyncio
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
from custom_nodes.xenodes.nodes.fast_groups_muter_bypasser import (
    FastGroupsMuterBypasserNode,
    FastGroupsMuterBypasserExtension,
    comfy_entrypoint,
)


class TestFastGroupsMuterBypasser(unittest.TestCase):
    def test_define_schema(self):
        schema = FastGroupsMuterBypasserNode.define_schema()
        self.assertEqual(schema.node_id, "XENodes.FastGroupsMuterBypasser")
        self.assertEqual(len(schema.inputs), 0)
        self.assertEqual(len(schema.outputs), 0)
        self.assertFalse(schema.is_output_node)

    def test_execute(self):
        output = FastGroupsMuterBypasserNode.execute()
        # NodeOutput() without args has result None
        self.assertIsNone(output.result)

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, FastGroupsMuterBypasserExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(FastGroupsMuterBypasserNode, nodes)


if __name__ == "__main__":
    unittest.main()
