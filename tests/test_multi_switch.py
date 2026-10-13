import unittest
import asyncio
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
from custom_nodes.xenodes.nodes.multi_switch import (
    MultiSwitchNode,
    MultiSwitchExtension,
    comfy_entrypoint,
)


class TestMultiSwitch(unittest.TestCase):
    def test_define_schema_and_input_types(self):
        schema = MultiSwitchNode.define_schema()
        self.assertEqual(schema.node_id, "XENodes.MultiSwitch")
        self.assertEqual(len(schema.inputs), 2)
        self.assertEqual(len(schema.outputs), 2)

        input_types = MultiSwitchNode.INPUT_TYPES()
        self.assertIn("optional", input_types)
        self.assertIn("inputs.input00", input_types["optional"])
        self.assertIn("inputs.input49", input_types["optional"])

    def test_check_lazy_status_skip(self):
        # select < 0 means skip
        status = MultiSwitchNode.check_lazy_status(-1, {})
        self.assertEqual(status, [])

    def test_check_lazy_status_not_evaluated(self):
        # Selected input is not yet provided in inputs dict
        status = MultiSwitchNode.check_lazy_status(2, {})
        self.assertEqual(status, ["inputs.input02"])

        # Selected input tuple has None value
        status_tuple = MultiSwitchNode.check_lazy_status(3, {"input03": (None, "node_3.output")})
        self.assertEqual(status_tuple, ["node_3.output"])

        # Selected input is None
        status_none = MultiSwitchNode.check_lazy_status(0, {"input00": None})
        self.assertEqual(status_none, ["inputs.input00"])

    def test_check_lazy_status_already_evaluated(self):
        status = MultiSwitchNode.check_lazy_status(1, {"input01": "already_evaluated"})
        self.assertEqual(status, [])

    def test_execute(self):
        # select < 0 returns None and the select index
        out_skip = MultiSwitchNode.execute(-1, {"input00": "ignored"})
        self.assertEqual(out_skip.result, (None, -1))

        # normal execution selects designated input
        out_val = MultiSwitchNode.execute(1, {"input00": "zero", "input01": "one"})
        self.assertEqual(out_val.result, ("one", 1))

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, MultiSwitchExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(MultiSwitchNode, nodes)


if __name__ == "__main__":
    unittest.main()
