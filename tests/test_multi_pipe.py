import unittest
import asyncio
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
from custom_nodes.xenodes.nodes.multi_pipe import (
    MultiPipeInNode,
    MultiPipeOutNode,
    MultiPipeExtension,
    MAX_SLOTS,
    comfy_entrypoint,
)


class TestMultiPipe(unittest.TestCase):
    def test_pipe_in_schema_and_input_types(self):
        schema = MultiPipeInNode.define_schema()
        self.assertEqual(schema.node_id, "XENodes.MultiPipeIn")
        input_types = MultiPipeInNode.INPUT_TYPES()
        self.assertIn("optional", input_types)
        # Check that dynamic slots are registered up to MAX_SLOTS
        self.assertIn("slots.slot00", input_types["optional"])
        self.assertIn(f"slots.slot{MAX_SLOTS - 1:02d}", input_types["optional"])
        self.assertIn("pipe", input_types["optional"])

    def test_pipe_in_execute(self):
        # Empty input
        out_empty = MultiPipeInNode.execute()
        self.assertEqual(out_empty.result, ({},))

        # Dot notation kwargs
        out_kwargs = MultiPipeInNode.execute(
            **{"slots.slot00": "image_data", "slots.slot01": 123}
        )
        self.assertEqual(out_kwargs.result[0], {"slot00": "image_data", "slot01": 123})

        # Underscore notation kwargs
        out_under = MultiPipeInNode.execute(
            **{"slots_slot02": "latent"}
        )
        self.assertEqual(out_under.result[0], {"slot02": "latent"})

        # Pipe extension/override
        base_pipe = {"slot00": "original", "slot01": "keep"}
        out_merged = MultiPipeInNode.execute(
            pipe=base_pipe,
            **{"slots.slot00": "overridden", "slots.slot02": "new"}
        )
        expected = {"slot00": "overridden", "slot01": "keep", "slot02": "new"}
        self.assertEqual(out_merged.result[0], expected)

        # Dictionary slots argument
        out_dict_slots = MultiPipeInNode.execute(
            slots={"slot03": "value3"}
        )
        self.assertEqual(out_dict_slots.result[0], {"slot03": "value3"})

    def test_pipe_out_schema(self):
        schema = MultiPipeOutNode.define_schema()
        self.assertEqual(schema.node_id, "XENodes.MultiPipeOut")
        self.assertEqual(len(schema.outputs), MAX_SLOTS)
        self.assertEqual(schema.outputs[0].display_name, "slot00")
        self.assertEqual(schema.outputs[MAX_SLOTS - 1].display_name, f"slot{MAX_SLOTS - 1:02d}")

    def test_pipe_out_execute(self):
        # Unpack populated pipe
        pipe_data = {"slot00": "first", "slot02": 42}
        out = MultiPipeOutNode.execute(pipe_data)
        self.assertEqual(len(out.result), MAX_SLOTS)
        self.assertEqual(out.result[0], "first")
        self.assertIsNone(out.result[1])
        self.assertEqual(out.result[2], 42)

        # Unpack None pipe
        out_none = MultiPipeOutNode.execute(None)
        self.assertEqual(len(out_none.result), MAX_SLOTS)
        self.assertTrue(all(val is None for val in out_none.result))

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, MultiPipeExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(MultiPipeInNode, nodes)
        self.assertIn(MultiPipeOutNode, nodes)


if __name__ == "__main__":
    unittest.main()
