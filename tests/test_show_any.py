import unittest
import asyncio
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
from custom_nodes.xenodes.nodes.show_any import (
    ShowAnyNode,
    ShowAnyExtension,
    comfy_entrypoint,
)


class TestShowAny(unittest.TestCase):
    def test_define_schema(self):
        schema = ShowAnyNode.define_schema()
        self.assertEqual(schema.node_id, "XENodes.ShowAny")
        self.assertEqual(len(schema.inputs), 1)
        self.assertEqual(len(schema.outputs), 1)
        self.assertTrue(schema.is_output_node)

    def test_execute_data_formatting(self):
        # None value
        out_none = ShowAnyNode.execute(value=None)
        self.assertIsNone(out_none.result[0])
        self.assertEqual(out_none.ui["text"], ["None"])

        # String value
        out_str = ShowAnyNode.execute(value="hello world")
        self.assertEqual(out_str.result[0], "hello world")
        self.assertEqual(out_str.ui["text"], ["hello world"])

        # Number value
        out_int = ShowAnyNode.execute(value=42)
        self.assertEqual(out_int.result[0], 42)
        self.assertEqual(out_int.ui["text"], ["42"])

        # Dict / JSON value
        out_dict = ShowAnyNode.execute(value={"key": "val"})
        self.assertEqual(out_dict.result[0], {"key": "val"})
        self.assertIn('"key": "val"', out_dict.ui["text"][0])

    def test_workflow_node_helpers(self):
        # Test find_target_node and mutate_workflow_data
        workflow_nodes = [
            {"id": "1", "type": "SomeNode"},
            {
                "id": "2",
                "type": "GroupNode",
                "nodes": [
                    {"id": "sub1", "type": "SubNode", "widgets_values": ["old_value"]}
                ],
            },
        ]

        # Simple top-level find
        node1 = ShowAnyNode.find_target_node(workflow_nodes, "1")
        self.assertIsNotNone(node1)
        self.assertEqual(node1["id"], "1")

        # Hierarchical find
        subnode = ShowAnyNode.find_target_node(workflow_nodes, "2:sub1")
        self.assertIsNotNone(subnode)
        self.assertEqual(subnode["id"], "sub1")

        # Get and mutate widget values
        val = ShowAnyNode.get_workflow_widget_value(workflow_nodes, "2:sub1")
        self.assertEqual(val, "old_value")

        mutated = ShowAnyNode.mutate_workflow_data(workflow_nodes, "2:sub1", "new_value")
        self.assertTrue(mutated)
        self.assertEqual(ShowAnyNode.get_workflow_widget_value(workflow_nodes, "2:sub1"), "new_value")

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, ShowAnyExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(ShowAnyNode, nodes)


if __name__ == "__main__":
    unittest.main()
