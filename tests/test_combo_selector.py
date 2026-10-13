import unittest
import asyncio
try:
    from custom_nodes.xenodes.tests import conftest_base
except ImportError:
    try:
        from . import conftest_base
    except ImportError:
        import conftest_base
from custom_nodes.xenodes.nodes.combo_selector import ComboSelectorNode, ComboSelectorExtension, comfy_entrypoint


class TestComboSelector(unittest.TestCase):
    def test_define_schema(self):
        schema = ComboSelectorNode.define_schema()
        self.assertEqual(schema.node_id, "XENodes.ComboSelector")
        self.assertEqual(len(schema.inputs), 2)
        self.assertEqual(len(schema.outputs), 2)

    def test_execute_json_list(self):
        # Normal JSON array
        output = ComboSelectorNode.execute(index=1, hidden_list='["apple", "banana", "cherry"]')
        self.assertEqual(output.result, ("banana", "banana"))

    def test_execute_modulo_wrap(self):
        # Index exceeds length: 4 % 3 == 1
        output = ComboSelectorNode.execute(index=4, hidden_list='["apple", "banana", "cherry"]')
        self.assertEqual(output.result, ("banana", "banana"))

    def test_execute_comma_separated_fallback(self):
        # Fallback when hidden_list is not valid JSON
        output = ComboSelectorNode.execute(index=0, hidden_list="apple, banana, cherry")
        self.assertEqual(output.result, ("apple", "apple"))

    def test_execute_empty_list(self):
        # Empty list should return None and empty string
        output = ComboSelectorNode.execute(index=0, hidden_list="[]")
        self.assertEqual(output.result, (None, ""))
        output_empty_str = ComboSelectorNode.execute(index=0, hidden_list="")
        self.assertEqual(output_empty_str.result, (None, ""))

    def test_extension_entrypoint(self):
        ext = asyncio.run(comfy_entrypoint())
        self.assertIsInstance(ext, ComboSelectorExtension)
        nodes = asyncio.run(ext.get_node_list())
        self.assertIn(ComboSelectorNode, nodes)


if __name__ == "__main__":
    unittest.main()
