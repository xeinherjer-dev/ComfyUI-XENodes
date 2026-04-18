from __future__ import annotations
from typing_extensions import override
import json
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

class ComboSelectorNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.ComboSelector",
            display_name="Combo Selector",
            category="XENodes",
            inputs=[
                io.Int.Input("index", default=0, min=0),
                io.String.Input("hidden_list", default="[]"),
            ],
            outputs=[
                _io.Custom("COMBO").Output(display_name="COMBO"),
                io.String.Output(display_name="STRING"),
            ],
        )

    @classmethod
    def execute(cls, index: int, hidden_list: str) -> io.NodeOutput:
        try:
            items = json.loads(hidden_list)
        except json.JSONDecodeError:
            # Fallback if it's not a valid JSON string
            items = [item.strip() for item in hidden_list.split(",") if item.strip()]

        if not items:
            return io.NodeOutput(None, "")

        if not isinstance(items, list):
            items = [items]

        # Use modulo arithmetic to wrap around if index is out of bounds
        actual_index = index % len(items) if len(items) > 0 else 0
        selected_value = items[actual_index]

        return io.NodeOutput(selected_value, str(selected_value))

class ComboSelectorExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ComboSelectorNode]

async def comfy_entrypoint() -> ComboSelectorExtension:
    return ComboSelectorExtension()
