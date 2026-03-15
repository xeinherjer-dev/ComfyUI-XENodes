from __future__ import annotations
from typing import Any
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

class MultiSwitchNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Template for matching any input type
        template = io.MatchType.Template("any")
        
        # Autogrow template for dynamic inputs
        autogrow_template = _io.Autogrow.TemplatePrefix(
            input=io.MatchType.Input("value", template=template, lazy=True),
            prefix="input_",
            min=1,
            max=50
        )
        
        return io.Schema(
            node_id="MultiSwitch",
            display_name="Multi-Switch",
            category="logic",
            inputs=[
                # Selector index
                io.Int.Input("select", default=0, min=0, max=49),
                # Dynamic inputs
                _io.Autogrow.Input("inputs", template=autogrow_template)
            ],
            outputs=[
                # Main selected output
                io.MatchType.Output(template=template, display_name="output"),
                # Added select_index output as requested
                io.Int.Output(display_name="select_index"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, select: int, inputs: _io.Autogrow.Type) -> list[str]:
        keys = list(inputs.keys())
        if 0 <= select < len(keys):
            selected_key = keys[select]
            if inputs[selected_key] is None:
                return [f"inputs.{selected_key}"]
        return []

    @classmethod
    def execute(cls, select: int, inputs: _io.Autogrow.Type) -> io.NodeOutput:
        keys = list(inputs.keys())
        selected_value = None
        if 0 <= select < len(keys):
            selected_key = keys[select]
            selected_value = inputs[selected_key]
        
        # Return both the selected value and the selected index
        return io.NodeOutput(selected_value, select)

class MultiSwitchExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MultiSwitchNode]

async def comfy_entrypoint() -> MultiSwitchExtension:
    return MultiSwitchExtension()
