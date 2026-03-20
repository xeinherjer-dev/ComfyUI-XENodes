from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

class MultiSwitchNode(io.ComfyNode):
    @staticmethod
    def _selected_key(select: int, inputs: _io.Autogrow.Type) -> str | None:
        keys = list(inputs.keys())
        if 0 <= select < len(keys):
            return keys[select]
        return None

    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("any")
        autogrow_template = _io.Autogrow.TemplatePrefix(
            input=io.MatchType.Input("value", template=template, lazy=True),
            prefix="input",
            min=1,
            max=50
        )

        return io.Schema(
            node_id="MultiSwitch",
            display_name="Multi-Switch",
            category="logic",
            inputs=[
                io.Int.Input("select", default=0, min=0, max=49),
                _io.Autogrow.Input("inputs", template=autogrow_template),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
                io.Int.Output(display_name="select"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, select: int, inputs: _io.Autogrow.Type) -> list[str]:
        selected_key = cls._selected_key(select, inputs)
        if selected_key is not None and inputs[selected_key] is None:
            return [f"inputs.{selected_key}"]
        return []

    @classmethod
    def execute(cls, select: int, inputs: _io.Autogrow.Type) -> io.NodeOutput:
        selected_key = cls._selected_key(select, inputs)
        selected_value = inputs[selected_key] if selected_key is not None else None
        return io.NodeOutput(selected_value, select)

class MultiSwitchExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MultiSwitchNode]

async def comfy_entrypoint() -> MultiSwitchExtension:
    return MultiSwitchExtension()
