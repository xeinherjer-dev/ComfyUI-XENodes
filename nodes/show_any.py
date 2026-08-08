from __future__ import annotations
from typing_extensions import override
import json

from comfy_api.latest import ComfyExtension, io

class ShowAnyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("*", allowed_types=io.AnyType)

        return io.Schema(
            node_id="XENodes.ShowAny",
            display_name="Show Any",
            category="xenodes/utils",
            inputs=[
                io.MatchType.Input("value", template=template, display_name="any"),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="any"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        value = kwargs.get("value")

        # Stringify for UI display
        if value is None:
            text_str = "None"
        elif isinstance(value, str):
            text_str = value
        elif isinstance(value, (int, float, bool)):
            text_str = str(value)
        else:
            try:
                text_str = json.dumps(value, indent=2)
            except Exception:
                text_str = str(value)

        return io.NodeOutput(value, ui={"text": [text_str]})


class ShowAnyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ShowAnyNode]


async def comfy_entrypoint() -> ShowAnyExtension:
    return ShowAnyExtension()

