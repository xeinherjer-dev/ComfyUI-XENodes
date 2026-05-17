from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class FastGroupsMuterBypasserNode(io.ComfyNode):
    """
    A virtual node that displays toggles for each group in the workflow,
    allowing quick muting or bypassing of all nodes within a group.

    Subgraph nodes inside a group are excluded from mute/bypass operations.
    The action mode (mute or bypass) can be toggled via properties.
    """

    @override
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.FastGroupsMuterBypasser",
            display_name="Fast Groups Muter & Bypasser",
            category="xenodes/utils",
            inputs=[],
            outputs=[],
            is_output_node=False,
        )

    @override
    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput()


class FastGroupsMuterBypasserExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [FastGroupsMuterBypasserNode]


async def comfy_entrypoint() -> FastGroupsMuterBypasserExtension:
    return FastGroupsMuterBypasserExtension()
