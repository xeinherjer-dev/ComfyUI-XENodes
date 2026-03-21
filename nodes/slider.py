from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class XESliderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XESlider",
            display_name="Slider",
            category="XENodes",
            inputs=[
                io.Int.Input("value", default=20, min=-4294967296, max=4294967296),
            ],
            outputs=[
                io.Int.Output("value", display_name=" "),
            ],
        )

    @classmethod
    def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(value)

class XESliderExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [XESliderNode]

async def comfy_entrypoint() -> XESliderExtension:
    return XESliderExtension()
