from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

# Custom output type that can connect to both INT and FLOAT inputs
class _NumberOutput(io.Output):
    io_type = "INT,FLOAT"

    def __init__(self, display_name: str | None = None, tooltip: str | None = None):
        super().__init__(None, display_name, tooltip)

    def get_io_type(self):
        return self.io_type

class SliderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.Slider",
            display_name="Slider",
            category="XENodes",
            inputs=[
                io.Float.Input("value", default=20.0, min=-4294967296.0, max=4294967296.0),
            ],
            outputs=[
                _NumberOutput(display_name=" "),
            ],
        )

    @classmethod
    def execute(cls, value: float) -> io.NodeOutput:
        # If the value is an integer (e.g., 20.0), convert it to int to ensure compatibility with INT nodes
        if value.is_integer():
            return io.NodeOutput(int(value))
        return io.NodeOutput(value)

class SliderExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SliderNode]

async def comfy_entrypoint() -> SliderExtension:
    return SliderExtension()
