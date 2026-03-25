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


class Slider2DNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.Slider2D",
            display_name="Slider 2D",
            category="XENodes",
            inputs=[
                io.Float.Input("X", default=512.0, min=-4294967296.0, max=4294967296.0),
                io.Float.Input("Y", default=512.0, min=-4294967296.0, max=4294967296.0),
            ],
            outputs=[
                _NumberOutput(display_name="X"),
                _NumberOutput(display_name="Y"),
            ],
        )

    @classmethod
    def execute(cls, X: float, Y: float) -> io.NodeOutput:
        out_x = int(X) if X.is_integer() else X
        out_y = int(Y) if Y.is_integer() else Y
        return io.NodeOutput(out_x, out_y)


class Slider2DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Slider2DNode]


async def comfy_entrypoint() -> Slider2DExtension:
    return Slider2DExtension()
