from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

# Custom XE_MULTI_PIPE type for pipe connections
XEMultiPipe = _io.Custom("XE_MULTI_PIPE")

# Number of dynamic slots
MAX_SLOTS = 50


class MultiPipeInNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Use AnyType (io_type="*") so each slot independently accepts any type.
        autogrow_template = _io.Autogrow.TemplateNames(
            input=io.AnyType.Input("value"),
            names=[f"slot{i:02d}" for i in range(MAX_SLOTS)],
            min=0
        )

        return io.Schema(
            node_id="XENodes.MultiPipeIn",
            display_name="Multi-Pipe In",
            category="XENodes",
            description="Bundle multiple inputs into a single XE_MULTI_PIPE connection. Inputs auto-grow as you connect them.",
            inputs=[
                _io.Autogrow.Input("slots", template=autogrow_template),
            ],
            outputs=[
                XEMultiPipe.Output(display_name="pipe", tooltip="Bundled pipe output"),
            ],
        )

    @classmethod
    def execute(cls, slots: _io.Autogrow.Type) -> io.NodeOutput:
        result = {}

        if slots:
            for key, value in slots.items():
                if value is not None:
                    result[key] = value

        return io.NodeOutput(result)

    @classmethod
    def INPUT_TYPES(cls):
        # Workaround for ComfyUI V3 API
        res = super(MultiPipeInNode, cls).INPUT_TYPES()
        import copy
        res = copy.deepcopy(res)
        if "optional" not in res:
            res["optional"] = {}
        for i in range(MAX_SLOTS):
            res["optional"][f"slots.slot{i:02d}"] = ("*", {})
        return res


class MultiPipeOutNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        outputs = []
        for i in range(MAX_SLOTS):
            outputs.append(
                io.AnyType.Output(
                    display_name=f"slot{i:02d}",
                    tooltip=f"Output for slot{i:02d}",
                )
            )

        return io.Schema(
            node_id="XENodes.MultiPipeOut",
            display_name="Multi-Pipe Out",
            category="XENodes",
            description="Expand a XE_MULTI_PIPE connection into individual outputs. Outputs sync with the connected Multi-Pipe In node.",
            inputs=[
                XEMultiPipe.Input("pipe", tooltip="Pipe input to unpack"),
            ],
            outputs=outputs,
        )

    @classmethod
    def execute(cls, pipe: dict | None) -> io.NodeOutput:
        results = []

        if pipe and isinstance(pipe, dict):
            for i in range(MAX_SLOTS):
                key = f"slot{i:02d}"
                results.append(pipe.get(key))
        else:
            results.extend([None] * MAX_SLOTS)

        return io.NodeOutput(*results)


class MultiPipeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MultiPipeInNode, MultiPipeOutNode]


async def comfy_entrypoint() -> MultiPipeExtension:
    return MultiPipeExtension()
