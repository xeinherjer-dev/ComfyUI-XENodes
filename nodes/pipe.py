from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

# Custom XEPIPE type for pipe connections
XEPipe = _io.Custom("XEPIPE")

# Number of dynamic slots
MAX_SLOTS = 50


class PipeInNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Use AnyType (io_type="*") so each slot independently accepts any type.
        # MatchType.Template shares type resolution across all slots using the same
        # template — connecting IMAGE to one slot would restrict all others to IMAGE.
        autogrow_template = _io.Autogrow.TemplateNames(
            input=io.AnyType.Input("value"),
            names=[f"slot{i:02d}" for i in range(MAX_SLOTS)],
            min=0
        )

        return io.Schema(
            node_id="XENodes.PipeIn",
            display_name="Pipe In",
            category="XENodes/Pipe",
            description="Bundle multiple inputs into a single XEPIPE connection. Inputs auto-grow as you connect them.",
            inputs=[
                _io.Autogrow.Input("slots", template=autogrow_template),
            ],
            outputs=[
                XEPipe.Output(display_name="pipe", tooltip="Bundled pipe output"),
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
        # Workaround for ComfyUI V3 API: same pattern as multi_switch
        res = super(PipeInNode, cls).INPUT_TYPES()
        import copy
        res = copy.deepcopy(res)
        if "optional" not in res:
            res["optional"] = {}
        for i in range(MAX_SLOTS):
            res["optional"][f"slots.slot{i:02d}"] = ("*", {})
        return res


class PipeOutNode(io.ComfyNode):
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
            node_id="XENodes.PipeOut",
            display_name="Pipe Out",
            category="XENodes/Pipe",
            description="Expand a XEPIPE connection into individual outputs. Outputs sync with the connected Pipe In node.",
            inputs=[
                XEPipe.Input("pipe", tooltip="Pipe input to unpack"),
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


class PipeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PipeInNode, PipeOutNode]


async def comfy_entrypoint() -> PipeExtension:
    return PipeExtension()
