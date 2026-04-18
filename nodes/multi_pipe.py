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
        return io.Schema(
            node_id="XENodes.MultiPipeIn",
            display_name="Multi-Pipe In",
            category="XENodes",
            description="Bundle multiple inputs into a single XE_MULTI_PIPE connection. Inputs auto-grow as you connect them.",
            inputs=[
                XEMultiPipe.Input("pipe", tooltip="Base pipe to extend/override"),
            ],
            outputs=[
                XEMultiPipe.Output(display_name="pipe", tooltip="Bundled pipe output"),
            ],
        )

    @classmethod
    def execute(cls, pipe: dict | None = None, **kwargs) -> io.NodeOutput:
        result = {}

        if pipe and isinstance(pipe, dict):
            import copy
            result = copy.copy(pipe)

        # Catch kwargs correctly since comfyui provides them
        for key, value in kwargs.items():
            if value is not None and (key.startswith("slots.") or key.startswith("slots_")):
                real_key = key.split(".")[-1].split("_")[-1]
                result[real_key] = value

        if "slots" in kwargs and isinstance(kwargs["slots"], dict):
            for k, v in kwargs["slots"].items():
                if v is not None:
                    result[k] = v

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
        # Move 'pipe' from required to optional if it was placed there
        if "required" in res and "pipe" in res["required"]:
            res["optional"]["pipe"] = res["required"].pop("pipe")
        elif "pipe" not in res["optional"]:
            res["optional"]["pipe"] = ("XE_MULTI_PIPE", {"tooltip": "Base pipe to extend/override"})
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
