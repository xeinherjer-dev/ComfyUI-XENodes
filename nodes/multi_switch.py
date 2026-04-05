from __future__ import annotations
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

class MultiSwitchNode(io.ComfyNode):
    @staticmethod
    def _selected_key(select: int) -> str:
        return f"input{select:02d}"

    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("any")
        
        names = [f"input{i:02d}" for i in range(50)]
        autogrow_template = _io.Autogrow.TemplateNames(
            input=io.MatchType.Input("value", template=template, lazy=True),
            names=names,
            min=0
        )

        return io.Schema(
            node_id="XENodes.MultiSwitch",
            display_name="Multi-Switch",
            category="XENodes",
            inputs=[
                io.Int.Input("select", default=0, min=0, max=49),
                _io.Autogrow.Input("inputs", template=autogrow_template, lazy=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
                io.Int.Output(display_name="select"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, select: int, inputs: _io.Autogrow.Type) -> list[str]:
        selected_key = cls._selected_key(select)
        
        # If the expected key isn't in inputs yet, it still needs to be evaluated.
        if selected_key not in inputs:
            return [f"inputs.{selected_key}"]
            
        value = inputs.get(selected_key)
        if isinstance(value, tuple):
            actual_value, full_key = value
            if actual_value is None:
                return [full_key]
        elif value is None:
            return [f"inputs.{selected_key}"]
            
        return []

    @classmethod
    def execute(cls, select: int, inputs: _io.Autogrow.Type) -> io.NodeOutput:
        selected_key = cls._selected_key(select)
        selected_value = inputs.get(selected_key)
        return io.NodeOutput(selected_value, select)

    @classmethod
    def INPUT_TYPES(cls):
        # Workaround for ComfyUI V3 API: graph.py's get_input_info does not expand dynamic inputs.
        # It checks INPUT_TYPES() explicitly, causing dynamic lazy pins to be unconditionally executed.
        # We manually inject all 50 possible dynamic inputs here so the engine knows they are lazy.
        res = super(MultiSwitchNode, cls).INPUT_TYPES()
        import copy
        res = copy.deepcopy(res)
        if "optional" not in res:
            res["optional"] = {}
        for i in range(50):
            res["optional"][f"inputs.input{i:02d}"] = ("*", {"lazy": True})
        return res

class MultiSwitchExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MultiSwitchNode]

async def comfy_entrypoint() -> MultiSwitchExtension:
    return MultiSwitchExtension()
