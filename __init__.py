from comfy_api.latest import ComfyExtension, io
from .nodes.multi_switch import comfy_entrypoint as multi_switch_entrypoint
from .nodes.slider import comfy_entrypoint as slider_entrypoint
from .nodes.slider2d import comfy_entrypoint as slider2d_entrypoint
from .nodes.save_video import comfy_entrypoint as save_video_entrypoint
from .nodes.save_image import comfy_entrypoint as save_image_entrypoint

WEB_DIRECTORY = "./web"

class XENodesExtension(ComfyExtension):
    def __init__(self, extensions: list[ComfyExtension]):
        self.extensions = extensions

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        nodes = []
        for ext in self.extensions:
            nodes.extend(await ext.get_node_list())
        return nodes

async def comfy_entrypoint() -> XENodesExtension:
    exts = [
        await multi_switch_entrypoint(),
        await slider_entrypoint(),
        await slider2d_entrypoint(),
        await save_video_entrypoint(),
        await save_image_entrypoint(),
    ]
    return XENodesExtension(exts)

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
