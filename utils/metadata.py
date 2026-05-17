from comfy.cli_args import args

def get_saved_metadata(node_cls) -> dict:
    """
    Extracts metadata (prompt, extra_pnginfo) from a ComfyUI node class.
    """
    saved_metadata = {}
    if not args.disable_metadata:
        if node_cls.hidden.extra_pnginfo is not None:
            saved_metadata.update(node_cls.hidden.extra_pnginfo)
        if node_cls.hidden.prompt is not None:
            saved_metadata["prompt"] = node_cls.hidden.prompt
    return saved_metadata
