from comfy.cli_args import args

def get_saved_metadata(node_cls) -> dict:
    """
    Extracts metadata (prompt, extra_pnginfo) from a ComfyUI node class.
    """
    saved_metadata = {}
    hidden = getattr(node_cls, "hidden", None)
    if not args.disable_metadata and hidden is not None:
        if getattr(hidden, "extra_pnginfo", None) is not None:
            saved_metadata.update(hidden.extra_pnginfo)
        if getattr(hidden, "prompt", None) is not None:
            saved_metadata["prompt"] = hidden.prompt
    return saved_metadata
