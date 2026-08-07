from __future__ import annotations

import os
import numpy as np
from PIL import Image as PILImage
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, Input, ui
from comfy_api.latest._ui import ImageSaveHelper, SavedImages, SavedResult, FolderType
from comfy.cli_args import args
import folder_paths


def _parse_image_options(format_input: dict | str, kwargs: dict) -> tuple[str, bool, int, int]:
    sources = []
    if isinstance(format_input, dict):
        sources.append(format_input)
    elif isinstance(format_input, str):
        sources.append({"format": format_input})
    if kwargs:
        sources.append(kwargs)

    format_str = "png"
    lossless = False
    quality = 90
    compression = 6

    for src in sources:
        if "format" in src and isinstance(src["format"], str):
            format_str = src["format"]
        
        if "lossless" in src and src["lossless"] is not None:
            lossless = bool(src["lossless"])
            
        if "quality" in src and src["quality"] is not None:
            try:
                quality = int(src["quality"])
            except (ValueError, TypeError):
                pass
                
        if "compression" in src and src["compression"] is not None:
            try:
                compression = int(src["compression"])
            except (ValueError, TypeError):
                pass

    return format_str, lossless, quality, compression


class SaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveImage",
            display_name="Save Image",
            category="xenodes/image",
            description="Saves the input images as PNG or WebP. WebP supports lossless and quality control.",
            inputs=[
                io.Image.Input("images", tooltip="The images to save."),
                io.String.Input(
                    "filename_prefix",
                    default="image/ComfyUI",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                ),
                io.DynamicCombo.Input(
                    "format",
                    options=[
                        io.DynamicCombo.Option(
                            "png",
                            [
                                io.Int.Input("compression", default=6, min=0, max=9, optional=True, tooltip="PNG compression level (0-9)."),
                            ]
                        ),
                        io.DynamicCombo.Option(
                            "webp",
                            [
                                io.Boolean.Input("lossless", default=False, optional=True, tooltip="Enables WebP lossless encoding."),
                                io.Int.Input("quality", default=90, min=0, max=100, optional=True, tooltip="WebP quality (0-100)."),
                            ]
                        ),
                    ],
                    tooltip="The image format to save as.",
                ),
            ],
            outputs=[io.Image.Output("images")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: Input.Image, filename_prefix: str, format: dict | str = "png", **kwargs) -> io.NodeOutput:
        format_str, lossless, quality, compression = _parse_image_options(format, kwargs)

        from ..utils.text import apply_text_replacements
        filename_prefix = apply_text_replacements(filename_prefix, cls.hidden.prompt, cls.hidden.extra_pnginfo)

        if images is None or len(images) == 0:
            return io.NodeOutput(images, ui=SavedImages([]))

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            images[0].shape[1],
            images[0].shape[0],
        )

        results = []

        for batch_number, image_tensor in enumerate(images):
            # Tensor -> PIL
            img = PILImage.fromarray(
                np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            )

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

            if format_str == "webp":
                file = f"{filename_with_batch_num}_{counter:05}_.webp"
                file_path = os.path.join(full_output_folder, file)

                exif_data = ImageSaveHelper._create_webp_metadata(img, cls)
                img.save(
                    file_path,
                    format="webp",
                    lossless=lossless,
                    quality=quality,
                    exif=exif_data,
                )
            else:  # png
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                file_path = os.path.join(full_output_folder, file)

                compress_level = max(0, min(9, compression))

                metadata = ImageSaveHelper._create_png_metadata(cls)
                img.save(
                    file_path,
                    pnginfo=metadata,
                    compress_level=compress_level,
                )

            results.append(SavedResult(file, subfolder, FolderType.output))
            counter += 1

        return io.NodeOutput(images, ui=SavedImages(results))


class SaveImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveImage]


async def comfy_entrypoint() -> SaveImageExtension:
    return SaveImageExtension()
