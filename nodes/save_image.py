from __future__ import annotations

import os
import numpy as np
from PIL import Image as PILImage
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, Input, ui
from comfy_api.latest._ui import ImageSaveHelper, SavedImages, SavedResult, FolderType
from comfy.cli_args import args
import folder_paths


class SaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveImage",
            display_name="Save Image",
            category="XENodes",
            description="Saves the input images as PNG or WebP. WebP supports lossless and quality control.",
            inputs=[
                io.Image.Input("images", tooltip="The images to save."),
                io.String.Input(
                    "filename_prefix",
                    default="image/ComfyUI",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                ),
                io.Combo.Input("format", options=["png", "webp"], default="webp", tooltip="The image format to save as."),
                io.Boolean.Input("lossless", default=False, tooltip="For WebP, enables lossless encoding. For PNG, this is ignored (always lossless)."),
                io.Int.Input("quality", default=90, min=0, max=100, tooltip="For WebP, this is 0-100 quality. For PNG, this is compression level 0-9 (default 6)."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: Input.Image, filename_prefix: str, format: str, lossless: bool, quality: int) -> io.NodeOutput:
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

            if format == "webp":
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

                # For PNG, quality input (0-9 via JS) is used as compress_level directly
                compress_level = max(0, min(9, quality))

                metadata = ImageSaveHelper._create_png_metadata(cls)
                img.save(
                    file_path,
                    pnginfo=metadata,
                    compress_level=compress_level,
                )

            results.append(SavedResult(file, subfolder, FolderType.output))
            counter += 1

        return io.NodeOutput(ui=SavedImages(results))


class SaveImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveImage]


async def comfy_entrypoint() -> SaveImageExtension:
    return SaveImageExtension()
