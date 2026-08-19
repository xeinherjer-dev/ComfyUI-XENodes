from __future__ import annotations

import os
import torch
import numpy as np
from PIL import Image as PILImage, ImageOps as PILImageOps
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
import comfy.model_management
import node_helpers


class LoadImageFromFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.LoadImageFromFolder",
            display_name="Load Image From Folder",
            category="xenodes/image",
            description="Loads a single image from a directory or direct file path without resizing, ideal for queues and vision LLMs.",
            inputs=[
                io.String.Input(
                    "path",
                    default="",
                    tooltip="Path to a directory containing images, or a direct image file path.",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    step=1,
                    control_after_generate=True,
                    tooltip="Image index. Supports auto-incrementing / randomizing via control_after_generate for batch queue runs.",
                ),
                io.Combo.Input(
                    "sort_by",
                    options=["name", "date_modified", "date_created", "random"],
                    default="name",
                    tooltip="Sorting method for files in the directory.",
                ),
                io.Boolean.Input(
                    "reverse",
                    default=False,
                    optional=True,
                    tooltip="Reverse the sort order.",
                ),
                io.Boolean.Input(
                    "subfolders",
                    default=False,
                    optional=True,
                    tooltip="Include images in subdirectories recursively.",
                ),
            ],
            outputs=[
                io.Image.Output("image", display_name="IMAGE", tooltip="Loaded image tensor [1, H, W, 3] without resizing."),
                io.Mask.Output("mask", display_name="MASK", tooltip="Alpha mask if present, otherwise zeros."),
                io.String.Output("filename", display_name="filename", tooltip="Filename of the loaded image."),
                io.String.Output("filepath", display_name="filepath", tooltip="Full filepath of the loaded image."),
                io.Int.Output("current_index", display_name="index", tooltip="Current index within the image list."),
                io.Int.Output("total_images", display_name="total_images", tooltip="Total count of images found."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, path: str, index: int, sort_by: str = "name", reverse: bool = False, subfolders: bool = False) -> str:
        if sort_by == "random":
            return str(os.urandom(8))
        return f"{path}_{index}_{sort_by}_{reverse}_{subfolders}"

    @classmethod
    def execute(cls, path: str, index: int = 0, sort_by: str = "name", reverse: bool = False, subfolders: bool = False) -> io.NodeOutput:
        path = path.strip().strip('"').strip("'")
        if not path:
            raise ValueError("Path is empty. Please provide a valid directory or image file path.")

        if os.path.isfile(path):
            files = [path]
        elif os.path.isdir(path):
            valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".tiff", ".tga"}
            files = []
            if subfolders:
                for root, _, filenames in os.walk(path):
                    for f in filenames:
                        if os.path.splitext(f)[1].lower() in valid_exts:
                            files.append(os.path.join(root, f))
            else:
                for f in os.listdir(path):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        files.append(os.path.join(path, f))

            if not files:
                raise FileNotFoundError(f"No valid images found in: {path}")

            if sort_by == "name":
                files.sort()
            elif sort_by == "date_modified":
                files.sort(key=lambda x: os.path.getmtime(x))
            elif sort_by == "date_created":
                files.sort(key=lambda x: os.path.getctime(x))
            elif sort_by == "random":
                # Deterministic random shuffle based on index if desired or seeded
                rng = np.random.default_rng(index)
                rng.shuffle(files)

            if reverse and sort_by != "random":
                files.reverse()
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

        total_images = len(files)
        actual_index = index % total_images
        target_file = files[actual_index]
        filename = os.path.basename(target_file)

        # Load image without resizing
        with PILImage.open(target_file) as img:
            img = node_helpers.pillow(PILImageOps.exif_transpose, img)

            # Mask extraction
            if "A" in img.getbands():
                mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - mask
                mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            else:
                mask_tensor = torch.zeros((1, img.height, img.width), dtype=torch.float32)

            # Convert to RGB image tensor
            rgb_img = img.convert("RGB")
            img_arr = np.array(rgb_img).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(img_arr).unsqueeze(0)  # [1, H, W, 3]

        device = comfy.model_management.intermediate_device()
        dtype = comfy.model_management.intermediate_dtype()
        image_tensor = image_tensor.to(device=device, dtype=dtype)
        mask_tensor = mask_tensor.to(device=device, dtype=dtype)

        return io.NodeOutput(
            image_tensor,
            mask_tensor,
            filename,
            target_file,
            actual_index,
            total_images,
        )


class LoadImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LoadImageFromFolder]


async def comfy_entrypoint() -> LoadImageExtension:
    return LoadImageExtension()
