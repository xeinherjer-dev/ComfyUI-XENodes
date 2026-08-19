from __future__ import annotations

import os
import re
import torch
import numpy as np
from PIL import Image as PILImage, ImageOps as PILImageOps
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
from comfy_api.latest._ui import SavedImages, SavedResult, FolderType
import comfy.model_management
import folder_paths
import node_helpers


def _resolve_path(raw_path: str) -> str:
    path = raw_path.strip().strip('"').strip("'")
    if not path:
        return ""

    # If the path exists as-is, return it
    if os.path.exists(path):
        return path

    # Auto-convert Windows path to WSL path (e.g., C:\Users\... -> /mnt/c/Users/...)
    if re.match(r"^[a-zA-Z]:[\\/]", path):
        drive = path[0].lower()
        wsl_path = f"/mnt/{drive}/" + path[2:].replace("\\", "/").lstrip("/")
        if os.path.exists(wsl_path):
            return wsl_path

    # Standardize backslashes for unix environments if applicable
    norm_path = path.replace("\\", "/")
    if os.path.exists(norm_path):
        return norm_path

    return path


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
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, path: str, index: int, sort_by: str = "name", reverse: bool = False, subfolders: bool = False) -> str:
        if sort_by == "random":
            return str(os.urandom(8))
        return f"{path}_{index}_{sort_by}_{reverse}_{subfolders}"

    @classmethod
    def execute(cls, path: str, index: int = 0, sort_by: str = "name", reverse: bool = False, subfolders: bool = False) -> io.NodeOutput:
        resolved_path = _resolve_path(path)
        if not resolved_path:
            raise ValueError("Path is empty. Please provide a valid directory or image file path.")

        if os.path.isfile(resolved_path):
            files = [resolved_path]
        elif os.path.isdir(resolved_path):
            valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".tiff", ".tga"}
            files = []
            if subfolders:
                for root, _, filenames in os.walk(resolved_path):
                    for f in filenames:
                        if os.path.splitext(f)[1].lower() in valid_exts:
                            files.append(os.path.join(root, f))
            else:
                for f in os.listdir(resolved_path):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        files.append(os.path.join(resolved_path, f))

            if not files:
                raise FileNotFoundError(f"No valid images found in: {path}")

            if sort_by == "name":
                files.sort()
            elif sort_by == "date_modified":
                files.sort(key=lambda x: os.path.getmtime(x))
            elif sort_by == "date_created":
                files.sort(key=lambda x: os.path.getctime(x))
            elif sort_by == "random":
                # Deterministic random shuffle based on index
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

            # Save preview to temp directory for UI preview display
            temp_dir = folder_paths.get_temp_directory()
            preview_filename = f"preview_{os.urandom(6).hex()}.webp"
            preview_path = os.path.join(temp_dir, preview_filename)
            try:
                rgb_img.save(preview_path, format="webp", quality=90)
                ui_output = SavedImages([SavedResult(preview_filename, "", FolderType.temp)])
            except Exception:
                ui_output = SavedImages([])

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
            ui=ui_output,
        )


class LoadImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LoadImageFromFolder]


async def comfy_entrypoint() -> LoadImageExtension:
    return LoadImageExtension()
