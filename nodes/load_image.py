from __future__ import annotations

import os
import re
import asyncio
import subprocess
import shutil
import torch
import numpy as np
from PIL import Image as PILImage, ImageOps as PILImageOps
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
from comfy_api.latest._ui import SavedImages, SavedResult, FolderType
import comfy.model_management
import folder_paths
import node_helpers

# Dynamically discover all image formats supported by Pillow
SUPPORTED_IMAGE_EXTS = {
    ext.lower()
    for ext, fmt in PILImage.registered_extensions().items()
    if fmt in PILImage.OPEN and ext.lower() not in {".pdf", ".eps", ".ps", ".bin"}
}
if not SUPPORTED_IMAGE_EXTS:
    SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".tiff", ".tga"}

VALID_IMAGE_EXTS = SUPPORTED_IMAGE_EXTS
MAX_SCAN_FILES = 10000


def get_allowed_directories() -> list[tuple[str, str]]:
    """Returns a list of (type_name, canonical_directory_path)."""
    allowed = []
    output_dir = folder_paths.get_output_directory()
    if output_dir:
        allowed.append(("output", os.path.realpath(output_dir)))
    input_dir = folder_paths.get_input_directory()
    if input_dir:
        allowed.append(("input", os.path.realpath(input_dir)))
    temp_dir = folder_paths.get_temp_directory()
    if temp_dir:
        allowed.append(("temp", os.path.realpath(temp_dir)))
    return allowed


def is_path_safe_and_allowed(target_path: str) -> bool:
    """Checks if the target_path resolves strictly within one of the allowed directories."""
    if not target_path:
        return False
    try:
        real_target = os.path.realpath(target_path)
        for _, base_dir in get_allowed_directories():
            if folder_paths.is_within_directory(base_dir, real_target):
                return True
    except Exception:
        return False
    return False


def get_relative_filepath(target_path: str, fallback_path_prefix: str = "") -> str:
    """Returns a relative path prefixed with 'output', 'input', or 'temp' matching the input path format."""
    if not target_path:
        return ""
    try:
        norm_target = os.path.normpath(target_path).replace("\\", "/")
        for prefix, base_dir in get_allowed_directories():
            if not base_dir:
                continue
            norm_base = os.path.normpath(base_dir).replace("\\", "/")
            if norm_target == norm_base or norm_target.startswith(norm_base + "/"):
                rel = norm_target[len(norm_base):].lstrip("/")
                return prefix if not rel else f"{prefix}/{rel}"

        # Fallback with realpath in case of symlinks
        real_target = os.path.realpath(target_path)
        for prefix, base_dir in get_allowed_directories():
            if folder_paths.is_within_directory(base_dir, real_target):
                rel = os.path.relpath(real_target, base_dir).replace(os.sep, "/")
                return prefix if rel == "." else f"{prefix}/{rel}"
    except Exception:
        pass
    if fallback_path_prefix:
        clean_prefix = fallback_path_prefix.replace("\\", "/").rstrip("/")
        return f"{clean_prefix}/{os.path.basename(target_path)}"
    return target_path.replace("\\", "/")


def _resolve_path(raw_path: str) -> str:
    path = raw_path.strip().strip('"').strip("'")
    output_dir = folder_paths.get_output_directory() or ""
    input_dir = folder_paths.get_input_directory() or ""
    temp_dir = folder_paths.get_temp_directory() or ""

    # If empty, default to output directory
    if not path:
        return os.path.realpath(output_dir) if output_dir else ""

    # If user explicitly specifies output, input, or temp prefix
    norm_path = path.replace("\\", "/")
    if norm_path == "output" or norm_path.startswith("output/"):
        rel = norm_path[6:].lstrip("/")
        return os.path.realpath(os.path.join(output_dir, rel))
    elif norm_path == "input" or norm_path.startswith("input/"):
        rel = norm_path[5:].lstrip("/")
        return os.path.realpath(os.path.join(input_dir, rel))
    elif norm_path == "temp" or norm_path.startswith("temp/"):
        rel = norm_path[4:].lstrip("/")
        return os.path.realpath(os.path.join(temp_dir, rel))

    # Auto-convert Windows path to WSL path if running under Linux/WSL
    if re.match(r"^[a-zA-Z]:[\\/]", path):
        drive = path[0].lower()
        wsl_path = f"/mnt/{drive}/" + path[2:].replace("\\", "/").lstrip("/")
        if os.path.exists(wsl_path):
            path = wsl_path

    # If it's already an absolute path, normalize it
    if os.path.isabs(path):
        real_p = os.path.realpath(path)
        if is_path_safe_and_allowed(real_p):
            return real_p
        # If absolute path is outside allowed directories, disallow
        return ""

    # For relative paths without prefix, check output directory first, then input directory
    if output_dir:
        candidate_output = os.path.realpath(os.path.join(output_dir, path))
        if os.path.exists(candidate_output) and is_path_safe_and_allowed(candidate_output):
            return candidate_output

    if input_dir:
        candidate_input = os.path.realpath(os.path.join(input_dir, path))
        if os.path.exists(candidate_input) and is_path_safe_and_allowed(candidate_input):
            return candidate_input

    # Fallback to output directory path
    if output_dir:
        candidate = os.path.realpath(os.path.join(output_dir, path))
        if is_path_safe_and_allowed(candidate):
            return candidate

    return ""


def get_image_files(path: str, sort_by: str = "name", reverse: bool = False, subfolders: bool = False, index: int = 0) -> list[str]:
    resolved_path = _resolve_path(path)
    if not resolved_path or not is_path_safe_and_allowed(resolved_path):
        return []

    if os.path.isfile(resolved_path):
        ext = os.path.splitext(resolved_path)[1].lower()
        if ext in VALID_IMAGE_EXTS:
            return [resolved_path]
        return []

    elif os.path.isdir(resolved_path):
        files = []
        if subfolders:
            for root, _, filenames in os.walk(resolved_path):
                # Ensure symlinks inside directories do not escape allowed sandbox
                if not is_path_safe_and_allowed(root):
                    continue
                for f in filenames:
                    if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTS:
                        files.append(os.path.join(root, f))
                        if len(files) >= MAX_SCAN_FILES:
                            break
                if len(files) >= MAX_SCAN_FILES:
                    break
        else:
            for f in os.listdir(resolved_path):
                if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTS:
                    full_p = os.path.join(resolved_path, f)
                    if os.path.isfile(full_p):
                        files.append(full_p)
                        if len(files) >= MAX_SCAN_FILES:
                            break

        if not files:
            return []

        if sort_by == "name":
            files.sort()
        elif sort_by == "date_modified":
            files.sort(key=lambda x: os.path.getmtime(x))
        elif sort_by == "date_created":
            files.sort(key=lambda x: os.path.getctime(x))
        elif sort_by == "random":
            rng = np.random.default_rng(index)
            rng.shuffle(files)

        if reverse and sort_by != "random":
            files.reverse()

        return files
    return []


import time

_FOLDERS_CACHE = {"time": 0.0, "data": ["output", "input"]}
FOLDERS_CACHE_TTL = 300.0  # 5 minutes cache


def _find_image_dirs(base_dir: str, prefix: str) -> list[str]:
    """Finds all directories under base_dir that contain at least one valid image file."""
    if not base_dir or not os.path.exists(base_dir):
        return []

    # 1. Fast path on Linux/WSL using native 'find' command
    if os.name == "posix" and shutil.which("find"):
        try:
            ext_args = []
            for ext in sorted(SUPPORTED_IMAGE_EXTS):
                clean = ext.lstrip(".")
                if ext_args:
                    ext_args.append("-o")
                ext_args.extend(["-name", f"*.{clean.lower()}", "-o", "-name", f"*.{clean.upper()}"])
            cmd = ["find", base_dir, "-type", "f", "("] + ext_args + [")", "-printf", "%h\n"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
            raw_dirs = set(filter(None, proc.stdout.split("\n")))
            result = []
            for d in raw_dirs:
                rel = os.path.relpath(d, base_dir).replace(os.sep, "/")
                result.append(prefix if rel == "." else f"{prefix}/{rel}")
            return result
        except Exception:
            pass

    # 2. Cross-platform Python fallback
    valid_dirs = set()
    try:
        for root, _, filenames in os.walk(base_dir, followlinks=False):
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in VALID_IMAGE_EXTS:
                    rel = os.path.relpath(root, base_dir).replace(os.sep, "/")
                    valid_dirs.add(prefix if rel == "." else f"{prefix}/{rel}")
                    break
    except Exception:
        pass

    return list(valid_dirs)


DATE_REGEX = re.compile(r"(\d{4}[-_/]\d{2}[-_/]\d{2})")


def _sort_folders(folders: list[str], sort_by: str = "newest_first") -> list[str]:
    """Sorts a list of folder relative paths based on sort_by criteria.
    Accurately detects YYYY-MM-DD dates in paths so newest dates always appear first.
    """
    result = list(folders)

    if sort_by == "oldest_first":
        def _sort_key_oldest(path: str):
            m = DATE_REGEX.search(path)
            if m:
                date_str = m.group(1).replace("/", "-").replace("_", "-")
                return (0, date_str, path)
            return (1, "", path)
        result.sort(key=_sort_key_oldest)

    else:  # newest_first (default)
        def _sort_key_newest(path: str):
            m = DATE_REGEX.search(path)
            if m:
                date_str = m.group(1).replace("/", "-").replace("_", "-")
                # Group 0: has date. Invert character codes to sort descending
                inv_date = "".join(chr(255 - ord(c)) for c in date_str)
                return (0, inv_date, path)
            # Group 1: non-dated folders at the end, sorted alphabetically
            return (1, "", path)
        result.sort(key=_sort_key_newest)

    return result


def get_available_folders(sort_by: str = "newest_first") -> list[str]:
    """Returns a list of all folders in output and input directories that contain supported images, sorted by sort_by."""
    global _FOLDERS_CACHE
    now = time.time()
    if now - _FOLDERS_CACHE["time"] < FOLDERS_CACHE_TTL and _FOLDERS_CACHE["data"]:
        raw_folders = _FOLDERS_CACHE["data"]
    else:
        folders = []
        output_dir = folder_paths.get_output_directory()
        if output_dir:
            folders.extend(_find_image_dirs(output_dir, "output"))

        input_dir = folder_paths.get_input_directory()
        if input_dir:
            folders.extend(_find_image_dirs(input_dir, "input"))

        raw_folders = folders or ["output", "input"]
        _FOLDERS_CACHE = {"time": now, "data": raw_folders}

    return _sort_folders(raw_folders, sort_by=sort_by)


try:
    import io as python_io
    from aiohttp import web
    from server import PromptServer

    if hasattr(PromptServer, "instance") and PromptServer.instance:
        @PromptServer.instance.routes.get("/xenodes/load_image/folders")
        async def get_load_image_folders(request: web.Request) -> web.Response:
            try:
                force = request.rel_url.query.get("force", "false").lower() in ("true", "1")
                sort_by = request.rel_url.query.get("sort_by", "newest_first")
                if force:
                    global _FOLDERS_CACHE
                    _FOLDERS_CACHE["time"] = 0.0
                folders = await asyncio.to_thread(get_available_folders, sort_by)
                return web.json_response(folders)
            except Exception:
                return web.json_response(["output", "input"])

        @PromptServer.instance.routes.get("/xenodes/load_image/preview")
        async def get_load_image_preview(request: web.Request) -> web.Response:
            try:
                path = request.rel_url.query.get("path", "output")
                index_str = request.rel_url.query.get("index", "0")
                try:
                    index = int(index_str)
                except ValueError:
                    index = 0
                sort_by = request.rel_url.query.get("sort_by", "name")
                reverse = request.rel_url.query.get("reverse", "false").lower() in ("true", "1")
                subfolders = request.rel_url.query.get("subfolders", "false").lower() in ("true", "1")

                resolved_path = _resolve_path(path)
                if not resolved_path or not is_path_safe_and_allowed(resolved_path):
                    return web.Response(status=403, text="Access denied: path must be within input or output directory")

                def _generate_preview():
                    files = get_image_files(path, sort_by=sort_by, reverse=reverse, subfolders=subfolders, index=index)
                    if not files:
                        return None, 404, "No images found"

                    total_images = len(files)
                    actual_index = index % total_images
                    target_file = files[actual_index]

                    if not os.path.isfile(target_file):
                        return None, 404, "File not found"

                    with PILImage.open(target_file) as img:
                        img = node_helpers.pillow(PILImageOps.exif_transpose, img)
                        rgb_img = img.convert("RGB")

                        buf = python_io.BytesIO()
                        rgb_img.save(buf, format="WEBP", quality=85)
                        return (buf.getvalue(), total_images, actual_index, os.path.basename(target_file)), 200, None

                # Offload blocking I/O and PIL operations to thread pool to prevent blocking aiohttp event loop
                res_data, status, err_msg = await asyncio.to_thread(_generate_preview)
                if status != 200:
                    return web.Response(status=status, text=err_msg)

                body, total_images, actual_index, filename = res_data
                return web.Response(
                    body=body,
                    content_type="image/webp",
                    headers={
                        "Cache-Control": "public, max-age=10",
                        "X-Total-Images": str(total_images),
                        "X-Actual-Index": str(actual_index),
                        "X-Filename": filename,
                    },
                )
            except Exception:
                return web.Response(status=500, text="Internal server error")
except Exception:
    pass


class LoadImageFromFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.LoadImageFromFolder",
            display_name="Load Image From Folder",
            category="xenodes/image",
            description="Loads a single image from a directory or direct file path without resizing, ideal for queues and vision LLMs.",
            inputs=[
                io.Combo.Input(
                    "path",
                    options=get_available_folders("newest_first"),
                    default="output",
                    tooltip="Select a folder from the output or input directory.",
                ),
                io.Combo.Input(
                    "folder_sort",
                    options=["newest_first", "oldest_first"],
                    default="newest_first",
                    tooltip="Display order of directories in the path combo dropdown.",
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
                io.String.Output("filepath", display_name="filepath", tooltip="Relative filepath of the loaded image (e.g. output/2026-09-13/file.webp)."),
                io.Int.Output("current_index", display_name="index", tooltip="Current index within the image list."),
                io.Int.Output("total_images", display_name="total_images", tooltip="Total count of images found."),
            ],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, path: str = "output", folder_sort: str = "newest_first") -> bool | str:
        resolved_path = _resolve_path(path)
        if not resolved_path:
            return f"Invalid or disallowed path: '{path}'. Path must be located inside ComfyUI 'input' or 'output' directory."

        if not is_path_safe_and_allowed(resolved_path):
            return f"Access denied: '{path}' is outside allowed directories (input/output)."

        if not os.path.exists(resolved_path):
            return f"Path does not exist: '{path}'"

        return True

    @classmethod
    def fingerprint_inputs(cls, path: str = "output", folder_sort: str = "newest_first", index: int = 0, sort_by: str = "name", reverse: bool = False, subfolders: bool = False) -> str:
        if sort_by == "random":
            return str(os.urandom(8))
        return f"{path}_{index}_{sort_by}_{reverse}_{subfolders}"

    @classmethod
    def execute(cls, path: str = "output", folder_sort: str = "newest_first", index: int = 0, sort_by: str = "name", reverse: bool = False, subfolders: bool = False) -> io.NodeOutput:
        resolved_path = _resolve_path(path)
        if not resolved_path:
            raise ValueError(f"Invalid or disallowed path: '{path}'. Path must be located inside ComfyUI 'input' or 'output' directory.")

        if not is_path_safe_and_allowed(resolved_path):
            raise PermissionError(f"Access denied: '{path}' is outside allowed directories (input/output).")

        files = get_image_files(path, sort_by=sort_by, reverse=reverse, subfolders=subfolders, index=index)
        if not files:
            if os.path.exists(resolved_path):
                raise FileNotFoundError(f"No valid images found in: {path}")
            else:
                raise FileNotFoundError(f"Path does not exist: {path}")

        total_images = len(files)
        actual_index = index % total_images
        target_file = files[actual_index]
        filename = os.path.basename(target_file)
        relative_filepath = get_relative_filepath(target_file, fallback_path_prefix=path)

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
            relative_filepath,
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
