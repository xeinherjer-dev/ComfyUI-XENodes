from __future__ import annotations

import os
import torch
import numpy as np
import struct
from typing_extensions import override

try:
    import imagecodecs
except ImportError:
    imagecodecs = None

from comfy_api.latest import ComfyExtension, io, Input, ui
from comfy_api.latest._ui import SavedImages, SavedResult, FolderType
import folder_paths


def _scan_jpeg_segments(data: bytes) -> list[tuple[bytes, int, int]]:
    """Scan JPEG and return list of (marker, start_pos, total_length).
    Properly scans through SOS compressed data to find EOI."""
    segments = []
    if len(data) < 2 or data[0:2] != b'\xff\xd8':
        return segments
    segments.append((b'\xff\xd8', 0, 2))
    pos = 2
    while pos < len(data) - 1:
        if data[pos] != 0xff:
            break
        m2 = data[pos+1]
        marker = data[pos:pos+2]
        if m2 == 0xd8 or m2 == 0xd9 or (0xd0 <= m2 <= 0xd7):
            segments.append((marker, pos, 2))
            pos += 2
            if m2 == 0xd9:
                break
            continue
        if pos + 4 > len(data):
            break
        seg_len = struct.unpack('>H', data[pos+2:pos+4])[0]
        total = 2 + seg_len
        segments.append((marker, pos, total))
        pos += total
        if m2 == 0xda:  # SOS - scan byte-by-byte for EOI
            # In JPEG compressed data, \xff is always stuffed as \xff\x00
            # or 0xD0-0xD7 (RST). So \xff\xd9 unambiguously marks EOI.
            eoi_pos = pos
            while eoi_pos < len(data) - 1:
                if data[eoi_pos] == 0xff and data[eoi_pos + 1] == 0xd9:
                    segments.append((b'\xff\xd9', eoi_pos, 2))
                    pos = eoi_pos + 2
                    break
                eoi_pos += 1
            break
    return segments


def _build_container_xmp(gainmap_length: int, gain_map_max: float) -> bytes:
    """Build Adobe XMP + Google GContainer metadata for HDR gain map recognition."""
    content = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.1.2">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""'
        ' xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"'
        ' xmlns:GContainer="http://ns.google.com/photos/1.0/container/"'
        ' xmlns:GContainerItem="http://ns.google.com/photos/1.0/container/item/"'
        ' hdrgm:Version="1.0"'
        ' hdrgm:GainMapMin="0"'
        f' hdrgm:GainMapMax="{gain_map_max:.6f}"'
        ' hdrgm:Gamma="1"'
        ' hdrgm:OffsetSDR="0.015625"'
        ' hdrgm:OffsetHDR="0.015625"'
        ' hdrgm:BaseRendition="SDR">'
        '<GContainer:Directory>'
        '<rdf:Seq>'
        '<rdf:li rdf:parseType="Resource">'
        '<GContainerItem:Semantic>Primary</GContainerItem:Semantic>'
        '<GContainerItem:Mime>image/jpeg</GContainerItem:Mime>'
        '</rdf:li>'
        '<rdf:li rdf:parseType="Resource">'
        '<GContainerItem:Semantic>GainMap</GContainerItem:Semantic>'
        '<GContainerItem:Mime>image/jpeg</GContainerItem:Mime>'
        f'<GContainerItem:Length>{gainmap_length}</GContainerItem:Length>'
        '</rdf:li>'
        '</rdf:Seq>'
        '</GContainer:Directory>'
        '</rdf:Description>'
        '</rdf:RDF>'
        '</x:xmpmeta>'
        '<?xpacket end="w"?>'
    ).encode('utf-8')

    xmp_header = b'http://ns.adobe.com/xap/1.0/\x00'
    segment_data = xmp_header + content
    length = len(segment_data) + 2
    return b'\xff\xe1' + struct.pack('>H', length) + segment_data


def _update_mpf_offsets(data: bytearray, xmp_size: int) -> None:
    """Find MPF APP2 segment and shift all non-zero image offsets by xmp_size.

    MPF Image offsets are relative to tiff_base (= mpf_pos + 4 [marker+len] + 4 ['MPF\x00']).
    Since tiff_base itself shifts by xmp_size, offsets must be updated.
    """
    segments = _scan_jpeg_segments(bytes(data))
    for marker, pos, total in segments:
        if marker != b'\xff\xe2':
            continue
        if data[pos+4:pos+8] != b'MPF\x00':
            continue

        # tiff_base = pos+4 (skip marker+length) + 4 (skip 'MPF\x00')
        # Verified from diagnostic: tiff_base=46, Image[1].offset=1467 → 46+1467=1513=gainmap_start ✓
        # After XMP injection, MPF segment moves to pos+xmp_size.
        # Its offsets are RELATIVE to the (new) tiff_base, so they must be adjusted
        # by xmp_size to still point to the correct absolute positions.
        tiff_base = pos + 4 + 4

        bo = data[tiff_base:tiff_base+2]
        if bo == b'II':
            endian = '<'
        elif bo == b'MM':
            endian = '>'
        else:
            return

        ifd_off = struct.unpack_from(f'{endian}I', data, tiff_base + 4)[0]
        ifd_pos = tiff_base + ifd_off
        num_entries = struct.unpack_from(f'{endian}H', data, ifd_pos)[0]
        entry_pos = ifd_pos + 2

        for _ in range(num_entries):
            tag = struct.unpack_from(f'{endian}H', data, entry_pos)[0]
            count = struct.unpack_from(f'{endian}I', data, entry_pos + 4)[0]
            val_or_off = struct.unpack_from(f'{endian}I', data, entry_pos + 8)[0]

            if tag == 0xB002:  # MPEntry
                mp_start = tiff_base + val_or_off
                n_images = count // 16
                for j in range(n_images):
                    off_pos = mp_start + j * 16 + 8
                    cur = struct.unpack_from(f'{endian}I', data, off_pos)[0]
                    if cur != 0:  # 0 = primary image (self-reference)
                        struct.pack_into(f'{endian}I', data, off_pos, cur + xmp_size)
                break
            entry_pos += 12
        return


def inject_uhdr_xmp(jpeg_data: bytes, max_content_boost: float) -> bytes:
    """Inject Google Container XMP after SOI and fix MPF offsets."""
    # Find gainmap start by locating primary JPEG's EOI
    segments = _scan_jpeg_segments(jpeg_data)
    gainmap_start = -1
    for marker, pos, total in segments:
        if marker == b'\xff\xd9':
            gainmap_start = pos + 2
            break

    if gainmap_start == -1 or gainmap_start >= len(jpeg_data):
        print("[XENodes] Warning: could not find gainmap boundary in JPEG-R")
        return jpeg_data

    gainmap_length = len(jpeg_data) - gainmap_start
    gain_map_max = float(np.log2(max(max_content_boost, 1.0001)))

    app1 = _build_container_xmp(gainmap_length, gain_map_max)
    xmp_size = len(app1)

    # Inject XMP after SOI (byte 2), then fix MPF offsets
    new_data = bytearray(jpeg_data[:2] + app1 + jpeg_data[2:])
    _update_mpf_offsets(new_data, xmp_size)

    return bytes(new_data)


class SaveUltraHDRImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XENodes.SaveUltraHDR",
            display_name="Save Ultra HDR",
            category="XENodes",
            description="Saves Ultra HDR JPEG using imagecodecs with GContainer XMP and corrected MPF offsets.",
            inputs=[
                io.Image.Input("images", tooltip="The SDR images to save."),
                io.String.Input("filename_prefix", default="image/UltraHDR",
                                tooltip="The prefix for the file to save."),
                io.Float.Input("highlight_threshold", default=0.8, min=0.0, max=1.0, step=0.01,
                               tooltip="Luminance threshold to start gain map."),
                io.Float.Input("max_content_boost", default=4.0, min=1.0, max=10.0, step=0.1,
                               tooltip="Max HDR boost factor."),
                io.Int.Input("quality", default=95, min=1, max=100,
                             tooltip="JPEG quality."),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Image.Output("gainmap", tooltip="The calculated gain map as a grayscale image.")
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images: Input.Image,
        filename_prefix: str,
        highlight_threshold: float,
        max_content_boost: float,
        quality: int
    ) -> io.NodeOutput:
        if imagecodecs is None:
            raise ImportError("The 'imagecodecs' package is required.")

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            images[0].shape[2],
            images[0].shape[1],
        )

        results = []
        gainmaps = []

        for batch_number, tensor in enumerate(images):
            # 1. Linearize (sRGB -> linear) and calculate gain map
            tensor_linear = torch.pow(tensor.clamp(0, 1), 2.2)
            luminance = (0.2126 * tensor_linear[..., 0]
                         + 0.7152 * tensor_linear[..., 1]
                         + 0.0722 * tensor_linear[..., 2])
            threshold_linear = highlight_threshold ** 2.2
            mask = torch.clamp(
                (luminance - threshold_linear) / (1.0 - threshold_linear + 1e-5), min=0.0)
            gainmap_tensor = torch.pow(mask, 2.0)

            # Store gainmap as 3-channel IMAGE [H, W, 3] for output pin
            gainmaps.append(gainmap_tensor.unsqueeze(-1).repeat(1, 1, 3))

            # 2. Synthesize HDR (boost highlights in linear space)
            hdr_linear = tensor_linear * (
                1.0 + gainmap_tensor.unsqueeze(-1) * (max_content_boost - 1.0))
            alpha = torch.ones_like(hdr_linear[..., :1])
            rgba_np = torch.cat([hdr_linear, alpha], dim=-1).cpu().numpy().astype(np.float16)

            # 3. Encode to JPEG-R with imagecodecs
            try:
                ultrahdr_bytes = imagecodecs.ultrahdr_encode(
                    rgba_np,
                    level=quality,
                    transfer=imagecodecs.ULTRAHDR.CT.LINEAR,
                    gamut=imagecodecs.ULTRAHDR.CG.BT_709
                )
            except Exception as e:
                print(f"[XENodes] UltraHDR encoding failed: {e}")
                raise

            # 4. Inject GContainer XMP and fix MPF offsets for Chrome compatibility
            ultrahdr_bytes = inject_uhdr_xmp(ultrahdr_bytes, max_content_boost)

            # 5. Save
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"
            file_path = os.path.join(full_output_folder, file)
            with open(file_path, "wb") as f:
                f.write(ultrahdr_bytes)

            results.append(SavedResult(file, subfolder, FolderType.output))
            counter += 1

        return io.NodeOutput(images, gainmaps, ui=SavedImages(results))


class SaveUltraHDRImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SaveUltraHDRImage]


async def comfy_entrypoint() -> SaveUltraHDRImageExtension:
    return SaveUltraHDRImageExtension()
