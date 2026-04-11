/**
 * Monkey-patch: OGG metadata parser fix
 *
 * This is a TEMPORARY workaround that patches the broken OGG metadata
 * parser in ComfyUI_frontend until the official fix from the
 * PR below is merged and released.
 *
 * Once the official fix is available in the shipped frontend, this file
 * should be DELETED.
 *
 * Reference: https://github.com/Comfy-Org/ComfyUI_frontend/pull/9322
 * Issue: The original parser used string-matching on binary Ogg data, which
 *        is unreliable. This replacement correctly traces the Ogg page
 *        structure and parses Vorbis Comments according to spec.
 */

import { app } from "../../../../scripts/app.js";

// ---------------------------------------------------------------------------
// Fixed OGG metadata parser (ported from PR #9322)
// ---------------------------------------------------------------------------

const OGG_HEADER_SIZE = 27;
const OGG_PAGE_SEGMENTS_OFFSET = 26;
const OGG_MAX_SEGMENT_SIZE = 255;
const MAX_READ_BYTES = 2 * 1024 * 1024;

/**
 * Traverses the Ogg page structure to extract all segments belonging to the
 * OpusTags packet.
 * @param {Uint8Array} data
 * @param {TextDecoder} decoder
 * @returns {Uint8Array[]}
 */
function extractOpusTags(data, decoder) {
  const segments = [];
  let offset = 0;
  let inOpusTags = false;

  while (offset + OGG_HEADER_SIZE <= data.length) {
    const pageSignature = decoder.decode(data.subarray(offset, offset + 4));
    if (pageSignature !== "OggS") break;

    const pageSegmentsCount = data[offset + OGG_PAGE_SEGMENTS_OFFSET];
    const lengthsOffset = offset + OGG_HEADER_SIZE;
    let dataOffset = lengthsOffset + pageSegmentsCount;

    if (dataOffset > data.length) break;

    let pageProcessingEnded = false;

    for (let i = 0; i < pageSegmentsCount; i++) {
      const segmentLength = data[lengthsOffset + i];

      if (dataOffset + segmentLength > data.length) break;

      const segment = data.subarray(dataOffset, dataOffset + segmentLength);
      dataOffset += segmentLength;

      if (!inOpusTags) {
        if (segmentLength >= 8) {
          const segmentMagic = decoder.decode(segment.subarray(0, 8));
          if (segmentMagic === "OpusTags") {
            inOpusTags = true;
          }
        }
      }

      if (inOpusTags) {
        segments.push(segment);
        if (segmentLength < OGG_MAX_SEGMENT_SIZE) {
          pageProcessingEnded = true;
          break;
        }
      }
    }

    if (pageProcessingEnded) break;
    offset = dataOffset;
  }

  return segments;
}

/**
 * Parses a reconstructed OpusTags packet (Vorbis Comments) and extracts
 * ComfyUI metadata fields (prompt / workflow).
 * @param {Uint8Array} packetData
 * @param {TextDecoder} decoder
 * @returns {{ prompt?: any, workflow?: any }}
 */
function parseVorbisComments(packetData, decoder) {
  let readIndex = 8; // skip 'OpusTags' magic (8 bytes)
  const packetView = new DataView(
    packetData.buffer,
    packetData.byteOffset,
    packetData.byteLength
  );

  if (readIndex + 4 > packetData.length)
    return { prompt: undefined, workflow: undefined };

  const vendorLength = packetView.getUint32(readIndex, true);
  readIndex += 4;

  if (readIndex + vendorLength > packetData.length)
    return { prompt: undefined, workflow: undefined };
  readIndex += vendorLength;

  if (readIndex + 4 > packetData.length)
    return { prompt: undefined, workflow: undefined };

  const userCommentListLength = packetView.getUint32(readIndex, true);
  readIndex += 4;

  const result = {};
  for (let i = 0; i < userCommentListLength; i++) {
    if (readIndex + 4 > packetData.length) break;

    const commentLength = packetView.getUint32(readIndex, true);
    readIndex += 4;

    if (readIndex + commentLength > packetData.length) break;

    const text = decoder.decode(
      packetData.subarray(readIndex, readIndex + commentLength)
    );
    readIndex += commentLength;

    const separatorIndex = text.indexOf("=");
    if (separatorIndex !== -1) {
      const key = text.substring(0, separatorIndex).toLowerCase();
      const value = text.substring(separatorIndex + 1);
      if (key === "prompt") {
        try {
          result.prompt = JSON.parse(value);
        } catch (e) {
          console.warn("[XENodes/ogg_patch] Failed to parse prompt:", e);
        }
      } else if (key === "workflow") {
        try {
          result.workflow = JSON.parse(value);
        } catch (e) {
          console.warn("[XENodes/ogg_patch] Failed to parse workflow:", e);
        }
      }
    }

    if (result.prompt !== undefined && result.workflow !== undefined) break;
  }

  return result;
}

/**
 * Fixed replacement for getOggMetadata().
 * @param {File} file
 * @returns {Promise<{ prompt?: any, workflow?: any }>}
 */
async function getOggMetadataFixed(file) {
  const arrayBuffer = await file.slice(0, MAX_READ_BYTES).arrayBuffer();
  const data = new Uint8Array(arrayBuffer);
  const decoder = new TextDecoder("utf-8");

  const segments = extractOpusTags(data, decoder);
  if (segments.length === 0) {
    console.warn(
      "[XENodes/ogg_patch] No OpusTags found or invalid Ogg file"
    );
    return { prompt: undefined, workflow: undefined };
  }

  const totalLength = segments.reduce((sum, seg) => sum + seg.length, 0);
  const packetData = new Uint8Array(totalLength);
  let currentOffset = 0;
  for (const seg of segments) {
    packetData.set(seg, currentOffset);
    currentOffset += seg.length;
  }

  return parseVorbisComments(packetData, decoder);
}

// ---------------------------------------------------------------------------
// Monkey-patch: wrap app.handleFile to intercept OGG files
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "XENodes.OggMetadataPatch",
  async setup() {
    console.log("[XENodes/ogg_patch] OGG patch setup");
    const origHandleFile = app.handleFile.bind(app);

    app.handleFile = async function (file) {
      if (file?.type === "audio/ogg") {
        console.log(
          "[XENodes/ogg_patch] Intercepting OGG file, using fixed parser"
        );
        try {
          const { prompt, workflow } = await getOggMetadataFixed(file);
          if (workflow) {
            await app.loadGraphData(workflow);
          } else {
            console.warn(
              "[XENodes/ogg_patch] No workflow found in OGG metadata"
            );
          }
          // Note: prompt is usually handled by other parts of ComfyUI after loadGraphData,
          // so loading the workflow here is sufficient for file drop actions.
          return;
        } catch (e) {
          console.error(
            "[XENodes/ogg_patch] OGG patch failed, falling back to original handler:",
            e
          );
          // Fallback: try original implementation
          return origHandleFile(file);
        }
      }
      return origHandleFile(file);
    };
  },
});
