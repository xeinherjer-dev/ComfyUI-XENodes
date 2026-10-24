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

import { app } from '../../../../scripts/app.js'

// ---------------------------------------------------------------------------
// Inlined utilities from ComfyUI_frontend (@/utils/fileUtil, @/utils/jsonUtil)
// ---------------------------------------------------------------------------

/**
 * Reads a file (optionally capped to its first `maxBytes`) into an ArrayBuffer.
 * Resolves `null` when the read errors, aborts, or yields no result, so callers
 * can guard with a single truthiness check before parsing. The underlying
 * `FileReader` error reason is intentionally not surfaced; callers treat any
 * failure as "no metadata" rather than distinguishing causes.
 * @param file - The file to read.
 * @param maxBytes - Optional cap; when set, only the first `maxBytes` are read.
 */
function readFileAsArrayBuffer(file, maxBytes) {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = () => {
      resolve(reader.result instanceof ArrayBuffer ? reader.result : null)
    }
    reader.onerror = () => resolve(null)
    reader.onabort = () => resolve(null)
    const blob = maxBytes === undefined ? file : file.slice(0, maxBytes)
    reader.readAsArrayBuffer(blob)
  })
}

// Match a JSON string OR a non-finite token outside string.
// - `"(?:\\.|[^"\\])*"`  a quoted string - matched first so anything that
//                        looks like a token inside a string is skipped
// - `(?<![\w.-])`        skip non bare tokens (e.g. 1NaN)
// - `(-?Infinity|NaN)`   capture the non-finite token
// - `(?![\w.])`          skip non bare suffix tokens (e.g. NaN1)
const NON_FINITE_TOKEN =
  /"(?:\\.|[^"\\])*"|(?<![\w.-])(-?Infinity|NaN)(?![\w.])/g

function replaceNonFiniteTokens(text) {
  let hasWarned = false
  return text.replace(NON_FINITE_TOKEN, (match, token) => {
    if (token) {
      if (!hasWarned) {
        console.warn(
          'JSON contained non-finite numeric tokens (NaN/Infinity); they were replaced with null.'
        )
        hasWarned = true
      }
      return 'null'
    }
    return match
  })
}

/**
 * Parse JSON that may contain bare `NaN`/`Infinity`/`-Infinity` tokens
 * (which Python's `json.dumps` emits) by replacing them with `null` on
 * fallback. Coercion is lossy; a one-time warning is logged when it fires.
 */
function parseJsonWithNonFinite(text) {
  try {
    return JSON.parse(text)
  } catch {
    return JSON.parse(replaceNonFiniteTokens(text))
  }
}

// ---------------------------------------------------------------------------
// Fixed OGG metadata parser (ported from ComfyUI_frontend fix-ogg-syntax-9256)
// ---------------------------------------------------------------------------

const OGG_HEADER_SIZE = 27 // Base size of the Ogg page header (from magic number to just before segment count)
const OGG_PAGE_SEGMENTS_OFFSET = 26 // Offset position where the number of segments in the page is stored
const OGG_MAX_SEGMENT_SIZE = 255 // Ogg spec: the maximum size of a single segment is 255 bytes
const OGG_MAX_PAGE_SIZE = 65307 // Max size of a single Ogg page: 27 + 255 + 255 * 255

// Magic numbers as byte arrays to avoid per-page TextDecoder allocations
const OGG_S_MAGIC = new TextEncoder().encode('OggS')
const OPUS_HEAD_MAGIC = new TextEncoder().encode('OpusHead')
const OPUS_TAGS_MAGIC = new TextEncoder().encode('OpusTags')

// Metadata (OpusTags) in Ogg Opus is strictly located before any audio data (RFC 7845).
// We start with a 64 KB window (enough for typical single-page headers) and expand
// based on page-boundary requirements up to a generous 16 MB cap, aborting early as
// soon as audio data is reached or the packet completes.
const INITIAL_READ_BYTES = 64 * 1024
const MAX_READ_BYTES = 16 * 1024 * 1024

/**
 * Extracts ComfyUI metadata (prompt and workflow) from an Ogg/Opus audio file.
 *
 * Reads the head of the file as a binary ArrayBuffer through the shared
 * {@link readFileAsArrayBuffer} helper, locates the OpusTags packet by
 * tracing Ogg pages, and parses the embedded Vorbis Comments to retrieve
 * JSON-encoded prompt and workflow data. The read window starts at 64 KB and
 * grows strictly based on Ogg page boundaries until OpusTags completes or
 * audio data is reached (see {@link readOpusTagsPacket}).
 *
 * @param file - The Ogg/Opus file to extract metadata from.
 * @returns A promise resolving to a {@link ComfyMetadata} object containing
 *   the parsed `prompt` and `workflow` fields, or `undefined` for each if
 *   not found or if parsing fails.
 */
export async function getOggMetadata(file) {
  const packetData = await readOpusTagsPacket(file)
  if (!packetData) return { prompt: undefined, workflow: undefined }

  // Parse the reconstructed packet according to the Vorbis Comments specification to extract JSON
  return parseVorbisComments(packetData, new TextDecoder('utf-8'))
}

async function readOpusTagsPacket(file) {
  let readLimit = INITIAL_READ_BYTES
  const fileSize = file.size > 0 ? file.size : undefined

  for (;;) {
    const currentLimit = fileSize ? Math.min(readLimit, fileSize) : readLimit
    const buffer = await readFileAsArrayBuffer(file, currentLimit)
    if (!buffer) return undefined

    const scan = extractOpusTags(new Uint8Array(buffer))
    if (scan.status === 'complete') return concatSegments(scan.segments)

    const reachedEof =
      buffer.byteLength < currentLimit ||
      (fileSize !== undefined && buffer.byteLength >= fileSize)

    if (
      scan.status !== 'needMoreData' ||
      reachedEof ||
      readLimit >= MAX_READ_BYTES
    ) {
      warnIncompleteScan(scan, !reachedEof)
      return undefined
    }

    readLimit = Math.min(
      Math.max(readLimit + OGG_MAX_PAGE_SIZE, scan.neededBytes),
      MAX_READ_BYTES
    )
  }
}

/**
 * Logs why no complete OpusTags packet could be collected.
 */
function warnIncompleteScan(scan, hitReadCap) {
  if (
    scan.status === 'needMoreData' &&
    hitReadCap &&
    scan.segments.length > 0
  ) {
    console.warn(
      'Ogg metadata parsing: OpusTags packet extends beyond the 16 MB read cap ' +
        'and was truncated; prompt/workflow metadata will be unavailable.'
    )
  } else if (scan.segments.length > 0) {
    console.warn(
      'Ogg metadata parsing failed: OpusTags packet is truncated or corrupted'
    )
  } else {
    console.warn(
      'Ogg metadata parsing failed: No OpusTags found or invalid Ogg file'
    )
  }
}

/**
 * Concatenates the extracted segments to reconstruct the complete OpusTags packet (binary data).
 */
function concatSegments(segments) {
  const packetData = new Uint8Array(
    segments.reduce((sum, seg) => sum + seg.length, 0)
  )
  let currentOffset = 0
  for (const seg of segments) {
    packetData.set(seg, currentOffset)
    currentOffset += seg.length
  }
  return packetData
}

/**
 * Identifies the type of packet starting at the given segment based on header magic numbers.
 */
function identifyPacketType(segment, segmentLength) {
  if (segmentLength >= 8) {
    if (compareMagic(segment, 0, OPUS_TAGS_MAGIC)) return 'tags'
    if (compareMagic(segment, 0, OPUS_HEAD_MAGIC)) return 'head'
  }
  return segmentLength > 0 ? 'audio' : 'none'
}

/**
 * Advances packet scanning state for one segment.
 * Returns 'complete' or 'noOpusTags' if parsing must terminate, otherwise undefined.
 */
function stepSegment(segment, segmentLength, state, segments) {
  if (!state.inTags && !state.inHead) {
    const packetType = identifyPacketType(segment, segmentLength)
    if (packetType === 'tags') {
      state.inTags = true
    } else if (packetType === 'head') {
      state.inHead = segmentLength === OGG_MAX_SEGMENT_SIZE
    } else if (packetType === 'audio') {
      return 'noOpusTags'
    }
  } else if (state.inHead && segmentLength < OGG_MAX_SEGMENT_SIZE) {
    state.inHead = false
  }

  if (state.inTags) {
    segments.push(segment)
    if (segmentLength < OGG_MAX_SEGMENT_SIZE) {
      return 'complete'
    }
  }

  return undefined
}

/**
 * Processes segments within a single Ogg page to extract OpusTags data.
 */
function processPageSegments(
  data,
  lengthsOffset,
  startDataOffset,
  pageSegmentsCount,
  inOpusTags,
  inOpusHead,
  segments
) {
  let dataOffset = startDataOffset
  const state = { inTags: inOpusTags, inHead: inOpusHead }

  for (let i = 0; i < pageSegmentsCount; i++) {
    const segmentLength = data[lengthsOffset + i]
    const segment = data.subarray(dataOffset, dataOffset + segmentLength)
    dataOffset += segmentLength

    const terminal = stepSegment(segment, segmentLength, state, segments)
    if (terminal) {
      return {
        nextOffset: dataOffset,
        inOpusTags: false,
        inOpusHead: false,
        state: terminal
      }
    }
  }

  return {
    nextOffset: dataOffset,
    inOpusTags: state.inTags,
    inOpusHead: state.inHead,
    state: 'continue'
  }
}

/**
 * Calculates page offsets and verifies whether the entire page is available in the buffer.
 * Returns the parsed bounds, or a number representing `neededBytes` if the page is truncated.
 */
function parsePageBounds(data, offset) {
  const pageSegmentsCount = data[offset + OGG_PAGE_SEGMENTS_OFFSET]
  const lengthsOffset = offset + OGG_HEADER_SIZE
  const dataOffset = lengthsOffset + pageSegmentsCount

  if (dataOffset > data.length) return dataOffset

  let pagePayloadLength = 0
  for (let i = 0; i < pageSegmentsCount; i++) {
    pagePayloadLength += data[lengthsOffset + i]
  }
  const fullPageSize = OGG_HEADER_SIZE + pageSegmentsCount + pagePayloadLength
  return offset + fullPageSize > data.length
    ? offset + fullPageSize
    : { fullPageSize, pageSegmentsCount, dataOffset }
}

/**
 * Traverses the Ogg page structure to extract all segments belonging to the
 * OpusTags packet.
 */
function extractOpusTags(data) {
  const segments = []
  let offset = 0
  let inOpusTags = false
  let inOpusHead = false

  while (offset + OGG_HEADER_SIZE <= data.length) {
    if (!compareMagic(data, offset, OGG_S_MAGIC)) {
      return { status: 'invalid', segments }
    }

    const bounds = parsePageBounds(data, offset)
    if (typeof bounds === 'number') {
      return {
        status: 'needMoreData',
        segments,
        neededBytes: bounds
      }
    }

    const result = processPageSegments(
      data,
      offset + OGG_HEADER_SIZE,
      bounds.dataOffset,
      bounds.pageSegmentsCount,
      inOpusTags,
      inOpusHead,
      segments
    )

    inOpusTags = result.inOpusTags
    inOpusHead = result.inOpusHead
    offset += bounds.fullPageSize

    if (result.state === 'complete' || result.state === 'noOpusTags') {
      return { status: result.state, segments }
    }
  }

  return {
    status: 'needMoreData',
    segments,
    neededBytes: inOpusTags
      ? offset + OGG_MAX_PAGE_SIZE
      : offset + OGG_HEADER_SIZE
  }
}

/**
 * Skips the 8-byte OpusTags magic string and vendor string to locate the comments count offset.
 */
function getCommentsOffset(packetData, packetView) {
  const readIndex = 8 // Skip 'OpusTags' magic string (8 bytes)

  // Bounds check: ensure vendor length field is within packet
  if (readIndex + 4 > packetData.length) return null

  // Skip vendor string length (4 bytes) + vendor string
  const vendorLength = packetView.getUint32(readIndex, true)
  const afterVendor = readIndex + 4 + vendorLength

  // Bounds check: ensure vendor string and the user comment list length field
  // are within packet
  if (afterVendor + 4 > packetData.length) return null
  return afterVendor
}

/**
 * Parses an individual Vorbis comment KEY=VALUE pair into the ComfyMetadata result.
 */
function parseCommentEntry(text, result) {
  const separatorIndex = text.indexOf('=')
  if (separatorIndex === -1) return

  const key = text.substring(0, separatorIndex).toLowerCase()
  const value = text.substring(separatorIndex + 1)

  if (key === 'prompt') {
    try {
      result.prompt = parseJsonWithNonFinite(value)
    } catch (e) {
      console.error('Failed to parse Ogg prompt metadata', e)
    }
  } else if (key === 'workflow') {
    try {
      result.workflow = parseJsonWithNonFinite(value)
    } catch (e) {
      console.error('Failed to parse Ogg workflow metadata', e)
    }
  }
}

/**
 * Parses a reconstructed OpusTags packet according to the Vorbis Comments
 * specification and extracts ComfyUI metadata fields.
 *
 * Skips the 8-byte `OpusTags` magic string, reads the vendor string, then
 * iterates over user comments formatted as `KEY=VALUE` pairs. Recognises
 * `prompt` and `workflow` keys and JSON-parses their values into the
 * returned metadata object.
 *
 * @param packetData - The complete, concatenated OpusTags packet as a
 *   `Uint8Array`.
 * @param decoder - UTF-8 TextDecoder used to decode comment strings.
 * @returns A {@link ComfyMetadata} object with `prompt` and/or `workflow`
 *   populated from the embedded comments, or `undefined` for missing fields.
 */
function parseVorbisComments(packetData, decoder) {
  const packetView = new DataView(
    packetData.buffer,
    packetData.byteOffset,
    packetData.byteLength
  )

  const commentsOffset = getCommentsOffset(packetData, packetView)
  if (commentsOffset === null) {
    return { prompt: undefined, workflow: undefined }
  }

  // Get the number of user comments
  const userCommentListLength = packetView.getUint32(commentsOffset, true)
  let readIndex = commentsOffset + 4

  const result = {}
  for (let i = 0; i < userCommentListLength; i++) {
    // Bounds check: ensure comment length field is within packet
    if (readIndex + 4 > packetData.length) break

    // Vorbis Comments spec: Get comment length (32-bit little-endian)
    const commentLength = packetView.getUint32(readIndex, true)
    readIndex += 4

    // Bounds check: ensure comment data is within packet
    if (readIndex + commentLength > packetData.length) break

    // Extract and decode the comment string (UTF-8)
    const text = decoder.decode(
      packetData.subarray(readIndex, readIndex + commentLength)
    )
    readIndex += commentLength

    parseCommentEntry(text, result)
    if (result.prompt !== undefined && result.workflow !== undefined) break
  }

  return result
}

/**
 * Compares a portion of a Uint8Array with a magic byte sequence.
 */
function compareMagic(data, offset, magic) {
  for (let i = 0; i < magic.length; i++) {
    if (data[offset + i] !== magic[i]) return false
  }
  return true
}

// ---------------------------------------------------------------------------
// Monkey-patch: wrap app.handleFile to intercept OGG files
// ---------------------------------------------------------------------------

app.registerExtension({
  name: 'XENodes.OggMetadataPatch',
  async setup() {
    console.log('[XENodes/ogg_patch] OGG patch setup')
    const origHandleFile = app.handleFile.bind(app)

    app.handleFile = async function (file) {
      if (file?.type === 'audio/ogg') {
        console.log(
          '[XENodes/ogg_patch] Intercepting OGG file, using fixed parser'
        )
        try {
          const { prompt, workflow } = await getOggMetadata(file)
          if (workflow) {
            await app.loadGraphData(workflow)
          } else {
            console.warn(
              '[XENodes/ogg_patch] No workflow found in OGG metadata'
            )
          }
          // Note: prompt is usually handled by other parts of ComfyUI after loadGraphData,
          // so loading the workflow here is sufficient for file drop actions.
          return
        } catch (e) {
          console.error(
            '[XENodes/ogg_patch] OGG patch failed, falling back to original handler:',
            e
          )
          // Fallback: try original implementation
          return origHandleFile(file)
        }
      }
      return origHandleFile(file)
    }
  }
})
