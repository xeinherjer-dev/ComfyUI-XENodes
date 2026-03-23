# ComfyUI-XENodes

A collection of custom nodes for ComfyUI, featuring the versatile Multi-Switch node.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes` directory:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/xeinherjer-dev/ComfyUI-XENodes.git
   ```

2. Start (or restart) ComfyUI.

## Included Nodes

### Multi-Switch

A general-purpose switch node that selects one input from many and routes it to a single output.

- **Autogrow**: Input slots automatically increase as you connect more nodes.
- **Custom UI**: Convenient selection buttons are displayed directly on the node, showing the source node names of connected inputs.
- **Hide Connections**: Toggle the visibility of connection slots via the right-click menu to keep your workflow clean and compact.

![Multi-Switch UI](assets/screenshot1.webp)

### Slider

A versatile slider node that outputs a numerical value, compatible with both INT and FLOAT inputs.

- **Automatic Casting**: Automatically returns `int` for whole numbers and `float` for fractional ones, ensuring compatibility across all node types.

### Save Video

A powerful video saving node that writes video files natively, providing more control over encoding settings than the default Save Image node.

- **Native Encoding**: Saves video directly using FFmpeg (PyAV) with support for modern codecs.
- **AV1 Support**: Includes support for the highly efficient AV1 codec.
- **Format Options**: Choice of `mp4` and `webm` containers.
- **CRF Control**: Adjustable Constant Rate Factor for precise quality management.
- **Metadata Persistence**: Preserves ComfyUI prompt and metadata within the video file.
- **Audio Integration**: Automatically handles audio synchronization when available.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
