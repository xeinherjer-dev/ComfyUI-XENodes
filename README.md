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

A video saving node with configurable **format** (`mp4`, `webm`), **codec** (`h264`, `h265`, `av1`), and **crf** settings.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
