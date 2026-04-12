# ComfyUI-XENodes

A collection of custom nodes for ComfyUI, featuring Multi-Switch, Slider, Slider 2D, Save Image, Save Video, Dynamic Combo Selector, and Show Any nodes..

## Features

- **Nodes 2.0 Support**: Modern UI and compatibility for latest ComfyUI versions.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes` directory:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/xeinherjer-dev/ComfyUI-XENodes.git xenodes
   ```

2. Start (or restart) ComfyUI.

## Included Nodes

### Multi-Switch

A general-purpose switch node that selects one input from many and routes it to a single output.

- **Nodes 2.0 Compatible**: Dynamic UI that stays clean and responsive in the latest ComfyUI.
- **Autogrow**: Input slots automatically increase as you connect more nodes.
- **Custom UI**: Convenient selection buttons are displayed directly on the node, showing the source node names of connected inputs.
- **Hide Connections**: Toggle the visibility of connection slots via the right-click menu to keep your workflow clean and compact.

![Multi-Switch UI](assets/screenshot1.webp)

### Slider

A versatile slider node where the output port type dynamically switches between `INT` and `FLOAT` based on the `step` setting.

- **Dynamic Type Switching**: Automatically sets the output type to `INT` for integer steps and `FLOAT` for decimal steps, ensuring seamless connectivity with other nodes.
- **Automatic Casting**: During backend execution, numerical values are appropriately output as `int` for whole numbers and `float` for fractional ones.

### Slider 2D

An intuitive 2D slider node for manipulating X and Y values simultaneously on a coordinate plane.

- **Precision Auto-Detection**: Automatically adjusts the display precision (number of decimal places) based on the `stepX` and `stepY` settings.
- **Dynamic Type Switching**: Similar to the Slider node, the output port types for each axis change dynamically based on their respective step settings.
- **Snap Feature**: Enable the `snap` property to snap the handle to the specified step increments.

### Save Image

An image saving node with configurable **format** (`png`, `webp`), **lossless** encoding, and **quality** settings.

### Save Video

A video saving node with configurable **format** (`mp4`, `webm`), **codec** (`h264`, `h265`, `av1`), and **crf** settings.

### Dynamic Combo Selector

A utility node that allows selecting a COMBO (dropdown) value by its numerical **index**.

- **Auto-Discovery**: Dynamically reads the available options from the connected target node.
- **Index Selection**: Input an `INT` to select items by index.
- **Dual Output**: Provides both **COMBO** and **STRING** types for flexible connectivity.

### Show Any

A visual debug node that displays any input value as text directly on the node.

- **Any Input**: Accepts any data type and converts it to a readable string representation (JSON for complex objects).
- **Composite ID Support**: Fully compatible with Nodes 2.0 Group Nodes, ensuring progress is saved correctly even when nested.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
