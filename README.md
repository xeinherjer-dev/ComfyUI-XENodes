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

A general-purpose switch node that selects one input from many (up to 50) and routes it to a single output.

- **Autogrow**: Input slots automatically increase as you connect more nodes.
- **Custom UI**: Convenient selection buttons are displayed directly on the node, showing the source node names of connected inputs.
- **Hide Connections (Nodes 2.0 / Node-Red style support)**: Toggle the visibility of connection slots via the right-click menu to keep your workflow clean and compact.

## Key Features

- **Hide/Show Connections**:
  By right-clicking the node and selecting `Hide Connections`, you can hide all input/output pins and select widgets. This leaves only the custom selection buttons, creating a very clean look—perfect for organized, complex workflows.
  
- **Smart Dynamic Buttons**:
  Selection buttons are automatically generated for each connected input. These buttons display the title of the origin node, making it immediately clear which input is currently active.

## Screenshots

![Multi-Switch UI](assets/screenshot1.webp)

## Technical Details

- **Backend**: Implemented in Python using the latest `comfy_api.latest` (io.ComfyNode) system.
- **Frontend**: Custom JavaScript implementation compatible with both standard LiteGraph and the newer Nodes 2.0 (Vue 2.0) interface.
