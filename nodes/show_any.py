from __future__ import annotations
from typing_extensions import override
import json
import logging

from comfy_api.latest import ComfyExtension, io

class ShowAnyNode(io.ComfyNode):
    # Per-instance cache of the last valid value (keyed by unique_id)
    _last_value: dict = {}

    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("*", allowed_types=io.AnyType)

        return io.Schema(
            node_id="XENodes.ShowAny",
            display_name="Show Any",
            category="XENodes",
            inputs=[
                io.MatchType.Input("value", template=template, display_name="any"),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="any"),
            ],
            is_output_node=True,
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo]
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        value = kwargs.get("value")

        # Retrieve unique_id for cache operations
        unique_id = None
        hold_last_value = False
        if hasattr(cls, "hidden") and cls.hidden:
            unique_id = cls.hidden.unique_id
            extra_pnginfo = cls.hidden.extra_pnginfo

            if extra_pnginfo and "workflow" in extra_pnginfo:
                workflow = extra_pnginfo["workflow"]
                nodes = workflow.get("nodes", [])
                definitions = workflow.get("definitions", {})

                # Read the property directly from the node representation
                hold_last_value = cls.get_node_property(nodes, str(unique_id), "hold_last_value", definitions, False)

        if unique_id:
            if value is None and hold_last_value:
                # Fall back to the cached value when input is None
                value = cls._last_value.get(unique_id)
            elif value is not None:
                if hold_last_value:
                    # Update the cache with the new valid value
                    cls._last_value[unique_id] = value
                else:
                    # Cache disabled: clear any stored value
                    cls._last_value.pop(unique_id, None)

        # Convert arbitrary input to its string representation for UI display
        if value is None:
            text_str = "None"
        elif isinstance(value, str):
            text_str = value
        elif isinstance(value, (int, float, bool)):
            text_str = str(value)
        else:
            try:
                text_str = json.dumps(value, indent=2)
            except Exception:
                text_str = str(value)

        # Workflow persistence (Subgraph & Group Node support)
        # Using the framework's native hidden context which properly resolves composite IDs
        if hasattr(cls, "hidden") and cls.hidden:
            extra_pnginfo = cls.hidden.extra_pnginfo

            if extra_pnginfo and "workflow" in extra_pnginfo:
                workflow = extra_pnginfo["workflow"]
                nodes = workflow.get("nodes", [])
                definitions = workflow.get("definitions", {})

                if not cls.mutate_workflow_data(nodes, str(unique_id), text_str, definitions):
                    logging.warning(f"[XENodes.ShowAny] Failed to update workflow data for node {unique_id}")

        # Return original value (for the 'any' output)
        # The 'ui' dict triggers the frontend JavaScript to update the node's visual widget
        return io.NodeOutput(value, ui={"text": [text_str]})

    @classmethod
    def get_node_property(cls, nodes: list, target_id: str, prop_name: str, definitions: dict, default=False) -> bool:
        """
        Splits hierarchical node IDs to recursively search and read a property from a node.
        """
        id_parts = target_id.split(':', 1)
        current_id = id_parts[0]
        remaining_id = id_parts[1] if len(id_parts) > 1 else None

        for node in nodes:
            if str(node.get("id")) != current_id:
                continue

            if remaining_id is None:
                props = node.get("properties", {})
                return props.get(prop_name, default)

            node_type = node.get("type")
            for sub_list in ShowAnyNode._get_sub_nodes(node, node_type, definitions):
                if sub_list:
                    ret = ShowAnyNode.get_node_property(sub_list, remaining_id, prop_name, definitions, default)
                    if ret != default:
                        return ret

        return default


    @staticmethod
    def mutate_workflow_data(nodes: list, target_id: str, new_text: str, definitions: dict = None) -> bool:
        """
        Splits hierarchical node IDs (e.g., "7:1") to recursively search and overwrite widgets_values.
        Supports ComfyUI Group Node (V2) definitions["subgraphs"] structure.
        """
        id_parts = target_id.split(':', 1)
        current_id = id_parts[0]
        remaining_id = id_parts[1] if len(id_parts) > 1 else None

        for node in nodes:
            if str(node.get("id")) != current_id:
                continue

            # Final target node reached: overwrite its widgets_values
            if remaining_id is None:
                wv = node.setdefault("widgets_values", [])
                if wv:
                    wv[0] = new_text
                else:
                    wv.append(new_text)
                return True

            # Intermediate node: dive deeper into the subgraph
            node_type = node.get("type")
            for sub_list in ShowAnyNode._get_sub_nodes(node, node_type, definitions):
                if sub_list and ShowAnyNode.mutate_workflow_data(sub_list, remaining_id, new_text, definitions):
                    return True

        return False

    @staticmethod
    def _get_sub_nodes(node: dict, node_type: str, definitions: dict) -> list[list]:
        """
        Collects child node lists from a given node.
        Supports various storage formats: direct nesting, properties, and definitions["subgraphs"].
        """
        candidates = []

        # 1. Direct nesting (legacy subgraphs)
        for key in ("nodes", "inner_nodes", "subgraph"):
            val = node.get(key)
            if isinstance(val, list):
                candidates.append(val)
            elif isinstance(val, dict):
                candidates.append(val.get("nodes", []))

        # 2. Inside properties
        props = node.get("properties")
        if isinstance(props, dict):
            sg = props.get("subgraph")
            if isinstance(sg, dict):
                candidates.append(sg.get("nodes", []))
            elif isinstance(props.get("nodes"), list):
                candidates.append(props["nodes"])

        # 3. Inside workflow.definitions (Group Node V2 architecture)
        if definitions and node_type:
            if node_type in definitions:
                group_def = definitions[node_type]
                if isinstance(group_def, dict):
                    candidates.append(group_def.get("nodes", []))
            elif "subgraphs" in definitions:
                subgraphs = definitions["subgraphs"]
                if isinstance(subgraphs, list):
                    for sg_def in subgraphs:
                        if sg_def.get("id") == node_type or sg_def.get("name") == node_type:
                            candidates.append(sg_def.get("nodes", []))
                            break
                elif isinstance(subgraphs, dict) and node_type in subgraphs:
                    sg_def = subgraphs[node_type]
                    if isinstance(sg_def, dict):
                        candidates.append(sg_def.get("nodes", []))

        return candidates


class ShowAnyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ShowAnyNode]


async def comfy_entrypoint() -> ShowAnyExtension:
    return ShowAnyExtension()
