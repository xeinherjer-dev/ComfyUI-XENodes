from __future__ import annotations
from typing_extensions import override
import json
import logging

from comfy_api.latest import ComfyExtension, io


def _format_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, indent=2)
    except Exception:
        return str(value)


class ShowAnyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("*", allowed_types=io.AnyType)

        return io.Schema(
            node_id="XENodes.ShowAny",
            display_name="Show Any",
            category="xenodes/utils",
            inputs=[
                io.MatchType.Input("value", template=template, display_name="any"),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="any"),
            ],
            is_output_node=True,
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        value = kwargs.get("value")

        hidden = getattr(cls, "hidden", None)
        unique_id = str(hidden.unique_id) if (hidden and hidden.unique_id is not None) else None
        workflow = (hidden.extra_pnginfo or {}).get("workflow") if (hidden and hidden.extra_pnginfo) else None
        nodes = workflow.get("nodes", []) if isinstance(workflow, dict) else []
        definitions = workflow.get("definitions", {}) if isinstance(workflow, dict) else {}

        # If input is None (e.g. upstream muted), fall back to the saved workflow widget value
        if value is None and unique_id:
            cached_val = cls.get_workflow_widget_value(nodes, unique_id, definitions)
            if isinstance(cached_val, str):
                value = cached_val

        text_str = _format_value(value)

        # Sync preview text into workflow metadata so saved images retain the latest text
        if unique_id and nodes:
            if not cls.mutate_workflow_data(nodes, unique_id, text_str, definitions):
                logging.warning(f"[XENodes.ShowAny] Failed to update workflow data for node {unique_id}")

        return io.NodeOutput(value, ui={"text": [text_str]})

    @staticmethod
    def find_target_node(nodes: list, target_id: str, definitions: dict | None = None) -> dict | None:
        """
        Recursively searches the workflow graph for a node by hierarchical ID (e.g., "7:1").
        """
        current_id, _, remaining_id = str(target_id).partition(":")

        for node in nodes:
            if not isinstance(node, dict) or str(node.get("id")) != current_id:
                continue
            if not remaining_id:
                return node
            for sub_nodes in ShowAnyNode._get_sub_nodes(node, node.get("type"), definitions):
                found = ShowAnyNode.find_target_node(sub_nodes, remaining_id, definitions)
                if found is not None:
                    return found
        return None

    @staticmethod
    def get_workflow_widget_value(nodes: list, target_id: str, definitions: dict | None = None) -> str | None:
        target_node = ShowAnyNode.find_target_node(nodes, target_id, definitions)
        wv = target_node.get("widgets_values") if target_node else None
        return wv[0] if isinstance(wv, list) and wv else None

    @staticmethod
    def mutate_workflow_data(nodes: list, target_id: str, new_text: str, definitions: dict | None = None) -> bool:
        target_node = ShowAnyNode.find_target_node(nodes, target_id, definitions)
        if not target_node:
            return False
        wv = target_node.get("widgets_values")
        if isinstance(wv, list) and wv:
            wv[0] = new_text
        else:
            target_node["widgets_values"] = [new_text]
        return True

    @staticmethod
    def _get_sub_nodes(node: dict, node_type: str | None, definitions: dict | None) -> list[list[dict]]:
        candidates: list[list[dict]] = []

        def _extract_nodes(container: object) -> None:
            if isinstance(container, list):
                candidates.append(container)
            elif isinstance(container, dict) and isinstance(container.get("nodes"), list):
                candidates.append(container["nodes"])

        for key in ("nodes", "inner_nodes", "subgraph"):
            _extract_nodes(node.get(key))

        props = node.get("properties")
        if isinstance(props, dict):
            _extract_nodes(props.get("subgraph"))
            _extract_nodes(props.get("nodes"))

        if isinstance(definitions, dict) and node_type:
            _extract_nodes(definitions.get(node_type))
            subgraphs = definitions.get("subgraphs")
            if isinstance(subgraphs, list):
                for sg_def in subgraphs:
                    if isinstance(sg_def, dict) and (sg_def.get("id") == node_type or sg_def.get("name") == node_type):
                        _extract_nodes(sg_def)
                        break
            elif isinstance(subgraphs, dict):
                _extract_nodes(subgraphs.get(node_type))

        return candidates


class ShowAnyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ShowAnyNode]


async def comfy_entrypoint() -> ShowAnyExtension:
    return ShowAnyExtension()
