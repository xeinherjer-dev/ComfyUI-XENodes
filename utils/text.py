import re
from datetime import datetime

def format_date(pattern: str, dt: datetime) -> str:
    # Matches yyyy, yy, dd, d, MM, M, hh, h, mm, m, ss, s
    token_rx = re.compile(r"yyyy|yy|dd|d|MM|M|hh|h|mm|m|ss|s")
    
    def repl(m):
        t = m.group(0)
        if t == "yyyy":
            return f"{dt.year:04d}"
        elif t == "yy":
            return f"{dt.year % 100:02d}"
        elif t == "MM":
            return f"{dt.month:02d}"
        elif t == "M":
            return f"{dt.month}"
        elif t == "dd":
            return f"{dt.day:02d}"
        elif t == "d":
            return f"{dt.day}"
        elif t == "hh":
            return f"{dt.hour:02d}"
        elif t == "h":
            return f"{dt.hour}"
        elif t == "mm":
            return f"{dt.minute:02d}"
        elif t == "m":
            return f"{dt.minute}"
        elif t == "ss":
            return f"{dt.second:02d}"
        elif t == "s":
            return f"{dt.second}"
        return t

    return token_rx.sub(repl, pattern)

def get_node_display_name(class_type: str) -> str:
    try:
        import nodes
        if hasattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS"):
            if class_type in nodes.NODE_DISPLAY_NAME_MAPPINGS:
                return nodes.NODE_DISPLAY_NAME_MAPPINGS[class_type]
    except Exception:
        pass
    return class_type

def get_widget_value(node_title_or_sr: str, widget_name: str, prompt=None, extra_pnginfo=None) -> str | None:
    if not prompt:
        return None

    node_id = None
    node_obj = None
    if extra_pnginfo and "workflow" in extra_pnginfo:
        workflow = extra_pnginfo["workflow"]
        if "nodes" in workflow:
            for node in workflow["nodes"]:
                title = node.get("title")
                node_type = node.get("type")
                sr_name = node.get("properties", {}).get("Node name for S&R")
                
                # Check all possible names that could refer to this node
                candidates = []
                if title:
                    candidates.append(title)
                if node_type:
                    candidates.append(node_type)
                    display_name = get_node_display_name(node_type)
                    if display_name:
                        candidates.append(display_name)
                    
                    # Hardcoded overrides for built-in PrimitiveString display names (e.g. "Text String")
                    if node_type == "PrimitiveString":
                        candidates.extend(["Text String", "Text", "String"])
                    elif node_type == "PrimitiveStringMultiline":
                        candidates.extend(["Text (Multiline)", "Text String Multiline", "Multiline Text"])
                if sr_name:
                    candidates.append(sr_name)
                
                if node_title_or_sr in candidates:
                    node_id = str(node.get("id"))
                    node_obj = node
                    break

    if not node_id:
        if node_title_or_sr in prompt:
            node_id = node_title_or_sr

    # Try to get value directly from workflow node widgets (highly reliable, works for PrimitiveNode)
    if node_obj:
        widgets = node_obj.get("widgets")
        if widgets:
            for w in widgets:
                if isinstance(w, dict) and w.get("name") == widget_name:
                    val = w.get("value")
                    if val is not None:
                        return str(val)

    # Fallback: get from prompt inputs
    if node_id and node_id in prompt:
        node_info = prompt[node_id]
        inputs = node_info.get("inputs", {})
        if widget_name in inputs:
            val = inputs[widget_name]
            if isinstance(val, list):
                return None
            return str(val)

    return None

def apply_text_replacements(value: str, prompt=None, extra_pnginfo=None) -> str:
    if not isinstance(value, str):
        return value

    def replace(match):
        text = match.group(1)
        split = text.split('.')
        if len(split) != 2:
            if split[0].startswith('date:'):
                fmt = split[0][5:]
                return format_date(fmt, datetime.now())
            return match.group(0)

        node_name, widget_name = split[0], split[1]
        val = get_widget_value(node_name, widget_name, prompt, extra_pnginfo)
        if val is not None:
            # Clean up filename invalid characters
            clean_val = re.sub(r'[/?<>\\:*|"\x00-\x1F\x7F]', '_', val)
            return clean_val
        
        return match.group(0)

    return re.sub(r"%([^%]+)%", replace, value)
