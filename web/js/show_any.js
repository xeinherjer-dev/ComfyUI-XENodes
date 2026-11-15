import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const PREVIEW_WIDGET_NAME = "text_preview";

function formatPreviewText(text) {
    if (text == null) return "";
    const items = Array.isArray(text) ? text.flat(Infinity) : [text];
    return items
        .filter((item) => item != null)
        .map((item) => (typeof item === "string" ? item : String(item)))
        .join("\n");
}

app.registerExtension({
    name: "XENodes.ShowAny",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "XENodes.ShowAny") return;

        function ensurePreviewWidget(node) {
            let widget = node.widgets?.find((w) => w.is_xenode_preview);
            if (!widget) {
                widget = ComfyWidgets["STRING"](
                    node,
                    PREVIEW_WIDGET_NAME,
                    ["STRING", { multiline: true }],
                    app
                )?.widget;
                if (widget) {
                    if (widget.element) {
                        widget.element.readOnly = true;
                        widget.element.style.opacity = "0.8";
                    }
                    widget.is_xenode_preview = true;
                }
            }
            return widget;
        }

        function updatePreview(node, text) {
            const widget = ensurePreviewWidget(node);
            if (!widget) return;
            widget.value = formatPreviewText(text);

            requestAnimationFrame(() => {
                if (node.computeSize && node.setSize) {
                    const sz = node.computeSize();
                    node.setSize([
                        Math.max(node.size?.[0] ?? 0, sz[0]),
                        Math.max(node.size?.[1] ?? 0, sz[1]),
                    ]);
                    (node.graph || app.graph)?.setDirtyCanvas?.(true, true);
                }
            });
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const res = onNodeCreated?.apply(this, arguments);
            ensurePreviewWidget(this);
            return res;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (message?.text !== undefined) {
                updatePreview(this, message.text);
            }
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            const vals = info?.widgets_values;
            if (Array.isArray(vals) && vals.length > 0) {
                const strings = vals.filter((v) => typeof v === "string");
                if (strings.length > 0) {
                    updatePreview(this, strings);
                }
            }
        };

        const serialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function () {
            const data = serialize?.apply(this, arguments);
            const widget = this.widgets?.find((w) => w.is_xenode_preview);
            if (
                data &&
                widget?.value &&
                (!Array.isArray(data.widgets_values) || data.widgets_values.length === 0)
            ) {
                data.widgets_values = [widget.value];
            }
            return data;
        };
    },
});
