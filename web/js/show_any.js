import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "XENodes.ShowAny",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "XENodes.ShowAny") {
            const TAG = (node) => `[XENodes.ShowAny #${node?.id ?? "?"} "${node?.title || "Show Any"}"]`;

            /**
             * Populate text preview widgets
             * @param {string|string[]} text 
             */
            function populate(text) {
                if (this.widgets) {
                    for (let i = this.widgets.length - 1; i >= 0; i--) {
                        if (this.widgets[i].is_xenode_preview) {
                            if (this.widgets[i].element) {
                                this.widgets[i].element.remove();
                            }
                            this.widgets[i].onRemove?.();
                            this.widgets.splice(i, 1);
                        }
                    }
                }

                let v = text;
                if (!v) v = [];
                if (!Array.isArray(v)) v = [v];

                let idx = 0;
                for (let list of v) {
                    if (!Array.isArray(list)) list = [list];
                    for (const l of list) {
                        try {
                            const widgetName = `text_preview_${idx++}`;
                            const w = ComfyWidgets["STRING"](this, widgetName, ["STRING", { multiline: true }], app)?.widget;
                            if (!w) {
                                console.error(`${TAG(this)} ComfyWidgets["STRING"] returned null/undefined for ${widgetName}`);
                                continue;
                            }
                            if (w.element) {
                                w.element.readOnly = true;
                                w.element.style.opacity = 0.8;
                            }
                            w.value = l;
                            w.is_xenode_preview = true;
                            w.serialize_ignore = true;
                        } catch (e) {
                            console.error(`${TAG(this)} Preview widget creation failed:`, e, "item was:", l);
                        }
                    }
                }
                
                requestAnimationFrame(() => {
                    if (this.computeSize && this.setSize) {
                        const sz = this.computeSize();
                        this.setSize([Math.max(this.size[0], sz[0]), Math.max(this.size[1], sz[1])]);
                        app.graph.setDirtyCanvas(true, true);
                    }
                });
            }

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);
                
                if (!message) {
                    console.warn(`${TAG(this)} onExecuted received null/undefined message!`);
                    return;
                }

                const texts = message.text || [];
                if (!message.text) {
                    console.warn(`${TAG(this)} onExecuted message has no 'text' property:`, message);
                }
                populate.call(this, texts);
            };

            const VALUES = Symbol();
            const configure = nodeType.prototype.configure;
            nodeType.prototype.configure = function() {
                this[VALUES] = arguments[0]?.widgets_values;
                return configure?.apply(this, arguments);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                onConfigure?.apply(this, arguments);

                const vals = this[VALUES];
                if (vals && vals.length) {
                    const dynamicVals = vals.filter(v => typeof v === "string");
                    if (dynamicVals.length > 0) {
                        requestAnimationFrame(() => populate.call(this, dynamicVals));
                    }
                }
            };

            const serialize = nodeType.prototype.serialize;
            nodeType.prototype.serialize = function() {
                const data = serialize?.apply(this, arguments);
                const hasPreview = this.widgets?.some(w => w.is_xenode_preview && w.value);
                if (hasPreview && (!data?.widgets_values || data.widgets_values.length === 0)) {
                    console.warn(`${TAG(this)} Preview text exists on widget but widgets_values is empty during serialize!`, {
                        node_id: this.id,
                        widgets: this.widgets
                    });
                }
                return data;
            };
        }
    }
});
