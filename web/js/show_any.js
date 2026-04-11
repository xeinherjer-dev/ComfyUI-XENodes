import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "XENodes.ShowAny",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "XENodes.ShowAny") {
            /**
             * Populate text preview widgets
             * @param {string|string[]} text 
             */
            function populate(text) {
                if (this.widgets) {
                    for (let i = this.widgets.length - 1; i >= 0; i--) {
                        if (this.widgets[i].is_xenode_preview) {
                            if (this.widgets[i].inputEl) {
                                this.widgets[i].inputEl.remove();
                            }
                            this.widgets[i].onRemove?.();
                            this.widgets.splice(i, 1);
                        }
                    }
                }

                let v = text;
                if (!v) v = [];
                if (!Array.isArray(v)) v = [v];

                for (let list of v) {
                    if (!Array.isArray(list)) list = [list];
                    for (const l of list) {
                        try {
                            const w = ComfyWidgets["STRING"](this, "", ["STRING", { multiline: true }], app).widget;
                            w.inputEl.readOnly = true;
                            w.inputEl.style.opacity = 0.8; 
                            w.value = l;
                            w.is_xenode_preview = true;
                            w.serialize_ignore = true;
                        } catch (e) {
                            console.error("[XENodes] Widget creation failed:", e);
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
                
                if (!message) return;

                const texts = message.text || [];
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
        }
    }
});
