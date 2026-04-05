import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XENodes.DynamicComboSelector";
const NODE_NAME = "XENodes.DynamicComboSelector";

app.registerExtension({
    name: EXTENSION_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
            
            // Force the first output pin to be exactly "COMBO" type in LiteGraph to prevent any connections
            if (this.outputs && this.outputs.length > 0) {
                this.outputs[0].type = "COMBO";
                this.outputs[0].name = "COMBO";
            }
            if (this.outputs && this.outputs.length > 1) {
                this.outputs[1].name = "STRING";
            }

            // hide the hidden_list widget properly
            let hiddenWidget = null;
            if (this.widgets) {
                const hiddenIndex = this.widgets.findIndex(w => w.name === "hidden_list");
                if (hiddenIndex !== -1) {
                    hiddenWidget = this.widgets[hiddenIndex];
                    hiddenWidget.type = "hidden";
                    hiddenWidget.hidden = true;
                    hiddenWidget.options = hiddenWidget.options || {};
                    hiddenWidget.options.hidden = true;
                    hiddenWidget.computeSize = () => [0, 0];
                    
                    // Force reactivity hack
                    this.widgets.splice(hiddenIndex, 1);
                    this.widgets.splice(hiddenIndex, 0, hiddenWidget);
                }
            }

            const indexWidget = this.widgets?.find(w => w.name === "index");

            // Add combo widget for visualization and manual selection
            const comboWidget = this.comboWidget = this.addWidget("combo", "select item", "", (val) => {
                // When combo changes, update index widget if possible
                if (comboWidget.options && comboWidget.options.values) {
                    const idx = comboWidget.options.values.indexOf(val);
                    if (idx !== -1 && indexWidget) {
                        indexWidget.value = idx;
                    }
                }
            }, { values: ["(No list connected)"] });

            // Ensure combo widget doesn't serialize its own value to the backend 
            // since we only need index and hidden_list for the Python execution
            comboWidget.serialize = false;

            // Hook into index widget to update combo box when index changes manually
            if (indexWidget) {
                const originalIndexCallback = indexWidget.callback;
                indexWidget.callback = function (val) {
                    if (originalIndexCallback) originalIndexCallback.apply(this, arguments);
                    if (comboWidget.options && comboWidget.options.values) {
                        const len = comboWidget.options.values.length;
                        if (len > 0 && typeof val === "number") {
                            // Valid index wrapping
                            const wrapIdx = Math.floor(val) % len;
                            const actualIdx = wrapIdx < 0 ? wrapIdx + len : wrapIdx;
                            comboWidget.value = comboWidget.options.values[actualIdx];
                        }
                    }
                };
            }

            this.updateListPreview = function (listValues) {
                if (hiddenWidget) {
                    hiddenWidget.value = JSON.stringify(listValues || []);
                }

                if (!listValues || listValues.length === 0) {
                    comboWidget.options.values = ["(No list connected)"];
                    comboWidget.value = "(No list connected)";
                } else {
                    comboWidget.options.values = listValues;
                    // Try to sync with current index
                    const currIndex = indexWidget ? indexWidget.value : 0;
                    if (typeof currIndex === "number") {
                        const wrapIdx = Math.floor(currIndex) % listValues.length;
                        const actualIdx = wrapIdx < 0 ? wrapIdx + listValues.length : wrapIdx;
                        comboWidget.value = listValues[actualIdx];
                    }
                }

                if (this.computeSize && this.setSize) {
                    const newSize = this.computeSize([this.size[0], this.size[1]]);
                    this.setSize(newSize);
                }
                app.canvas?.setDirty(true, true);
            };

            this.updateListPreview([]);

            // Extra check for initial connection state for brand new nodes
            setTimeout(() => {
                const isIndexConnected = this.inputs?.some(i => i.name === "index" && i.link != null);
                if (this.comboWidget) {
                    this.comboWidget.disabled = !!isIndexConnected;
                }
            }, 10);

            const originalOnConfigure = this.onConfigure;
            this.onConfigure = function(info) {
                if (originalOnConfigure) {
                    originalOnConfigure.apply(this, arguments);
                }
                if (hiddenWidget && hiddenWidget.value) {
                    try {
                        const listValues = JSON.parse(hiddenWidget.value);
                        if (listValues && listValues.length > 0) {
                            // Update UI but skip rewriting the hidden widget to itself if possible,
                            // or just call updateListPreview which does it safely anyway.
                            const oldVal = hiddenWidget.value;
                            this.updateListPreview(listValues);
                            hiddenWidget.value = oldVal; // restore exact serialized string just in case
                        }
                    } catch (e) {}
                }
                const isIndexConnected = this.inputs?.some(i => i.name === "index" && i.link != null);
                if (this.comboWidget) {
                    this.comboWidget.disabled = !!isIndexConnected;
                }
            };

            return result;
        };

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, slotIndex, isConnected, link_info, _ioSlot) {
            const result = originalOnConnectionsChange 
                ? originalOnConnectionsChange.apply(this, arguments) 
                : undefined;

            if (type === LiteGraph.INPUT) {
                const input = this.inputs[slotIndex];
                if (input && input.name === "index" && this.comboWidget) {
                    this.comboWidget.disabled = isConnected;
                }
            }

            if (type === LiteGraph.OUTPUT && slotIndex === 0) {
                const linkId = typeof link_info === "object" ? link_info.id : link_info;
                const link = isConnected ? app.graph.links[linkId] : null;

                if (!link) {
                    if (this.updateListPreview) this.updateListPreview([]);
                    return result;
                }

                const targetNode = app.graph.getNodeById(link.target_id);
                if (!targetNode) return result;

                const targetSlot = link.target_slot;
                const targetInput = targetNode.inputs[targetSlot];
                if (!targetInput) return result;

                // Sometimes the widget is named differently from the input name, handle safely
                const widgetName = targetInput.widget ? targetInput.widget.name : targetInput.name;
                const targetWidget = targetNode.widgets ? targetNode.widgets.find(w => w.name === widgetName) : null;

                let listValues = null;

                if (targetWidget && targetWidget.options && Array.isArray(targetWidget.options.values)) {
                    listValues = targetWidget.options.values;
                } else if (targetInput.widget && targetInput.widget.options && Array.isArray(targetInput.widget.options.values)) {
                    listValues = targetInput.widget.options.values;
                } else if (targetNode.constructor.nodeData && targetNode.constructor.nodeData.input) {
                    const inputsDef = {
                        ...targetNode.constructor.nodeData.input.required,
                        ...targetNode.constructor.nodeData.input.optional
                    };
                    const inputDef = inputsDef[targetInput.name];
                    if (inputDef && Array.isArray(inputDef[0])) {
                        listValues = inputDef[0];
                    }
                }

                if (listValues && this.updateListPreview) {
                    this.updateListPreview(listValues);
                }
            }

            return result;
        };
    }
});
