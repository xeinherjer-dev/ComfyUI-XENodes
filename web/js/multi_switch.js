import { app } from "../../../scripts/app.js";

// Multi-Switch Node UI Extension for Node 2.0 (ComfyUI-xe-nodes)
// This script converts the 'select' integer input into a list of clickable buttons.

app.registerExtension({
    name: "XeNodes.MultiSwitch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "MultiSwitch") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Find the 'select' widget
                const selectWidget = this.widgets.find(w => w.name === "select");
                if (!selectWidget) return r;

                // Create a container for our buttons
                const container = document.createElement("div");
                container.className = "multi-switch-container";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.gap = "0px";
                container.style.padding = "0px";
                container.style.backgroundColor = "rgba(0,0,0,0.3)";
                container.style.borderRadius = "0px";
                container.style.margin = "0px 0px";
                container.style.border = "0px solid rgba(255,255,255,0.1)";
                container.style.alignItems = "stretch";
                container.style.pointerEvents = "auto";

                // Utility to measure text width
                const ctx = document.createElement("canvas").getContext("2d");
                ctx.font = "bold 11px sans-serif";

                // Function to update button styles based on current selection
                const updateButtons = () => {
                    const currentValue = selectWidget.value;
                    // Note: Since index in container.children might not match selectWidget.value 
                    // because we hide unconnected ones, we store the original index on the button.
                    Array.from(container.children).forEach((btn) => {
                        const idx = parseInt(btn.dataset.index);
                        if (idx === currentValue) {
                            btn.style.backgroundColor = "#444444"; // active color
                            btn.style.color = "white";
                            btn.style.borderColor = "#666666";
                            btn.style.boxShadow = "none";
                        } else {
                            btn.style.backgroundColor = "#252525"; // inactive color
                            btn.style.color = "#888";
                            btn.style.borderColor = "#333";
                            btn.style.boxShadow = "none";
                        }
                    });
                };

                let maxLabelWidth = 100;

                // Function to rebuild buttons based on connected inputs
                const rebuildButtons = () => {
                    while (container.firstChild) {
                        container.removeChild(container.firstChild);
                    }

                    const inputSlots = this.inputs.filter(input => input.name.includes("input_"));
                    maxLabelWidth = 100;

                    for (let i = 0; i < inputSlots.length; i++) {
                        const inputSlot = inputSlots[i];

                        // User requested to hide UI for unconnected inputs
                        if (inputSlot.link == null) continue;

                        const btn = document.createElement("button");
                        btn.dataset.index = i;
                        let label = `${i}`;

                        const link = app.graph.links[inputSlot.link];
                        if (link) {
                            const originNode = app.graph.getNodeById(link.origin_id);
                            if (originNode) {
                                label = originNode.title || originNode.type;
                                // Removed truncation as requested
                            }
                        }

                        // Track max width for computeSize
                        const metrics = ctx.measureText(label);
                        if (metrics.width > maxLabelWidth) maxLabelWidth = metrics.width;

                        btn.innerText = label;
                        btn.title = `Switch to input ${i}`;
                        btn.style.cursor = "pointer";
                        btn.style.border = "1px solid";
                        btn.style.borderRadius = "4px";
                        btn.style.padding = "8px 10px";
                        btn.style.fontSize = "11px";
                        btn.style.fontWeight = "bold";
                        btn.style.textAlign = "left";
                        btn.style.transition = "all 0.1s ease";
                        btn.style.outline = "none";
                        btn.style.width = "100%";
                        btn.style.whiteSpace = "nowrap";
                        btn.style.overflow = "hidden";
                        btn.style.textOverflow = "ellipsis"; // Still use ellipsis if container is too small, but node will try to expand
                        btn.style.minHeight = "28px";

                        btn.onclick = (e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            if (selectWidget.value !== i) {
                                selectWidget.value = i;
                            }
                        };

                        // Hover effects
                        btn.onmouseenter = () => {
                            if (selectWidget.value !== i) {
                                btn.style.backgroundColor = "#353535";
                                btn.style.color = "#fff";
                            }
                        };
                        btn.onmouseleave = () => {
                            updateButtons();
                        };

                        container.appendChild(btn);
                    }
                    updateButtons();
                    if (this.setSize) this.setSize(this.computeSize());
                };

                // Hide the original numeric widget
                selectWidget.type = "hidden";
                if (selectWidget.element) selectWidget.element.style.display = "none";
                selectWidget.hidden = true;

                // Add the container as a DOM widget
                const domWidget = this.addDOMWidget("select_buttons", "BUTTONS", container, {
                    getValue() { return selectWidget.value; },
                    setValue(v) {
                        selectWidget.value = v;
                    }
                });

                const BUTTON_LINE_HEIGHT = 32;
                domWidget.computeSize = (width) => {
                    const buttonCount = container.children.length;
                    const height = (buttonCount * BUTTON_LINE_HEIGHT) + 20;
                    return [width, height];
                };

                // Standard computeSize to accommodate buttons in expanded mode
                const originalComputeSize = this.computeSize;
                this.computeSize = function () {
                    if (this.flags.collapsed) {
                        return originalComputeSize ? originalComputeSize.apply(this, arguments) : [200, 30];
                    }

                    const SLOT_HEIGHT = 22;
                    const HEADER_HEIGHT = 46;
                    const inputCount = (this.inputs ? this.inputs.length : 0);
                    const outputCount = (this.outputs ? this.outputs.length : 0);
                    const maxSlots = Math.max(inputCount, outputCount, 1);
                    const slotsHeight = maxSlots * SLOT_HEIGHT + HEADER_HEIGHT;

                    const buttonCount = container.children.length;
                    const neededButtonsHeight = (buttonCount * BUTTON_LINE_HEIGHT) + 30;

                    const finalHeight = slotsHeight + neededButtonsHeight;

                    // Adjust width based on max label width
                    const finalWidth = Math.max(200, maxLabelWidth + 40);

                    return [finalWidth, finalHeight];
                };

                // --- Sync logic (Simplified and Guarded) ---
                let isUpdating = false;
                const orgCallback = selectWidget.callback;
                selectWidget.callback = function (v) {
                    if (isUpdating) return;
                    isUpdating = true;
                    try {
                        if (orgCallback) orgCallback.apply(this, arguments);
                        updateButtons();
                    } finally {
                        isUpdating = false;
                    }
                };

                let val = selectWidget.value;
                Object.defineProperty(selectWidget, "value", {
                    get() { return val; },
                    set(v) {
                        if (val === v) return;
                        val = v;
                        if (isUpdating) return;
                        isUpdating = true;
                        try {
                            updateButtons();
                            if (this.callback) this.callback(v);
                        } finally {
                            isUpdating = false;
                        }
                    },
                    configurable: true
                });

                const onConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function () {
                    const res = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                    // Slightly longer timeout to ensure app.graph.links is updated
                    setTimeout(rebuildButtons, 30);
                    return res;
                };

                setTimeout(() => {
                    rebuildButtons();
                    if (this.setSize) this.setSize(this.computeSize());
                }, 100);

                return r;
            };
        }
    }
});
