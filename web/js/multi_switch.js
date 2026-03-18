import { app } from "../../../scripts/app.js";

// Multi-Switch Node UI Extension for Node 2.0 (ComfyUI-xe-nodes)
// This script converts the 'select' integer input into a list of clickable buttons.

app.registerExtension({
    name: "XeNodes.MultiSwitch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "MultiSwitch") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            // Add new property for hiding connections
            nodeType.prototype.properties = nodeType.prototype.properties || {};
            if (nodeType.prototype.properties.hidden_connections === undefined) {
                nodeType.prototype.properties.hidden_connections = false;
            }

            // Add context menu option
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function (_, options) {
                if (getExtraMenuOptions) getExtraMenuOptions.apply(this, arguments);

                options.push({
                    content: this.properties.hidden_connections ? "Show Connections" : "Hide Connections",
                    callback: () => {
                        this.properties.hidden_connections = !this.properties.hidden_connections;
                        if (this.rebuildButtons) this.rebuildButtons();
                        this.setSize(this.computeSize());
                        this.setDirtyCanvas(true, true);
                    }
                });
            };

            // Override drawSlots to skip dot rendering when connections are hidden
            const drawSlots = nodeType.prototype.drawSlots;
            nodeType.prototype.drawSlots = function (ctx, options) {
                if (this.properties?.hidden_connections) return; // Skip all slot dot drawing
                if (drawSlots) return drawSlots.call(this, ctx, options);
            };

            // Override onRemoved to clean up the dynamic style element
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                const styleEl = document.getElementById(`multi-switch-style-${this.id}`);
                if (styleEl) styleEl.remove();
                if (onRemoved) {
                    onRemoved.apply(this, arguments);
                }
            };

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Find the 'select' widget
                const selectWidget = this.widgets.find(w => w.name === "select");
                if (!selectWidget) return r;

                // Store the original type so we can restore it when showing connections
                const originalSelectType = selectWidget.type || "customtext";

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
                container.style.border = "0px";
                container.style.alignItems = "stretch";
                container.style.pointerEvents = "auto";

                // Utility to measure text width
                const ctx = document.createElement("canvas").getContext("2d");
                ctx.font = "bold 11px sans-serif";

                // Function to update button styles based on current selection
                const updateButtons = () => {
                    const currentValue = Math.floor(selectWidget.value || 0);
                    Array.from(container.children).forEach((btn) => {
                        const idx = parseInt(btn.dataset.index);
                        if (idx === currentValue) {
                            btn.style.backgroundColor = "#444444"; // Active color
                            btn.style.color = "white";
                            btn.style.border = "1px solid #666666";
                            btn.style.borderLeft = "4px solid #4CAF50"; // ComfyUI's green accent
                            btn.style.fontWeight = "900";
                        } else {
                            btn.style.backgroundColor = "#252525"; // Inactive color
                            btn.style.color = "#888";
                            btn.style.border = "1px solid #333";
                            btn.style.borderLeft = "1px solid #333";
                            btn.style.fontWeight = "bold";
                        }
                    });
                };

                let maxLabelWidth = 100;

                // Function to rebuild buttons based on connected inputs
                const rebuildButtons = this.rebuildButtons = () => {
                    while (container.firstChild) {
                        container.removeChild(container.firstChild);
                    }

                    // Handle Vue Node 2.0 DOM slots hiding via CSS injection
                    const styleId = `multi-switch-style-${this.id}`;
                    let styleEl = document.getElementById(styleId);
                    if (!styleEl) {
                        styleEl = document.createElement("style");
                        styleEl.id = styleId;
                        document.head.appendChild(styleEl);
                    }

                    // Toggle the select widget visibility too
                    if (!selectWidget.options) selectWidget.options = {};

                    if (this.properties.hidden_connections) {
                        selectWidget.hidden = true;
                        selectWidget.options.hidden = true;
                        selectWidget.type = "hidden";

                        let topOffset = 46; // Matches body's top padding (~46px incl header)
                        let css = `
                            [data-node-id="${this.id}"] .lg-slot {
                                position: absolute !important;
                                opacity: 0 !important;
                                pointer-events: none !important;
                                width: 10px !important;
                                height: 32px !important;
                                margin: 0 !important; 
                                padding: 0 !important;
                            }
                            [data-node-id="${this.id}"] .lg-slot--output {
                                right: 0 !important;
                                top: ${topOffset}px !important;
                            }
                        `;
                        let childIndex = 1;
                        (this.inputs || []).forEach(input => {
                            if (input.name && input.name.startsWith("input_")) {
                                input.label = "";
                                css += `[data-node-id="${this.id}"] .lg-slot--input:nth-child(${childIndex}) { left: -5px !important; top: ${topOffset}px !important; }\n`;
                                topOffset += 32;
                            } else {
                                css += `[data-node-id="${this.id}"] .lg-slot--input:nth-child(${childIndex}) { left: -5px !important; top: 46px !important; }\n`;
                            }
                            childIndex++;
                        });
                        (this.outputs || []).forEach(output => {
                            output.label = "";
                        });
                        styleEl.innerHTML = css;
                    } else {
                        selectWidget.hidden = false;
                        selectWidget.options.hidden = false;
                        selectWidget.type = originalSelectType; // Restore normal rendering type
                        styleEl.innerHTML = "";
                    }

                    const inputSlots = (this.inputs || []).filter(input => input.name.includes("input_"));
                    maxLabelWidth = 100;

                    for (let i = 0; i < inputSlots.length; i++) {
                        const inputSlot = inputSlots[i];
                        // Restore labels when not hidden
                        if (!this.properties.hidden_connections && inputSlot.name.startsWith("input_")) {
                            inputSlot.label = inputSlot.name;
                        }

                        if (inputSlot.link == null) continue;

                        const btn = document.createElement("button");
                        btn.dataset.index = i;

                        // Visualize the index number, default to (Empty)
                        let label = `[${i}] (Empty)`;

                        const link = app.graph.links[inputSlot.link];
                        if (link) {
                            const originNode = app.graph.getNodeById(link.origin_id);
                            if (originNode) {
                                // Combine origin node title and index
                                label = `[${i}] ${originNode.title || originNode.type}`;
                            }
                        }

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
                        btn.style.outline = "none";
                        btn.style.width = "100%";
                        btn.style.whiteSpace = "nowrap";
                        btn.style.overflow = "hidden";
                        btn.style.textOverflow = "ellipsis";
                        btn.style.minHeight = "32px";

                        btn.onclick = (e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            if (selectWidget.value !== i) {
                                selectWidget.value = i;
                                if (selectWidget.callback) {
                                    selectWidget.callback(i);
                                }
                                if (this.change) {
                                    this.change();
                                }
                                this.setDirtyCanvas(true, true);
                            }
                        };

                        container.appendChild(btn);
                    }
                    updateButtons();

                    if (container.children.length === 0) {
                        container.style.display = "none";
                    } else {
                        container.style.display = "flex";
                    }

                    // --- Auto-resize handling ---
                    if (this.size && this.computeSize) {
                        const targetSize = this.computeSize();
                        this.size[0] = Math.max(this.size[0], targetSize[0]);
                        this.size[1] = targetSize[1];
                    }

                    if (app.canvas && app.canvas.setDirty) {
                        app.canvas.setDirty(true, true);
                    }
                };

                // Add the container as a DOM widget
                const domWidget = this.addDOMWidget("select_buttons", "BUTTONS", container, {
                    getValue() { return selectWidget.value; },
                    setValue(v) {
                        selectWidget.value = v;
                    }
                });

                // Delegate specific widget height to the widget itself
                domWidget.computeSize = (width) => {
                    const buttonCount = container.children.length;
                    return [width, (buttonCount * 32) + 10];
                };

                // Override computeSize
                const originalComputeSize = this.computeSize;
                this.computeSize = function () {
                    const size = originalComputeSize ? originalComputeSize.apply(this, arguments) : [200, 100];
                    if (this.flags.collapsed) return size;

                    const buttonCount = container.children.length;
                    const HEADER_HEIGHT = 46;

                    if (this.properties.hidden_connections) {
                        // "Hide Connections" mode: size by buttons only, no slot rows
                        size[1] = buttonCount * 32 + HEADER_HEIGHT + 10;
                        size[0] = Math.max(120, maxLabelWidth + 40);
                        // Force widgets to the top, ignoring slot layout spacing
                        this.widgets_up = true;
                        this.widgets_start_y = 0;
                    } else {
                        const SLOT_HEIGHT = 22;
                        const inputCount = (this.inputs ? this.inputs.length : 0);
                        const outputCount = (this.outputs ? this.outputs.length : 0);
                        const maxSlots = Math.max(inputCount, outputCount, 1);
                        size[1] = Math.max(buttonCount * 32 + HEADER_HEIGHT + 10, maxSlots * SLOT_HEIGHT + HEADER_HEIGHT);
                        size[0] = Math.max(size[0], maxLabelWidth + 60);
                        // Restore standard widget placement
                        this.widgets_up = undefined;
                        this.widgets_start_y = undefined;
                    }

                    return size;
                };

                // --- Clamp resize bounds ---
                const onResize = this.onResize;
                this.onResize = function (size) {
                    if (onResize) onResize.apply(this, arguments);
                    const minSize = this.computeSize();
                    size[0] = Math.max(size[0], minSize[0]);
                    size[1] = Math.max(size[1], minSize[1]);
                };

                // Sync logic to keep Select and UI buttons in sync
                const orgCallback = selectWidget.callback;
                selectWidget.callback = function (v) {
                    if (orgCallback) orgCallback.apply(this, arguments);
                    updateButtons();
                    if (app.canvas && app.canvas.setDirty) {
                        app.canvas.setDirty(true, true);
                    }
                };

                // Synchronous cleanup when connections change
                const onConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function (type, index, connected, link_info) {
                    const res = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;

                    // Step 1: Collect current state and identify connected slots
                    const allInputSlots = (this.inputs || []).map((inp, idx) => ({ inp, idx }))
                        .filter(({ inp }) => inp.name && inp.name.startsWith("input_"));

                    // Step 2: Remove surplus unconnected slots (keep only 1 spare)
                    const unconnectedSlots = allInputSlots.filter(({ inp }) => inp.link == null);
                    const toRemove = unconnectedSlots.slice(1);
                    for (let i = toRemove.length - 1; i >= 0; i--) {
                        this.removeInput(toRemove[i].idx);
                    }

                    // Step 3: Rename sequentially to maintain input_0, input_1, etc.
                    let seqIdx = 0;
                    for (let j = 0; j < (this.inputs || []).length; j++) {
                        const slot = this.inputs[j];
                        if (slot.name && slot.name.startsWith("input_")) {
                            const newName = `input_${seqIdx}`;
                            slot.name = newName;
                            slot.label = newName;
                            seqIdx++;
                        }
                    }

                    // Notice: We completely removed the logic that forces selectWidget.value to 0.
                    // This allows the value to survive workflow loading.

                    rebuildButtons();
                    return res;
                };

                // Initial build
                rebuildButtons();

                return r;
            };
        }
    }
});