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
            nodeType.prototype.drawSlots = function(ctx, options) {
                if (this.properties?.hidden_connections) return; // Skip all slot dot drawing
                if (drawSlots) return drawSlots.call(this, ctx, options);
            };

            // Override onRemoved to clean up the dynamic style element
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
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
                            btn.style.backgroundColor = "#444444"; // active color
                            btn.style.color = "white";
                            btn.style.borderColor = "#666666";
                        } else {
                            btn.style.backgroundColor = "#252525"; // inactive color
                            btn.style.color = "#888";
                            btn.style.borderColor = "#333";
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

                    // Force Vue to notice the change in the widget options
                    if (this.setDirtyCanvas) {
                        this.setDirtyCanvas(true, true);
                    }
                    if (app.canvas && app.canvas.setDirty) {
                        app.canvas.setDirty(true, true);
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
                        let label = `${i}`;

                        const link = app.graph.links[inputSlot.link];
                        if (link) {
                            const originNode = app.graph.getNodeById(link.origin_id);
                            if (originNode) {
                                label = originNode.title || originNode.type;
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

                    app.graph.setDirtyCanvas(true, true);
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

                // Sync logic to keep Select and UI buttons in sync
                const orgCallback = selectWidget.callback;
                selectWidget.callback = function (v) {
                    if (orgCallback) orgCallback.apply(this, arguments);
                    updateButtons();
                    if (app.canvas && app.canvas.setDirty) {
                        app.canvas.setDirty(true, true);
                    }
                };

                // Watch for value changes via property access
                let lastVal = selectWidget.value;
                const checkValueChange = () => {
                    if (selectWidget.value !== lastVal) {
                        lastVal = selectWidget.value;
                        updateButtons();
                    }
                    requestAnimationFrame(checkValueChange);
                };
                requestAnimationFrame(checkValueChange);

                let cleanupScheduled = false;
                const scheduleCleanup = () => {
                    if (cleanupScheduled) return;
                    cleanupScheduled = true;
                    setTimeout(() => {
                        cleanupScheduled = false;
                        
                        // Step 1: Collect current state and identify connected slots
                        const allInputSlots = (this.inputs || []).map((inp, idx) => ({ inp, idx }))
                            .filter(({ inp }) => inp.name && inp.name.startsWith("input_"));

                        // Separate connected from unconnected
                        const connectedSlots = allInputSlots.filter(({ inp }) => inp.link != null);
                        
                        // We need: connected slots (compacted) + exactly one spare at the end.
                        // Step 2: Remove surplus unconnected slots — keep only 1 spare.
                        const unconnectedSlots = allInputSlots.filter(({ inp }) => inp.link == null);
                        // Remove all but 1 unconnected slot (remove from highest index first to avoid shift issues)
                        const toRemove = unconnectedSlots.slice(1); // keep the first one as spare
                        for (let i = toRemove.length - 1; i >= 0; i--) {
                            this.removeInput(toRemove[i].idx);
                        }
                        
                        // Step 3: Compact — move all connected slots to the front,
                        // except this may conflict with Autogrow internals. 
                        // The safest approach is just to rename sequentially.
                        // Re-read inputs after removal
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

                        // Step 4: Adjust select index  
                        const newConnectedCount = (this.inputs || []).filter(s => s.name && s.name.startsWith("input_") && s.link != null).length;
                        if (selectWidget.value >= newConnectedCount) {
                            selectWidget.value = Math.max(0, newConnectedCount - 1);
                        }

                        rebuildButtons();
                    }, 50);
                };

                const onConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function (type, index, connected, link_info) {
                    const res = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                    scheduleCleanup();
                    return res;
                };

                // Initial build
                setTimeout(rebuildButtons, 100);

                return r;
            };
        }
    }
});
