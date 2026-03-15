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
                container.style.border = "0px";
                container.style.alignItems = "stretch";
                container.style.pointerEvents = "auto";

                // Utility to measure text width
                const ctx = document.createElement("canvas").getContext("2d");
                ctx.font = "bold 11px sans-serif";

                // Function to update button styles based on current selection
                const updateButtons = () => {
                    const currentValue = selectWidget.value;
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
                const rebuildButtons = () => {
                    while (container.firstChild) {
                        container.removeChild(container.firstChild);
                    }

                    const inputSlots = (this.inputs || []).filter(input => input.name.includes("input_"));
                    maxLabelWidth = 100;

                    for (let i = 0; i < inputSlots.length; i++) {
                        const inputSlot = inputSlots[i];
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
                        btn.style.minHeight = "28px";

                        btn.onclick = (e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            if (selectWidget.value !== i) {
                                selectWidget.value = i;
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

                // Hide the original numeric widget
                selectWidget.type = "hidden";
                if (selectWidget.element) selectWidget.element.style.display = "none";
                selectWidget.hidden = true;

                // Add the container as a DOM widget
                // In Node 2.0 / LiteGraph, dom widgets can have their computeSize.
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

                // Override computeSize only to ensure width is sufficient for labels,
                // but rely on super/original for the rest of the layout logic.
                const originalComputeSize = this.computeSize;
                this.computeSize = function () {
                    const size = originalComputeSize ? originalComputeSize.apply(this, arguments) : [200, 100];
                    if (this.flags.collapsed) return size;
                    
                    const buttonCount = container.children.length;
                    if (buttonCount === 0) {
                        // Compact mode for no buttons
                        const SLOT_HEIGHT = 22;
                        const HEADER_HEIGHT = 46;
                        const inputCount = (this.inputs ? this.inputs.length : 0);
                        const outputCount = (this.outputs ? this.outputs.length : 0);
                        const maxSlots = Math.max(inputCount, outputCount, 1);
                        size[1] = maxSlots * SLOT_HEIGHT + HEADER_HEIGHT;
                    }

                    // Update width if labels are long
                    size[0] = Math.max(size[0], maxLabelWidth + 60);
                    return size;
                };

                // Guarded sync logic to prevent infinite recursion
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
