import { app } from "../../../scripts/app.js";

// Modern Multi-Switch Node UI Extension for Node 2.0
// This script converts the 'select' integer input into a list of clickable buttons.

app.registerExtension({
    name: "Comfy.ModernMultiSwitch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "ModernMultiSwitch") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Find the 'select' widget
                const selectWidget = this.widgets.find(w => w.name === "select");
                if (!selectWidget) return r;

                // Create a container for our buttons
                const container = document.createElement("div");
                container.className = "modern-switch-container";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.gap = "2px";
                container.style.padding = "4px";
                container.style.backgroundColor = "rgba(0,0,0,0.3)";
                container.style.borderRadius = "6px";
                container.style.margin = "4px 4px";
                container.style.border = "1px solid rgba(255,255,255,0.1)";
                container.style.alignItems = "stretch";

                // Function to update button styles based on current selection
                const updateButtons = () => {
                    const currentValue = selectWidget.value;
                    Array.from(container.children).forEach((btn, idx) => {
                        if (idx === currentValue) {
                            btn.style.backgroundColor = "#444de4"; // active color
                            btn.style.color = "white";
                            btn.style.borderColor = "#666eff";
                            btn.style.boxShadow = "0 0 8px rgba(68, 77, 228, 0.4)";
                        } else {
                            btn.style.backgroundColor = "#252525"; // inactive color
                            btn.style.color = "#888";
                            btn.style.borderColor = "#333";
                            btn.style.boxShadow = "none";
                        }
                    });
                };

                // Function to rebuild buttons based on connected inputs
                const rebuildButtons = () => {
                    while (container.firstChild) {
                        container.removeChild(container.firstChild);
                    }

                    // In Node 2.0 Autogrow, inputs are typically named 'inputs.input_x'
                    const inputSlots = this.inputs.filter(input => input.name.includes("input_"));
                    const count = inputSlots.length;

                    for (let i = 0; i < count; i++) {
                        const btn = document.createElement("button");
                        const inputSlot = inputSlots[i];
                        let label = `${i}`;

                        // Try to get a meaningful label from the connected node
                        if (inputSlot && inputSlot.link != null) {
                            const link = app.graph.links[inputSlot.link];
                            if (link) {
                                const originNode = app.graph.getNodeById(link.origin_id);
                                if (originNode) {
                                    label = originNode.title || originNode.type;
                                    if (label.length > 20) label = label.substring(0, 17) + "...";
                                }
                            }
                        }

                        btn.innerText = label;
                        btn.title = `Switch to input ${i}`;
                        btn.style.cursor = "pointer";
                        btn.style.border = "1px solid";
                        btn.style.borderRadius = "4px";
                        btn.style.padding = "6px 10px";
                        btn.style.fontSize = "11px";
                        btn.style.fontWeight = "bold";
                        btn.style.textAlign = "left";
                        btn.style.transition = "all 0.1s ease";
                        btn.style.outline = "none";
                        btn.style.width = "100%";
                        btn.style.whiteSpace = "nowrap";
                        btn.style.overflow = "hidden";
                        btn.style.textOverflow = "ellipsis";

                        btn.onclick = (e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            selectWidget.value = i;
                            if (selectWidget.callback) {
                                selectWidget.callback(i);
                            }
                            updateButtons();
                            app.graph.setDirtyCanvas(true, true);
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
                };

                // Hide the original numeric widget aggressively
                selectWidget.type = "hidden";
                if (selectWidget.element) selectWidget.element.style.display = "none";
                selectWidget.hidden = true;

                // Add the container as a DOM widget
                this.addDOMWidget("select_buttons", "BUTTONS", container, {
                    getValue() { return selectWidget.value; },
                    setValue(v) {
                        selectWidget.value = v;
                        updateButtons();
                    }
                });

                // --- Improved Collapse Handling ---
                // Override onSetCollapsed to handle container visibility
                const onSetCollapsed = this.onSetCollapsed;
                this.onSetCollapsed = function (collapsed) {
                    if (onSetCollapsed) onSetCollapsed.apply(this, arguments);
                    // Ensure the container remains visible
                    container.style.display = "flex";
                };

                // Override computeSize to ensure space for buttons when collapsed
                const computeSize = this.computeSize;
                this.computeSize = function () {
                    let size = computeSize ? computeSize.apply(this, arguments) : [200, 100];
                    if (this.flags.collapsed) {
                        // Estimate height based on number of buttons
                        const buttonCount = container.children.length;
                        const estimatedHeight = 30 + (buttonCount * 26) + 10;
                        return [size[0], estimatedHeight];
                    }
                    return size;
                };

                // Override draw to fix container position when collapsed
                const onDrawForeground = this.onDrawForeground;
                this.onDrawForeground = function (ctx) {
                    if (onDrawForeground) onDrawForeground.apply(this, arguments);

                    if (this.flags.collapsed) {
                        // Force container to be visible and correctly positioned
                        container.style.position = "absolute";
                        container.style.top = "20px";
                        container.style.left = "0px";
                        container.style.width = "100%";
                        container.style.zIndex = "100";
                    } else {
                        container.style.position = "static";
                        container.style.width = "auto";
                    }
                };

                // Synchronize when the select widget changes from external sources
                const orgCallback = selectWidget.callback;
                selectWidget.callback = function () {
                    if (orgCallback) orgCallback.apply(this, arguments);
                    updateButtons();
                };

                // Add property observer to catch all changes to .value
                let val = selectWidget.value;
                Object.defineProperty(selectWidget, "value", {
                    get() { return val; },
                    set(v) {
                        val = v;
                        updateButtons();
                        if (this.callback) this.callback(v);
                    },
                    configurable: true
                });

                // Rely on native Node 2.0 Autogrow for slot management
                const onConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function () {
                    const res = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                    setTimeout(rebuildButtons, 10);
                    return res;
                };

                // Initial build
                setTimeout(() => {
                    rebuildButtons();
                    if (this.setSize) this.setSize(this.computeSize());
                }, 100);

                return r;
            };
        }
    }
});
