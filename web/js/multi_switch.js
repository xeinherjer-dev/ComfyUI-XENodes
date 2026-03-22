import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XeNodes.MultiSwitch";
const NODE_NAME = "MultiSwitch";
const MANAGED_INPUT_PREFIX = "inputs.input";
const BUTTON_HEIGHT = 32;
const MIN_LABEL_WIDTH = 100;

const formatInputLabel = (name) => name?.startsWith(MANAGED_INPUT_PREFIX)
    ? name.slice("inputs.".length)
    : name;

const getManagedInputs = (node) => (node.inputs || []).filter(
    (input) => input.name?.startsWith(MANAGED_INPUT_PREFIX)
);

const clearContainer = (container) => {
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }
};

const setButtonActiveState = (button, isActive) => {
    button.style.backgroundColor = isActive ? "#444444" : "#252525";
    button.style.color = isActive ? "white" : "#bbbbbb";
    button.style.border = isActive ? "1px solid #666666" : "1px solid #333";
    button.style.borderLeft = isActive ? "4px solid #4CAF50" : "1px solid #333";
    button.style.fontWeight = isActive ? "900" : "bold";
};

const createButton = (node, selectWidget, index, label) => {
    const button = document.createElement("button");
    button.dataset.index = String(index);
    button.innerText = label;
    button.title = `Switch to input ${index}`;

    button.style.cursor = "pointer";
    button.style.border = "1px solid";
    button.style.borderRadius = "4px";
    button.style.padding = "8px 10px";
    button.style.fontSize = "12px";
    button.style.fontWeight = "bold";
    button.style.textAlign = "left";
    button.style.outline = "none";
    button.style.width = "100%";
    button.style.whiteSpace = "nowrap";
    button.style.overflow = "hidden";
    button.style.textOverflow = "ellipsis";
    button.style.minHeight = `${BUTTON_HEIGHT}px`;

    button.onclick = (event) => {
        event.preventDefault();
        event.stopPropagation();

        if (selectWidget.value === index) return;

        selectWidget.value = index;
        selectWidget.callback?.(index);
        node.change?.();
        node.setDirtyCanvas(true, true);
    };

    return button;
};

app.registerExtension({
    name: EXTENSION_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
            if (originalGetExtraMenuOptions) {
                originalGetExtraMenuOptions.apply(this, arguments);
            }
            const isHidden = this.properties?.hide_connections === true;
            options.push({
                content: isHidden ? "Show Connections" : "Hide Connections",
                callback: () => {
                    this.properties = this.properties || {};
                    this.properties.hide_connections = !isHidden;
                    if (this.applyHideConnections) {
                        this.applyHideConnections();
                    }
                    app.canvas?.setDirty(true, true);
                }
            });
        };

        const originalDrawSlots = nodeType.prototype.drawSlots;
        nodeType.prototype.drawSlots = function(ctx, options) {
            if (this.properties?.hide_connections === true) {
                return; // Skip drawing connection dots when hidden
            }
            if (originalDrawSlots) {
                return originalDrawSlots.apply(this, arguments);
            }
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated
                ? originalOnNodeCreated.apply(this, arguments)
                : undefined;

            const selectWidget = this.widgets?.find((widget) => widget.name === "select");
            if (!selectWidget) return result;

            this.properties = this.properties || {};
            if (this.properties.hide_connections === undefined) {
                this.properties.hide_connections = false;
            }

            this.applyHideConnections = () => {
                const isHidden = this.properties.hide_connections === true;
                
                const styleId = `multi-switch-style-${this.id}`;
                let styleEl = document.getElementById(styleId);
                if (!styleEl) {
                    styleEl = document.createElement("style");
                    styleEl.id = styleId;
                    document.head.appendChild(styleEl);
                }

                if (isHidden) {
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
                    `;
                    styleEl.innerHTML = css;
                } else {
                    styleEl.innerHTML = "";
                }

                if (this.widgets) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        if (this.widgets[i].name === "select") {
                            this.widgets[i].hidden = isHidden;
                        }
                    }
                }

                if (this.change) {
                    this.change(); // Notify listeners of structural changes
                }

                if (this.computeSize && this.setSize) {
                    const newSize = this.computeSize([this.size[0], this.size[1]]);
                    this.setSize([newSize[0], newSize[1]]);
                }
            };


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

            const measureContext = document.createElement("canvas").getContext("2d");
            if (measureContext) {
                measureContext.font = "bold 12px sans-serif";
            }

            let maxLabelWidth = MIN_LABEL_WIDTH;

            const updateButtons = () => {
                // Check if the select pin is connected
                const isSelectConnected = this.inputs?.some(i => i.name === "select" && i.link != null);
                const currentValue = Math.floor(selectWidget.value || 0);
                
                // Disable the select widget if connected
                selectWidget.disabled = isSelectConnected;

                Array.from(container.children).forEach((button) => {
                    const index = Number.parseInt(button.dataset.index ?? "-1", 10);
                    // Clear selection highlight when select input is connected
                    const isActive = isSelectConnected ? false : index === currentValue;
                    setButtonActiveState(button, isActive);
                    
                    // Disable and gray out (reduce opacity) buttons when connected
                    button.disabled = isSelectConnected;
                    button.style.opacity = isSelectConnected ? "0.5" : "1.0";
                    button.style.cursor = isSelectConnected ? "default" : "pointer";
                });
            };

            const rebuildButtons = this.rebuildButtons = () => {
                clearContainer(container);

                const inputSlots = getManagedInputs(this);
                
                let maxValidIndex = 0;
                maxLabelWidth = MIN_LABEL_WIDTH;

                for (let i = 0; i < inputSlots.length; i++) {
                    const inputSlot = inputSlots[i];
                    inputSlot.label = formatInputLabel(inputSlot.name);

                    if (inputSlot.link == null) continue;

                    maxValidIndex = Math.max(maxValidIndex, i);

                    const link = app.graph?.links?.[inputSlot.link];
                    const originNode = link ? app.graph.getNodeById(link.origin_id) : null;
                    const label = `[${i}] ${originNode?.title || originNode?.type || "(Empty)"}`;

                    const measuredWidth = measureContext
                        ? measureContext.measureText(label).width
                        : label.length * 8;
                    maxLabelWidth = Math.max(maxLabelWidth, measuredWidth);

                    container.appendChild(createButton(this, selectWidget, i, label));
                }

                selectWidget.options = selectWidget.options || {};
                selectWidget.options.max = maxValidIndex;
                updateButtons();
                container.style.display = container.children.length === 0 ? "none" : "flex";

                if (this.applyHideConnections) {
                    this.applyHideConnections();
                }

                if (this.size && this.computeSize) {
                    const targetSize = this.computeSize();
                    if (targetSize[0] > this.size[0] || targetSize[1] !== this.size[1]) {
                        const newWidth = Math.max(this.size[0], targetSize[0]);
                        if (this.setSize) {
                            this.setSize([newWidth, targetSize[1]]);
                        } else {
                            this.size[0] = newWidth;
                            this.size[1] = targetSize[1];
                        }
                    }
                }

                app.canvas?.setDirty(true, true);
            };

            const domWidget = this.addDOMWidget("select_buttons", "BUTTONS", container, {
                getValue() {
                    return selectWidget.value;
                },
                setValue(value) {
                    selectWidget.value = value;
                }
            });

            domWidget.computeSize = (width) => [
                width,
                (container.children.length * BUTTON_HEIGHT) + 10
            ];

            const originalComputeSize = this.computeSize;
            this.computeSize = function () {
                const size = originalComputeSize
                    ? originalComputeSize.apply(this, arguments)
                    : [200, 100];

                if (this.flags.collapsed) return size;

                const buttonCount = container.children.length;
                const HEADER_HEIGHT = 46;

                if (this.properties?.hide_connections) {
                    size[1] = buttonCount * BUTTON_HEIGHT + HEADER_HEIGHT + 10;
                    size[0] = Math.max(120, maxLabelWidth + 40);
                    this.widgets_up = true;
                    this.widgets_start_y = 0;
                } else {
                    size[0] = Math.max(size[0], maxLabelWidth + 60);
                    this.widgets_up = undefined; 
                    this.widgets_start_y = undefined;
                }

                return size;
            };

            const originalOnResize = this.onResize;
            this.onResize = function (size) {
                originalOnResize?.apply(this, arguments);
                const minSize = this.computeSize();
                size[0] = Math.max(size[0], minSize[0]);
                size[1] = Math.max(size[1], minSize[1]);
            };

            const originalSelectCallback = selectWidget.callback;
            selectWidget.callback = function () {
                originalSelectCallback?.apply(this, arguments);
                updateButtons();
                app.canvas?.setDirty(true, true);
            };

            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                const response = originalOnConnectionsChange
                    ? originalOnConnectionsChange.apply(this, arguments)
                    : undefined;
                rebuildButtons();
                return response;
            };

            rebuildButtons();
            return result;
        };
    }
});
