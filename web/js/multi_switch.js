import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XENodes.MultiSwitch";
const NODE_NAME = "XENodes.MultiSwitch";
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
    button.style.fontWeight = "normal";
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
    button.style.fontWeight = "normal";
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

        const allowUnselect = node.properties?.allow_unselect === true;

        // Clicking the already-selected button deselects (sets select to -1) if allow_unselect property is true
        if (selectWidget.value === index) {
            if (!allowUnselect) return; // Ignore click if already selected and unselect is disallowed
            selectWidget.value = -1;
        } else {
            selectWidget.value = index;
        }

        selectWidget.callback?.(selectWidget.value);
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
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
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

            const unselectedMode = this.properties?.unselected_mode || "None";
            options.push({
                content: `Unselected Mode: ${unselectedMode}`,
                callback: () => {
                    const modes = ["None", "Mute", "Bypass"];
                    const nextMode = modes[(modes.indexOf(unselectedMode) + 1) % modes.length];
                    this.properties = this.properties || {};
                    this.properties.unselected_mode = nextMode;
                    if (this.updateUnselectedNodesModes) {
                        this.updateUnselectedNodesModes(nextMode === "None");
                    }
                    app.canvas?.setDirty(true, true);
                }
            });

            const allowUnselect = this.properties?.allow_unselect === true;
            options.push({
                content: `Allow Unselect: ${allowUnselect ? "On" : "Off"}`,
                callback: () => {
                    this.properties = this.properties || {};
                    this.properties.allow_unselect = !allowUnselect;
                    if (this.rebuildButtons) {
                        this.rebuildButtons();
                    }
                    app.canvas?.setDirty(true, true);
                }
            });
        };

        const originalDrawSlots = nodeType.prototype.drawSlots;
        nodeType.prototype.drawSlots = function (ctx, options) {
            if (this.properties?.hide_connections === true) {
                return; // Skip drawing connection dots when hidden
            }
            if (originalDrawSlots) {
                return originalDrawSlots.apply(this, arguments);
            }
        };

        const originalOnPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function (name, value) {
            if (originalOnPropertyChanged) {
                originalOnPropertyChanged.apply(this, arguments);
            }
            if (name === "hide_connections") {
                this.applyHideConnections?.();
            } else if (name === "unselected_mode") {
                this.updateUnselectedNodesModes?.(value === "None");
            } else if (name === "allow_unselect") {
                if (this.rebuildButtons) {
                    this.rebuildButtons();
                }
            }
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated
                ? originalOnNodeCreated.apply(this, arguments)
                : undefined;

            const selectWidget = this.widgets?.find((widget) => widget.name === "select");
            if (!selectWidget) return result;

            // Define properties for the property panel
            this.addProperty("hide_connections", false, "boolean");
            this.addProperty("unselected_mode", "None", "enum", { values: ["None", "Mute", "Bypass"] });
            this.addProperty("allow_unselect", false, "boolean");

            this.applyHideConnections = () => {
                const isHidden = this.properties.hide_connections === true;

                // Manage dynamic CSS for hiding slot connections
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

                // Handle widget visibility (Using type mutation + Vue Reactivity Hack)
                if (this.widgets) {
                    const index = this.widgets.findIndex(w => w.name === "select");
                    if (index !== -1) {
                        const w = this.widgets[index];
                        
                        // Backup original type
                        if (w.type !== "hidden") {
                            w._original_type = w.type;
                        }

                        // Update properties
                        w.hidden = isHidden;
                        w.type = isHidden ? "hidden" : (w._original_type || "number");
                        w.options = w.options || {};
                        w.options.hidden = isHidden;

                        // IMPORTANT: Force Vue.js to re-render the widget UI immediately
                        // Force Vue reactivity by removing and immediately re-inserting the widget at the same index
                        this.widgets.splice(index, 1);
                        this.widgets.splice(index, 0, w);
                    }
                }

                // Notify listeners of structural changes
                if (this.change) {
                    this.change();
                }

                // Recalculate height while preserving the user-resized width.
                if (this.computeSize && this.setSize) {
                    const minSize = this.computeSize();
                    const newWidth = Math.max(this.size[0], minSize[0]);
                    this.setSize([newWidth, minSize[1]]);
                }

                // Force canvas redraw
                const graph = this.graph || app.graph;
                if (graph) {
                    graph.setDirtyCanvas(true, true);
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
                measureContext.font = "12px sans-serif";
            }

            let maxLabelWidth = MIN_LABEL_WIDTH;

            this.updateUnselectedNodesModes = (restoreAll = false) => {
                const modeStr = this.properties?.unselected_mode || "None";
                const isNone = modeStr === "None";
                if (isNone && !restoreAll) return;

                const targetMode = modeStr === "Mute" ? 2 : 4; // 2: Mute, 4: Bypass
                const currentValue = Math.floor(selectWidget.value || 0);
                const inputSlots = getManagedInputs(this);

                let changed = false;
                const graph = this.graph || app.graph;
                for (let i = 0; i < inputSlots.length; i++) {
                    const inputSlot = inputSlots[i];
                    if (inputSlot.link == null) continue;

                    const link = graph?.links?.[inputSlot.link];
                    const originNode = link ? graph.getNodeById(link.origin_id) : null;
                    if (originNode) {
                        const newMode = (i === currentValue || isNone) ? 0 : targetMode;
                        if (originNode.mode !== newMode) {
                            originNode.mode = newMode;
                            changed = true;
                        }
                    }
                }
                if (changed && graph) {
                    graph.setDirtyCanvas(true, true);
                }
            };

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

                    // Disable buttons when connected, but do not change opacity to prevent base artifacts from showing
                    button.disabled = isSelectConnected;
                    button.style.cursor = isSelectConnected ? "default" : "pointer";
                });

                if (this.updateUnselectedNodesModes) {
                    this.updateUnselectedNodesModes();
                }
            };

            const rebuildButtons = this.rebuildButtons = () => {
                clearContainer(container);

                const inputSlots = getManagedInputs(this);

                let maxValidIndex = 0;
                maxLabelWidth = MIN_LABEL_WIDTH;

                const graph = this.graph || app.graph;
                for (let i = 0; i < inputSlots.length; i++) {
                    const inputSlot = inputSlots[i];
                    inputSlot.label = formatInputLabel(inputSlot.name);

                    if (inputSlot.link == null) continue;

                    maxValidIndex = Math.max(maxValidIndex, i);

                    const link = graph?.links?.[inputSlot.link];
                    const originNode = link ? graph.getNodeById(link.origin_id) : null;
                    const label = `[${i}] ${originNode?.title || originNode?.type || "(Empty)"}`;

                    const measuredWidth = measureContext
                        ? measureContext.measureText(label).width
                        : label.length * 8;
                    maxLabelWidth = Math.max(maxLabelWidth, measuredWidth);

                    container.appendChild(createButton(this, selectWidget, i, label));
                }

                const allowUnselect = this.properties?.allow_unselect === true;
                selectWidget.options = selectWidget.options || {};
                selectWidget.options.min = allowUnselect ? -1 : 0;
                selectWidget.options.max = maxValidIndex;
                updateButtons();
                container.style.display = container.children.length === 0 ? "none" : "flex";

                if (this.applyHideConnections) {
                    this.applyHideConnections();
                }

                if (this.size && this.computeSize) {
                    const targetSize = this.computeSize();
                    // Preserve the user-resized width: only expand to minWidth if the current width is narrower.
                    const newWidth = Math.max(this.size[0], targetSize[0]);
                    const newHeight = targetSize[1];
                    if (newWidth !== this.size[0] || newHeight !== this.size[1]) {
                        if (this.setSize) {
                            this.setSize([newWidth, newHeight]);
                        } else {
                            this.size[0] = newWidth;
                            this.size[1] = newHeight;
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

            domWidget.computeLayoutSize = () => {
                const h = container.children.length * BUTTON_HEIGHT;
                return { minHeight: h, minWidth: 0 };
            };

            const originalComputeSize = this.computeSize;
            this.computeSize = function () {
                const size = originalComputeSize
                    ? originalComputeSize.apply(this, arguments)
                    : [200, 100];

                if (this.flags.collapsed) return size;

                const buttonCount = container.children.length;

                if (this.properties?.hide_connections) {
                    size[1] = buttonCount * BUTTON_HEIGHT + 30;
                    // Returns minimum width; actual width is preserved by callers (rebuildButtons / applyHideConnections).
                    size[0] = Math.max(120, maxLabelWidth + 40);
                    this.widgets_start_y = 0;
                } else {
                    size[1] += 10;
                    // Returns minimum width; callers preserve the user-resized width if it is already wider.
                    size[0] = Math.max(size[0], maxLabelWidth + 60);
                    this.widgets_start_y = undefined;
                }

                return size;
            };

            const originalOnResize = this.onResize;
            this.onResize = function (size) {
                originalOnResize?.apply(this, arguments);
                const minSize = this.computeSize ? this.computeSize() : [200, 100];
                // Allow the user to freely resize the width, but never below the minimum.
                size[0] = Math.max(size[0], minSize[0]);
                // Height is always driven by content (button count), not user drag.
                size[1] = minSize[1];
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

            const checkTitles = () => {
                if (this.flags.collapsed) return;
                const inputSlots = getManagedInputs(this);
                let currentFingerprint = "";
                const graph = this.graph || app.graph;
                for (let i = 0; i < inputSlots.length; i++) {
                    const inputSlot = inputSlots[i];
                    if (inputSlot.link == null) continue;
                    const link = graph?.links?.[inputSlot.link];
                    const originNode = link ? graph.getNodeById(link.origin_id) : null;
                    currentFingerprint += `${i}:${originNode?.title || originNode?.type || ""}|`;
                }

                if (currentFingerprint !== this.lastTitlesFingerprint) {
                    this.lastTitlesFingerprint = currentFingerprint;
                    this.rebuildButtons();
                }
            };

            this.lastTitlesFingerprint = "";
            this.title_check_interval = setInterval(checkTitles, 1000);

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (this.title_check_interval) {
                    clearInterval(this.title_check_interval);
                    this.title_check_interval = null;
                }
                return originalOnRemoved?.apply(this, arguments);
            };

            const originalOnDrawForeground = this.onDrawForeground;
            this.onDrawForeground = function (ctx) {
                return originalOnDrawForeground
                    ? originalOnDrawForeground.apply(this, arguments)
                    : undefined;
            };

            rebuildButtons();
            return result;
        };
    }
});
