import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XeNodes.MultiSwitch";
const NODE_NAME = "MultiSwitch";
const MANAGED_INPUT_PREFIX = "inputs.input_";
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
    button.style.color = isActive ? "white" : "#888";
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
    button.style.fontSize = "11px";
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

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated
                ? originalOnNodeCreated.apply(this, arguments)
                : undefined;

            const selectWidget = this.widgets?.find((widget) => widget.name === "select");
            if (!selectWidget) return result;

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
                measureContext.font = "bold 11px sans-serif";
            }

            let maxLabelWidth = MIN_LABEL_WIDTH;

            const updateButtons = () => {
                const currentValue = Math.floor(selectWidget.value || 0);
                Array.from(container.children).forEach((button) => {
                    const index = Number.parseInt(button.dataset.index ?? "-1", 10);
                    setButtonActiveState(button, index === currentValue);
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

                size[0] = Math.max(size[0], maxLabelWidth + 60);
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
