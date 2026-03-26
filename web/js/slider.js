import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XENodes.Slider";
const NODE_NAME = "XENodes.Slider";

app.registerExtension({
    name: EXTENSION_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (originalOnNodeCreated) {
                originalOnNodeCreated.apply(this, arguments);
            }

            // Default properties
            this.properties = this.properties || {};
            this.properties.value = this.properties.value ?? 20;
            this.properties.min = this.properties.min ?? 0;
            this.properties.max = this.properties.max ?? 100;
            this.properties.step = this.properties.step ?? 1;

            const styleId = "xe-slider-style";
            if (!document.getElementById(styleId)) {
                const styleEl = document.createElement("style");
                styleEl.id = styleId;
                styleEl.innerHTML = `
                    .xe-slider-container {
                        display: flex;
                        align-items: center;
                        background: rgba(30, 30, 30, 0.6);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 6px;
                        padding: 0px 8px;
                        margin: -31px 4px 0px 4px; /* Pull up more, remove bottom margin */
                        box-sizing: border-box;
                        width: calc(100% - 8px);
                        height: 20px;
                        backdrop-filter: blur(4px);
                    }
                    .xe-slider-range {
                        -webkit-appearance: none;
                        width: 100%;
                        background: transparent;
                        margin: 0 4px 0 0;
                        cursor: pointer;
                    }
                    .xe-slider-range::-webkit-slider-runnable-track {
                        width: 100%;
                        height: 4px;
                        background: rgba(255, 255, 255, 0.15);
                        border-radius: 2px;
                    }
                    .xe-slider-range::-webkit-slider-thumb {
                        -webkit-appearance: none;
                        height: 12px;
                        width: 12px;
                        border-radius: 50%;
                        background: #4A90E2;
                        margin-top: -4px;
                        box-shadow: 0 0 4px rgba(74, 144, 226, 0.6);
                        transition: transform 0.1s;
                    }
                    .xe-slider-range::-webkit-slider-thumb:hover {
                        transform: scale(1.2);
                        background: #5CA0F2;
                    }
                    .xe-number-input {
                        width: 40px;
                        background: transparent;
                        color: #eee;
                        border: none;
                        border-left: 1px solid rgba(255, 255, 255, 0.1);
                        padding-left: 4px;
                        font-family: inherit;
                        font-size: 12px;
                        text-align: right;
                        outline: none;
                        -moz-appearance: textfield;
                    }
                    .xe-number-input::-webkit-outer-spin-button,
                    .xe-number-input::-webkit-inner-spin-button {
                        -webkit-appearance: none;
                        margin: 0;
                    }
                `;
                document.head.appendChild(styleEl);
            }

            const container = document.createElement("div");
            container.className = "xe-slider-container";

            const sliderInput = document.createElement("input");
            sliderInput.className = "xe-slider-range";
            sliderInput.type = "range";

            const numberInput = document.createElement("input");
            numberInput.className = "xe-number-input";
            numberInput.type = "number";

            container.appendChild(sliderInput);
            container.appendChild(numberInput);

            const normalizeValue = (value) => {
                let normalized = parseFloat(value);
                if (Number.isNaN(normalized)) {
                    normalized = parseFloat(this.properties.min);
                }
                if (Number.isNaN(normalized)) {
                    normalized = 0;
                }
                return normalized;
            };

            const hideDataWidgets = () => {
                if (!this.widgets) return;
                for (const w of this.widgets) {
                    if (w.name !== "slider_ui") {
                        w.type = "hidden";
                        w.hidden = true;
                        w.options = w.options || {};
                        w.options.hidden = true;
                        // Do not clear w.name here! It breaks backend serialization.
                        if (w.computeSize) {
                            w.computeSize = () => [0, -4];
                        }
                    }
                }
            };

            const getDataWidgets = () => (this.widgets || []).filter(
                (widget) => (widget.name === "value" || widget.name === "Xi") && widget.type !== "SLIDER"
            );

            const syncDataWidgets = (value, invokeCallback = false) => {
                for (const widget of getDataWidgets()) {
                    widget.value = value;
                    if (invokeCallback && typeof widget.callback === "function") {
                        widget.callback(value, app.canvas, this, [value]);
                    }
                }
            };

            const updateInputs = () => {
                sliderInput.min = this.properties.min;
                sliderInput.max = this.properties.max;
                sliderInput.step = this.properties.step;

                numberInput.min = this.properties.min;
                numberInput.max = this.properties.max;
                numberInput.step = this.properties.step;

                // Sync value
                const valToSet = normalizeValue(this.properties.value);
                this.properties.value = valToSet;

                sliderInput.value = valToSet;
                numberInput.value = valToSet;

                // Disable if connected to an input node
                const isConnected = this.inputs && this.inputs.some(i => i.name === "value" && i.link != null);
                sliderInput.disabled = isConnected;
                numberInput.disabled = isConnected;
                container.style.opacity = isConnected ? "0.5" : "1.0";
                sliderInput.style.cursor = isConnected ? "default" : "pointer";

                syncDataWidgets(valToSet);
                hideDataWidgets();
            };

            const onValueChange = (val) => {
                const numVal = normalizeValue(val);

                const finalVal = numVal;
                if (this.properties.value === finalVal) return; // Prevent infinite loops

                this.properties.value = finalVal;
                syncDataWidgets(finalVal, true);
                updateInputs();
                app.canvas?.setDirty(true, true);
                if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
            };

            sliderInput.addEventListener("input", (e) => onValueChange(e.target.value));
            numberInput.addEventListener("change", (e) => onValueChange(e.target.value));

            // Prevent panning graph
            const stopPropagation = (e) => e.stopPropagation();
            sliderInput.addEventListener("mousedown", stopPropagation);
            numberInput.addEventListener("mousedown", stopPropagation);
            sliderInput.addEventListener("wheel", stopPropagation);
            numberInput.addEventListener("wheel", stopPropagation);

            // Hide "value" text from the output port and hide default widget
            let _initFrames = 0;
            const _initAdjust = () => {
                if (_initFrames++ < 20) {
                    requestAnimationFrame(_initAdjust);
                }

                hideDataWidgets();

                if (this.outputs && this.outputs.length > 0) {
                    this.outputs[0].label = "";
                    this.outputs[0].name = " ";
                    this.outputs = [...this.outputs];
                }

                if (this.computeSize && this.setSize) {
                    const newSize = this.computeSize([this.size[0], this.size[1]]);
                    this.setSize([newSize[0], Math.max(newSize[1], 30)]);
                }
                app.canvas?.setDirty(true, true);
            };
            requestAnimationFrame(_initAdjust);

            const domWidget = this.addDOMWidget("slider_ui", "SLIDER", container, {
                getValue: () => normalizeValue(this.properties?.value),
                setValue(v) { onValueChange(v); }
            });

            domWidget.computeSize = (width) => [Math.max(width, 180), 20];

            // Override node's computeSize to enforce compactness
            const originalComputeSize = this.computeSize;
            this.computeSize = function (size) {
                const computed = originalComputeSize ? originalComputeSize.apply(this, arguments) : [200, 30];
                return [computed[0], 30]; // Force extremely compact height
            };

            this.onResize = function (size) {
                const minSize = this.computeSize();
                size[1] = minSize[1]; // Force height to be minimum
            };

            this.serialize_widgets = true;
            updateInputs();

            const originalOnPropertyChanged = this.onPropertyChanged;
            this.onPropertyChanged = function (name, value) {
                if (originalOnPropertyChanged) {
                    originalOnPropertyChanged.apply(this, arguments);
                }
                updateInputs();
                app.canvas?.setDirty(true, true);
            };

            const originalOnConfigure = this.onConfigure;
            this.onConfigure = function (info) {
                originalOnConfigure?.apply(this, arguments);

                const configuredValue = info?.properties?.value
                    ?? info?.widgets_values?.find((value) => typeof value === "number");

                if (configuredValue !== undefined) {
                    this.properties.value = normalizeValue(configuredValue);
                }

                updateInputs();
                requestAnimationFrame(() => updateInputs());
            };

            const originalOnSerialize = this.onSerialize;
            this.onSerialize = function (info) {
                originalOnSerialize?.apply(this, arguments);

                const value = normalizeValue(this.properties?.value);
                this.properties.value = value;

                info.properties = info.properties || {};
                info.properties.value = value;

                if (Array.isArray(info.widgets_values) && this.widgets?.length) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        const widget = this.widgets[i];
                        if (widget.name === "value" || widget.name === "Xi" || widget.name === "slider_ui") {
                            info.widgets_values[i] = value;
                        }
                    }
                }
            };

            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.apply(this, arguments);
                }
                updateInputs();
            };
        };
    }
});
