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
                        padding: 2px 8px;
                        margin: -18px 4px 4px 4px; /* Pull up to hide gap */
                        box-sizing: border-box;
                        width: calc(100% - 8px);
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

            const updateInputs = () => {
                sliderInput.min = this.properties.min;
                sliderInput.max = this.properties.max;
                sliderInput.step = this.properties.step;
                
                numberInput.min = this.properties.min;
                numberInput.max = this.properties.max;
                numberInput.step = this.properties.step;

                // Sync value
                let valToSet = parseFloat(this.properties.value);
                if (isNaN(valToSet)) valToSet = parseFloat(this.properties.min) || 0;

                sliderInput.value = valToSet;
                numberInput.value = valToSet;

                // Disable if connected to an input node
                const isConnected = this.inputs && this.inputs.some(i => i.name === "value" && i.link != null);
                sliderInput.disabled = isConnected;
                numberInput.disabled = isConnected;
                container.style.opacity = isConnected ? "0.5" : "1.0";
                sliderInput.style.cursor = isConnected ? "default" : "pointer";

                // Sync internal backend widget
                if (this.widgets) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        const w = this.widgets[i];
                        if ((w.name === "value" || w.name === "Xi") && w.type !== "SLIDER") {
                            w.value = valToSet;
                            // Ensure the framework catches the update
                            if (w.callback && typeof w.callback === "function") {
                                // sometimes calling callback causes loops if it isn't protected,
                                // but the ComfyUI API generally doesn't loop from setWidgetValue
                                w.callback(valToSet, app.canvas, this, [valToSet]);
                            }
                        }
                    }
                }
            };

            const onValueChange = (val) => {
                let numVal = parseFloat(val);
                if (isNaN(numVal)) numVal = parseFloat(this.properties.min) || 0;

                const finalVal = numVal;
                if (this.properties.value === finalVal) return; // Prevent infinite loops

                this.properties.value = finalVal;
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
            setTimeout(() => {
                if (this.widgets) {
                    for (const w of this.widgets) {
                        if ((w.name === "value" || w.name === "Xi") && w.type !== "SLIDER") {
                            w.type = "hidden";
                            w.hidden = true;
                            // Do not clear w.name here! It breaks backend serialization.
                            if (w.computeSize) {
                                w.computeSize = () => [0, -4]; 
                            }
                        }
                    }
                }
                if (this.outputs && this.outputs.length > 0) {
                    this.outputs[0].label = ""; 
                    this.outputs[0].name = " ";
                }
                if (this.computeSize && this.setSize) {
                    const newSize = this.computeSize([this.size[0], this.size[1]]);
                    this.setSize([newSize[0], Math.max(newSize[1], 30)]);
                }
                app.canvas?.setDirty(true, true);
            }, 10);

            const domWidget = this.addDOMWidget("slider_ui", "SLIDER", container, {
                getValue() { return this.properties?.value || 0; },
                setValue(v) { onValueChange(v); }
            });

            domWidget.computeSize = (width) => [Math.max(width, 180), 20];
            
            // Override node's computeSize to enforce compactness
            const originalComputeSize = this.computeSize;
            this.computeSize = function(size) {
                const computed = originalComputeSize ? originalComputeSize.apply(this, arguments) : [200, 30];
                return [computed[0], 44]; // Force very compact height
            };
            
            updateInputs();

            const originalOnPropertyChanged = this.onPropertyChanged;
            this.onPropertyChanged = function(name, value) {
                if (originalOnPropertyChanged) {
                    originalOnPropertyChanged.apply(this, arguments);
                }
                updateInputs();
                app.canvas?.setDirty(true, true);
            };

            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function() {
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.apply(this, arguments);
                }
                updateInputs();
            };
        };
    }
});
