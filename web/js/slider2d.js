import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XENodes.Slider2D";
const NODE_NAME = "XENodes.Slider2D";

// Port area overhead: title area + 2 output slots + bottom margin
const PORT_OVERHEAD = 80;
const MIN_CANVAS_H = 50;

app.registerExtension({
    name: EXTENSION_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            if (originalOnNodeCreated) {
                originalOnNodeCreated.apply(this, arguments);
            }

            this.properties = this.properties || {};
            this.properties.valueX = this.properties.valueX ?? 512;
            this.properties.valueY = this.properties.valueY ?? 512;
            this.properties.minX = this.properties.minX ?? 0;
            this.properties.maxX = this.properties.maxX ?? 1024;
            this.properties.minY = this.properties.minY ?? 0;
            this.properties.maxY = this.properties.maxY ?? 1024;
            this.properties.stepX = this.properties.stepX ?? 1;
            this.properties.stepY = this.properties.stepY ?? 1;
            this.properties.snap = this.properties.snap ?? true;
            this.properties.dots = this.properties.dots ?? true;
            this.properties.frame = this.properties.frame ?? true;
            this.properties.frameAlert = this.properties.frameAlert ?? 0;
            this.properties.debug = this.properties.debug ?? false;
            
            // Explicitly remove legacy properties to hide them from the properties menu
            if ("decimalsX" in this.properties) delete this.properties.decimalsX;
            if ("decimalsY" in this.properties) delete this.properties.decimalsY;

            const debugLog = (...args) => {
                if (this.properties.debug) {
                    console.log("[Slider2D]", ...args);
                }
            };

            this.intpos = {
                x: (this.properties.valueX - this.properties.minX) / (this.properties.maxX - this.properties.minX),
                y: (this.properties.valueY - this.properties.minY) / (this.properties.maxY - this.properties.minY)
            };

            this.overhead = 0;

            // Initial size: 240px wide. Height will be adjusted to make canvas square.
            this.size = [240, 312];

            // Enable resizing
            this.resizable = true;

            const styleId = "xe-slider2d-style";
            if (!document.getElementById(styleId)) {
                const styleEl = document.createElement("style");
                styleEl.id = styleId;
                styleEl.innerHTML = `
                    .xe-slider2d-container {
                        position: relative;
                        background: rgba(15, 20, 15, 0.9);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 4px;
                        margin: 0px 4px 8px 4px; 
                        width: calc(100% - 8px); 
                        height: auto;
                        aspect-ratio: 1 / 1;
                        align-self: flex-start;
                        box-sizing: border-box;
                        overflow: hidden;
                        display: flex;
                        flex-direction: column;
                        backdrop-filter: blur(4px);
                    }
                    .xe-slider2d-canvas {
                        width: 100%;
                        height: 100%;
                        min-height: 0;
                        display: block; 
                        cursor: crosshair;
                        touch-action: none;
                    }
                `;
                document.head.appendChild(styleEl);
            }

            const container = document.createElement("div");
            container.className = "xe-slider2d-container";

            const canvas = document.createElement("canvas");
            canvas.className = "xe-slider2d-canvas";

            container.appendChild(canvas);

            const COLORS = {
                dots: "rgba(150, 210, 150, 0.25)",
                frame: "rgba(150, 210, 150, 0.1)",
                frameStroke: "rgba(150, 210, 150, 0.4)",
                handle: "#4A9A4A",
                handleOutline: "#fff"
            };

            let cachedRect = null;
            const updateRect = () => {
                cachedRect = canvas.getBoundingClientRect();
            };

            const syncWidgets = () => {
                if (this.widgets) {
                    for (const w of this.widgets) {
                        if (w.name === "X") w.value = this.properties.valueX;
                        if (w.name === "Y") w.value = this.properties.valueY;
                    }
                }
            };

            const getDecimals = (step) => {
                const s = String(step);
                if (s.indexOf(".") === -1) return 0;
                return s.split(".")[1].length;
            };

            const updatePortLabels = () => {
                if (!this.outputs) return;

                const decX = getDecimals(this.properties.stepX);
                const decY = getDecimals(this.properties.stepY);
                const valXText = this.properties.valueX.toFixed(decX);
                const valYText = this.properties.valueY.toFixed(decY);
                let changed = false;

                if (this.outputs[0] && this.outputs[0].label !== valXText) {
                    this.outputs[0].label = valXText;
                    this.outputs[0] = { ...this.outputs[0] };
                    changed = true;
                }
                if (this.outputs[1] && this.outputs[1].label !== valYText) {
                    this.outputs[1].label = valYText;
                    this.outputs[1] = { ...this.outputs[1] };
                    changed = true;
                }

                if (changed) {
                    this.outputs = [...this.outputs];
                    if (this.setDirtyCanvas) {
                        this.setDirtyCanvas(true, true);
                    }
                }

                syncWidgets();
            };

            const updateOutputTypes = () => {
                if (!this.outputs) return;

                const stepX = parseFloat(this.properties.stepX);
                const isIntX = Number.isInteger(stepX) && stepX >= 1;
                const typeX = isIntX ? "INT" : "FLOAT";
                if (this.outputs[0] && this.outputs[0].type !== typeX) {
                    this.outputs[0].type = typeX;
                }

                const stepY = parseFloat(this.properties.stepY);
                const isIntY = Number.isInteger(stepY) && stepY >= 1;
                const typeY = isIntY ? "INT" : "FLOAT";
                if (this.outputs[1] && this.outputs[1].type !== typeY) {
                    this.outputs[1].type = typeY;
                }
            };

            const draw = () => {
                const targetW = canvas.clientWidth;
                const targetH = canvas.clientHeight;

                if (targetW === 0 || targetH === 0) return;

                if (canvas.width !== targetW || canvas.height !== targetH) {
                    canvas.width = targetW;
                    canvas.height = targetH;
                    updateRect();
                }

                const ctx = canvas.getContext("2d");
                const w = canvas.width;
                const h = canvas.height;
                const shift = 5;
                const dw = w - shift * 2;
                const dh = h - shift * 2;

                ctx.clearRect(0, 0, w, h);

                if (this.properties.dots) {
                    ctx.fillStyle = COLORS.dots;
                    const stX = (dw * this.properties.stepX / (this.properties.maxX - this.properties.minX));
                    const stY = (dh * this.properties.stepY / (this.properties.maxY - this.properties.minY));
                    if (stX > 2 && stY > 2) {
                        ctx.beginPath();
                        for (let ix = 0; ix <= dw + 0.1; ix += stX) {
                            for (let iy = 0; iy <= dh + 0.1; iy += stY) {
                                ctx.rect(shift + ix - 0.5, shift + iy - 0.5, 1, 1);
                            }
                        }
                        ctx.fill();
                    }
                }

                const hX = shift + dw * this.intpos.x;
                const hY = shift + dh * (1 - this.intpos.y);

                if (this.properties.frame) {
                    ctx.fillStyle = COLORS.frame;
                    ctx.strokeStyle = COLORS.frameStroke;
                    ctx.beginPath();
                    ctx.rect(shift, hY, dw * this.intpos.x, dh * this.intpos.y);
                    ctx.fill();
                    ctx.stroke();
                }

                ctx.fillStyle = COLORS.handle;
                ctx.beginPath();
                ctx.arc(hX, hY, 7, 0, 2 * Math.PI);
                ctx.fill();

                ctx.strokeStyle = COLORS.handleOutline;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(hX, hY, 5, 0, 2 * Math.PI);
                ctx.stroke();
            };

            const updateValuesFromPos = (clientX, clientY, shiftKey = false) => {
                if (!cachedRect) updateRect();
                const rect = cachedRect;
                const shift = 5;
                const dw = rect.width - shift * 2;
                const dh = rect.height - shift * 2;

                let nx = (clientX - rect.left - shift) / dw;
                let ny = 1 - (clientY - rect.top - shift) / dh;

                nx = Math.max(0, Math.min(1, nx));
                ny = Math.max(0, Math.min(1, ny));

                if (shiftKey !== this.properties.snap) {
                    const sX = this.properties.stepX / (this.properties.maxX - this.properties.minX);
                    const sY = this.properties.stepY / (this.properties.maxY - this.properties.minY);
                    nx = Math.round(nx / sX) * sX;
                    ny = Math.round(ny / sY) * sY;
                }

                this.intpos.x = nx;
                this.intpos.y = ny;

                this.properties.valueX = this.properties.minX + (this.properties.maxX - this.properties.minX) * nx;
                this.properties.valueY = this.properties.minY + (this.properties.maxY - this.properties.minY) * ny;

                const decX = getDecimals(this.properties.stepX);
                const decY = getDecimals(this.properties.stepY);
                const pX = Math.pow(10, decX);
                const pY = Math.pow(10, decY);
                this.properties.valueX = Math.round(this.properties.valueX * pX) / pX;
                this.properties.valueY = Math.round(this.properties.valueY * pY) / pY;

                updatePortLabels();
                draw();
                if (this.setDirtyCanvas) this.setDirtyCanvas(true, false);
            };

            let isDragging = false;

            const handlePointerMove = (e) => {
                if (!isDragging) return;
                e.preventDefault();
                updateValuesFromPos(e.clientX, e.clientY, e.shiftKey);
            };

            const handlePointerUp = (e) => {
                if (!isDragging) return;
                isDragging = false;
                cachedRect = null;
                canvas.releasePointerCapture(e.pointerId);
                canvas.removeEventListener("pointermove", handlePointerMove);
                canvas.removeEventListener("pointerup", handlePointerUp);
            };

            canvas.addEventListener("pointerdown", (e) => {
                if (e.button !== 0) return;
                isDragging = true;
                e.preventDefault();
                e.stopPropagation();
                canvas.setPointerCapture(e.pointerId);
                updateRect();
                updateValuesFromPos(e.clientX, e.clientY, e.shiftKey);

                canvas.addEventListener("pointermove", handlePointerMove, { passive: false });
                canvas.addEventListener("pointerup", handlePointerUp);
            });

            canvas.addEventListener("wheel", (e) => e.stopPropagation());

            const domWidget = this.addDOMWidget("slider2d_ui", "SLIDER2D", container, {
                getMinHeight: () => MIN_CANVAS_H,
            });
            domWidget.draw = draw;

            // Use computeLayoutSize with minHeight === maxHeight to anchor widget height.
            // Height matches width to maintain square canvas.
            domWidget.computeLayoutSize = () => {
                const canvasW = this.size?.[0] ?? 240;
                const innerW = canvasW - 8;
                const h = Math.max(innerW, MIN_CANVAS_H);
                return { minHeight: h, maxHeight: h, minWidth: 0 };
            };

            const hideDataWidgets = () => {
                if (!this.widgets) return;
                for (const w of this.widgets) {
                    if (w.name !== "slider2d_ui") {
                        w.type = "hidden";
                        w.hidden = true;
                        w.options = w.options || {};
                        w.options.hidden = true;
                        w.computeSize = () => [0, -4];
                    }
                }
            };

            hideDataWidgets();
            updatePortLabels();

            const isDefaultSize = this.size[0] === 240 && this.size[1] === 312;

            // Adjust initial size to square once DOM layout is complete.
            // Use rAF x2 to wait for at least one layout pass.
            let _initFrames = 0;
            const _initAdjust = () => {
                if (_initFrames++ < 2) {
                    requestAnimationFrame(_initAdjust);
                    return;
                }
                hideDataWidgets();
                updatePortLabels();
                updateRect();
                updateOutputTypes();

                if (isDefaultSize) {
                    const cw = canvas.clientWidth;
                    const ch = canvas.clientHeight;
                    if (cw > 0 && ch > 0) {
                        const diff = cw - ch;
                        if (Math.abs(diff) > 1) {
                            this.size[1] += (diff + 12);
                        }
                    }
                }

                const cw = canvas.clientWidth;
                if (cw > 0) {
                    this.overhead = this.size[1] - (this.size[0] - 8) + 12;
                }

                updateRect();
                draw();
            };
            requestAnimationFrame(_initAdjust);

            const originalOnPropertyChanged = this.onPropertyChanged;
            this.onPropertyChanged = function (name, value) {
                if (originalOnPropertyChanged) originalOnPropertyChanged.apply(this, arguments);
                this.intpos.x = (this.properties.valueX - this.properties.minX) / (this.properties.maxX - this.properties.minX);
                this.intpos.y = (this.properties.valueY - this.properties.minY) / (this.properties.maxY - this.properties.minY);

                updatePortLabels();
                updateOutputTypes();
                draw();
                if (this.setDirtyCanvas) this.setDirtyCanvas(true, false);
            };

            this.onResize = function (size) {
                if (isDragging) isDragging = false;

                const lastW = this._lastSize?.[0];
                const lastH = this._lastSize?.[1];
                const dw = lastW != null ? Math.abs(size[0] - lastW) : 0;
                const dh = lastH != null ? Math.abs(size[1] - lastH) : 0;

                // Guard vertical resize: block if dragging primarily vertically
                if (dh > dw * 1.5 && dw < 5 && lastW != null) {
                    size[0] = lastW;
                    size[1] = lastH;
                } else {
                    // Update height to stay square based on width
                    if (this.overhead > 0) {
                        size[1] = size[0] - 8 + this.overhead;
                    }
                }

                this._lastSize = [size[0], size[1]];

                // Force DOM height to match canvas square to prevent rendering stretch
                if (container) {
                    const h = size[0] - 8;
                    container.style.height = h + "px";
                    container.style.minHeight = h + "px";
                    container.style.maxHeight = h + "px";
                }

                container.scrollTop = 0;
                if (container.parentElement) container.parentElement.scrollTop = 0;

                debugLog("onResize result:", size[0], size[1]);

                requestAnimationFrame(() => {
                    cachedRect = null;
                    updateRect();
                    draw();
                });
            };

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (originalOnRemoved) originalOnRemoved.apply(this, arguments);
                canvas.removeEventListener("pointermove", handlePointerMove);
                canvas.removeEventListener("pointerup", handlePointerUp);
            };

            const originalOnConfigure = this.onConfigure;
            this.onConfigure = function (info) {
                if (originalOnConfigure) originalOnConfigure.apply(this, arguments);

                // Ensure legacy properties are removed when configuring from saved state
                if (this.properties) {
                    if ("decimalsX" in this.properties) delete this.properties.decimalsX;
                    if ("decimalsY" in this.properties) delete this.properties.decimalsY;
                }

                updatePortLabels();
                updateOutputTypes();
                draw();
            };
        };
    }
});