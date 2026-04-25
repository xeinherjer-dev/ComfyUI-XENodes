import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XENodes.Slider2D";
const NODE_NAME = "XENodes.Slider2D";

const DEFAULT_SIZE = [240, 312];
const NODES2_MIN_CANVAS_W = 50;
const NODES2_MIN_CANVAS_H = 50;
const LEGACY_MIN_CANVAS_H = 80;
const LEGACY_SIDE_MARGIN = 4;
const LEGACY_BOTTOM_GAP = 0;
// Legacy mode: domWidget.y is always (outputCount + 0.2) * NODE_SLOT_HEIGHT + 2.
// With 2 outputs and NODE_SLOT_HEIGHT=20 → (2.2 × 20) + 2 = 46. Static value; no runtime read.
const LEGACY_TOP_OFFSET = (2 + 0.2) * (globalThis.LiteGraph?.NODE_SLOT_HEIGHT ?? 20) + 2;

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const toFiniteNumber = (value, fallback) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
};

const roundToDecimals = (value, decimals) => {
    const precision = Math.pow(10, decimals);
    return Math.round(value * precision) / precision;
};

const getDecimals = (step) => {
    const normalized = String(step);
    if (!normalized.includes(".")) return 0;
    return normalized.split(".")[1].length;
};

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
            this.properties.debug = this.properties.debug ?? false;

            const cleanupLegacyProperties = () => {
                if ("decimalsX" in this.properties) delete this.properties.decimalsX;
                if ("decimalsY" in this.properties) delete this.properties.decimalsY;
                if ("frameAlert" in this.properties) delete this.properties.frameAlert;
            };

            // Remove legacy / unused properties so they do not leak into the menu.
            cleanupLegacyProperties();

            const debugLog = (label, extra = {}) => {
                if (!this.properties.debug) return;
                const mode = isNodes2Environment() ? "nodes2" : "legacy";
                const y = domWidget?.y ?? 'undefined';
                let extraStr = "";
                for (let k in extra) extraStr += `, ${k}=${extra[k]}`;
                console.log(`[Slider2D-Diagnostics] ${label} - mode=${mode}, widgetY=${y}${extraStr}`);
            };

            const isNodes2Environment = () => {
                if (globalThis.LiteGraph?.vueNodesMode != null) {
                    return !!globalThis.LiteGraph.vueNodesMode;
                }
                return !!container.closest("comfy-node");
            };

            const getMinCanvasHeight = () =>
                isNodes2Environment() ? NODES2_MIN_CANVAS_H : LEGACY_MIN_CANVAS_H;

            const getLegacyBottomGap = () => LEGACY_BOTTOM_GAP;

            const getAxisState = (axisName) => {
                const rawMin = toFiniteNumber(this.properties[`min${axisName}`], 0);
                const rawMax = toFiniteNumber(this.properties[`max${axisName}`], rawMin);
                const min = Math.min(rawMin, rawMax);
                const max = Math.max(rawMin, rawMax);
                const span = max - min;
                const rawStep = Math.abs(toFiniteNumber(this.properties[`step${axisName}`], 1));
                const step = rawStep > 0 ? rawStep : 1;
                return {
                    min,
                    max,
                    span,
                    step,
                    decimals: getDecimals(step),
                };
            };

            const getNormalizedValue = (value, axis) => {
                if (axis.span <= 0) return 0;
                return clamp((value - axis.min) / axis.span, 0, 1);
            };

            const snapNormalizedValue = (value, axis) => {
                if (axis.span <= 0) return 0;
                const normalizedStep = axis.step / axis.span;
                if (!Number.isFinite(normalizedStep) || normalizedStep <= 0) {
                    return clamp(value, 0, 1);
                }
                return clamp(Math.round(value / normalizedStep) * normalizedStep, 0, 1);
            };

            const denormalizeValue = (value, axis) =>
                axis.span <= 0 ? axis.min : axis.min + axis.span * value;

            const syncIntposFromProperties = () => {
                const axisX = getAxisState("X");
                const axisY = getAxisState("Y");

                this.properties.valueX = clamp(
                    toFiniteNumber(this.properties.valueX, axisX.min),
                    axisX.min,
                    axisX.max
                );
                this.properties.valueY = clamp(
                    toFiniteNumber(this.properties.valueY, axisY.min),
                    axisY.min,
                    axisY.max
                );

                this.intpos = {
                    x: getNormalizedValue(this.properties.valueX, axisX),
                    y: getNormalizedValue(this.properties.valueY, axisY),
                };
            };

            const updateRect = () => {
                cachedRect = canvas.getBoundingClientRect();
            };

            const requestDraw = () => {
                cachedRect = null;
                updateRect();
                draw();
            };

            this.intpos = { x: 0, y: 0 };
            syncIntposFromProperties();

            this.size = [...DEFAULT_SIZE];
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
                        margin: 0px 4px 4px 4px;
                        width: calc(100% - 8px);
                        height: 100%;
                        min-height: 0;
                        align-self: flex-start;
                        box-sizing: border-box;
                        overflow: hidden;
                        display: flex;
                        flex-direction: column;
                        backdrop-filter: blur(4px);
                    }
                    .xe-slider2d-stage {
                        position: relative;
                        flex: 1 1 auto;
                        min-height: 0;
                        width: 100%;
                        overflow: hidden;
                    }
                    .xe-slider2d-canvas {
                        position: absolute;
                        inset: 0;
                        width: 100%;
                        height: 100%;
                        display: block;
                        cursor: crosshair;
                        touch-action: none;
                    }
                `;
                document.head.appendChild(styleEl);
            }

            const container = document.createElement("div");
            container.className = "xe-slider2d-container";

            const stage = document.createElement("div");
            stage.className = "xe-slider2d-stage";

            const canvas = document.createElement("canvas");
            canvas.className = "xe-slider2d-canvas";

            stage.appendChild(canvas);
            container.appendChild(stage);

            const COLORS = {
                dots: "rgba(150, 210, 150, 0.25)",
                frame: "rgba(150, 210, 150, 0.1)",
                frameStroke: "rgba(150, 210, 150, 0.4)",
                handle: "#4A9A4A",
                handleOutline: "#fff",
            };

            let cachedRect = null;
            let domWidget;
            let isDragging = false;

            const syncWidgets = (invokeCallback = false) => {
                if (!this.widgets) return;
                for (const widget of this.widgets) {
                    let valToSet = null;
                    if (widget.name === "X") valToSet = this.properties.valueX;
                    if (widget.name === "Y") valToSet = this.properties.valueY;

                    if (valToSet !== null) {
                        widget.value = valToSet;
                        // Callback to propagate final values to the Vue state in Nodes 2.0
                        if (invokeCallback && typeof widget.callback === "function") {
                            widget.callback(valToSet, app.canvas, this, [valToSet]);
                        }
                    }
                }
            };

            const refreshCoreState = (isDragging = false) => {
                if (this.outputs) {
                    let changed = false;
                    if (this.outputs[0] && this.outputs[0].label !== " ") {
                        this.outputs[0].label = " ";
                        this.outputs[0] = { ...this.outputs[0] };
                        changed = true;
                    }
                    if (this.outputs[1] && this.outputs[1].label !== " ") {
                        this.outputs[1].label = " ";
                        this.outputs[1] = { ...this.outputs[1] };
                        changed = true;
                    }
                    if (changed) {
                        this.outputs = [...this.outputs];
                        if (!isDragging) {
                            this.setDirtyCanvas?.(true, true);
                        }
                    }
                }
                syncWidgets(!isDragging);
            };

            const updateOutputTypes = () => {
                if (!this.outputs) return;

                const updateTypeForSlot = (slotIndex) => {
                    const links = this.outputs[slotIndex]?.links;
                    let newType = "*";
                    if (links && links.length > 0) {
                        const link = app.graph.links[links[0]];
                        if (link) {
                            const targetNode = app.graph.getNodeById(link.target_id);
                            const targetInput = targetNode?.inputs?.[link.target_slot];
                            const targetType = String(targetInput?.type).toUpperCase();
                            if (targetType === "INT" || targetType === "FLOAT" || targetType === "NUMBER") {
                                newType = targetType;
                            }
                        }
                    }
                    if (this.outputs[slotIndex] && this.outputs[slotIndex].type !== newType) {
                        this.outputs[slotIndex].type = newType;
                    }
                };

                updateTypeForSlot(0); // X
                updateTypeForSlot(1); // Y
            };

            const applyModeStyles = () => {
                if (isNodes2Environment()) {
                    stage.style.position = "relative";
                    stage.style.display = "block";
                    stage.style.flex = "1 1 auto";
                    stage.style.height = "";
                    stage.style.minHeight = "0";

                    canvas.style.position = "absolute";
                    canvas.style.inset = "0";
                    canvas.style.width = "100%";
                    canvas.style.height = "100%";
                    canvas.style.flex = "";
                    canvas.style.minHeight = "0";
                    return;
                }

                stage.style.position = "relative";
                stage.style.display = "flex";
                stage.style.flex = "1 1 auto";
                stage.style.height = "100%";
                stage.style.minHeight = getMinCanvasHeight() + "px";

                canvas.style.position = "relative";
                canvas.style.inset = "";
                canvas.style.width = "100%";
                canvas.style.height = "100%";
                canvas.style.flex = "1 1 auto";
                canvas.style.minHeight = getMinCanvasHeight() + "px";
            };

            const applyContainerSize = (nodeHeight = this.size[1]) => {
                applyModeStyles();

                if (isNodes2Environment()) {
                    container.style.width = "100%";
                    container.style.margin = "0px";
                    container.style.height = "100%";
                    container.style.minHeight = "0";
                    container.style.maxHeight = "none";
                    container.style.alignSelf = "stretch";
                    debugLog("applyContainerSize:nodes2", { requestedNodeHeight: nodeHeight });
                    return;
                }

                // Slots are hidden via drawSlots override + CSS + widgets_start_y=0,
                // so the DOM widget now starts directly below the title bar.
                // No negative top offset needed.
                const overlapUp = 0;
                const bottomGap = getLegacyBottomGap();
                
                const canvasHeight = Math.max(
                    nodeHeight - 30 - bottomGap, 
                    getMinCanvasHeight()
                );

                container.style.width = `calc(100% - ${LEGACY_SIDE_MARGIN * 2}px)`;
                container.style.margin = `0px ${LEGACY_SIDE_MARGIN}px ${bottomGap}px ${LEGACY_SIDE_MARGIN}px`;
                container.style.height = canvasHeight + "px";
                container.style.minHeight = canvasHeight + "px";
                container.style.maxHeight = canvasHeight + "px";
                container.style.alignSelf = "flex-start";

                debugLog("applyContainerSize:legacy_final", {
                    reqH: nodeHeight,
                    appH: canvasHeight,
                    overlap: overlapUp,
                    margin: container.style.margin
                });
            };

            const draw = () => {
                const targetW = canvas.clientWidth;
                const targetH = canvas.clientHeight;
                if (targetW <= 0 || targetH <= 0) return;

                if (canvas.width !== targetW || canvas.height !== targetH) {
                    canvas.width = targetW;
                    canvas.height = targetH;
                    updateRect();
                }

                const ctx = canvas.getContext("2d");
                if (!ctx) return;

                const axisX = getAxisState("X");
                const axisY = getAxisState("Y");
                const w = canvas.width;
                const h = canvas.height;
                const shift = 5;
                const dw = Math.max(0, w - shift * 2);
                const dh = Math.max(0, h - shift * 2);
                const handleX = shift + dw * clamp(this.intpos.x, 0, 1);
                const handleY = shift + dh * (1 - clamp(this.intpos.y, 0, 1));

                ctx.clearRect(0, 0, w, h);

                if (this.properties.dots && axisX.span > 0 && axisY.span > 0) {
                    const stepX = (dw * axisX.step) / axisX.span;
                    const stepY = (dh * axisY.step) / axisY.span;
                    if (stepX > 2 && stepY > 2) {
                        ctx.fillStyle = COLORS.dots;
                        ctx.beginPath();
                        for (let x = 0; x <= dw + 0.1; x += stepX) {
                            for (let y = 0; y <= dh + 0.1; y += stepY) {
                                ctx.rect(shift + x - 0.5, shift + y - 0.5, 1, 1);
                            }
                        }
                        ctx.fill();
                    }
                }

                if (this.properties.frame) {
                    ctx.fillStyle = COLORS.frame;
                    ctx.strokeStyle = COLORS.frameStroke;
                    ctx.beginPath();
                    ctx.rect(
                        shift,
                        handleY,
                        dw * clamp(this.intpos.x, 0, 1),
                        dh * clamp(this.intpos.y, 0, 1)
                    );
                    ctx.fill();
                    ctx.stroke();
                }

                ctx.fillStyle = COLORS.handle;
                ctx.beginPath();
                ctx.arc(handleX, handleY, 7, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = COLORS.handleOutline;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(handleX, handleY, 5, 0, Math.PI * 2);
                ctx.stroke();

                ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "right";
                ctx.textBaseline = "middle";
                const textX = this.properties.valueX.toFixed(axisX.decimals);
                const textY = this.properties.valueY.toFixed(axisY.decimals);
                
                ctx.fillText(textX, w - 8, 8);
                ctx.fillText(textY, w - 8, 24);
            };

            const updateValuesFromPos = (clientX, clientY, shiftKey = false, isDragging = false) => {
                if (!cachedRect) updateRect();
                if (!cachedRect) return;

                const axisX = getAxisState("X");
                const axisY = getAxisState("Y");
                const shift = 5;
                const dw = cachedRect.width - shift * 2;
                const dh = cachedRect.height - shift * 2;
                if (dw <= 0 || dh <= 0) return;

                let nx = (clientX - cachedRect.left - shift) / dw;
                let ny = 1 - (clientY - cachedRect.top - shift) / dh;

                nx = clamp(nx, 0, 1);
                ny = clamp(ny, 0, 1);

                if (shiftKey !== this.properties.snap) {
                    nx = snapNormalizedValue(nx, axisX);
                    ny = snapNormalizedValue(ny, axisY);
                }

                this.intpos.x = nx;
                this.intpos.y = ny;
                this.properties.valueX = roundToDecimals(denormalizeValue(nx, axisX), axisX.decimals);
                this.properties.valueY = roundToDecimals(denormalizeValue(ny, axisY), axisY.decimals);

                refreshCoreState(isDragging);
                draw();
                // When dragging, use light redraw (second arg false means not set dirty graph)
                this.setDirtyCanvas?.(true, false);
            };

            const handlePointerMove = (event) => {
                if (!isDragging) return;
                event.preventDefault();
                updateValuesFromPos(event.clientX, event.clientY, event.shiftKey, true);
            };

            const finishPointerDrag = (pointerId) => {
                if (!isDragging) return;
                isDragging = false;
                cachedRect = null;

                // Final sync and reactivity trigger
                refreshCoreState(false);
                this.setDirtyCanvas?.(true, true);

                if (pointerId != null && canvas.hasPointerCapture?.(pointerId)) {
                    canvas.releasePointerCapture(pointerId);
                }
                canvas.removeEventListener("pointermove", handlePointerMove);
                canvas.removeEventListener("pointerup", handlePointerUp);
                canvas.removeEventListener("pointercancel", handlePointerCancel);
            };

            const handleWheel = (event) => event.stopPropagation();
            const handlePointerUp = (event) => finishPointerDrag(event.pointerId);
            const handlePointerCancel = (event) => finishPointerDrag(event.pointerId);

            canvas.addEventListener("pointerdown", (event) => {
                const rect = canvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;

                if (x > rect.width - 24 && y < 40) {
                    return;
                }

                if (event.button !== 0) return;
                isDragging = true;
                event.preventDefault();
                event.stopPropagation();
                canvas.setPointerCapture(event.pointerId);
                updateRect();
                updateValuesFromPos(event.clientX, event.clientY, event.shiftKey);

                canvas.addEventListener("pointermove", handlePointerMove, { passive: false });
                canvas.addEventListener("pointerup", handlePointerUp);
                canvas.addEventListener("pointercancel", handlePointerCancel);
            });

            canvas.addEventListener("wheel", handleWheel, { passive: true });

            domWidget = this.addDOMWidget("slider2d_ui", "SLIDER2D", container, {
                getMinHeight: () => getMinCanvasHeight(),
                onDraw: () => draw(),
            });

            domWidget.computeLayoutSize = () => ({
                minHeight: getMinCanvasHeight(),
                minWidth: NODES2_MIN_CANVAS_W,
            });

            // Hide input slot DOM elements but keep output slots visible.
            // Set widgets_start_y = 0 so the DOM widget starts at the title bar.
            const slotStyleId = `xe-slider2d-slot-style-${this.id}`;
            const applyHideConnections = () => {
                let perNodeStyle = document.getElementById(slotStyleId);
                if (!perNodeStyle) {
                    perNodeStyle = document.createElement("style");
                    perNodeStyle.id = slotStyleId;
                    document.head.appendChild(perNodeStyle);
                }
                perNodeStyle.innerHTML = `
                    [data-node-id="${this.id}"] .lg-slot.lg-slot--input {
                        position: absolute !important;
                        opacity: 0 !important;
                        pointer-events: none !important;
                        width: 10px !important;
                        height: 20px !important;
                        margin: 0 !important;
                        padding: 0 !important;
                    }
                `;
                this.widgets_start_y = 0;
            };
            applyHideConnections();

            const hideDataWidgets = () => {
                if (!this.widgets) return;
                for (const widget of this.widgets) {
                    if (widget.name === "slider2d_ui") continue;
                    widget.type = "hidden";
                    widget.hidden = true;
                    widget.options = widget.options || {};
                    widget.options.hidden = true;
                    if (widget.computeSize) {
                        widget.computeSize = () => [0, -4];
                    }
                }
            };

            hideDataWidgets();
            refreshCoreState();
            updateOutputTypes();

            // applyContainerSize no longer depends on domWidget.y (static offset).
            // Just defer one frame for the canvas to get its final dimensions.
            hideDataWidgets();
            applyContainerSize(this.size[1]);
            requestAnimationFrame(() => requestDraw());

            const originalOnPropertyChanged = this.onPropertyChanged;
            this.onPropertyChanged = function () {
                if (originalOnPropertyChanged) {
                    originalOnPropertyChanged.apply(this, arguments);
                }
                syncIntposFromProperties();
                refreshCoreState();
                updateOutputTypes();
                requestDraw();
                this.setDirtyCanvas?.(true, false);
            };

            this.onResize = function (size) {
                if (!isNodes2Environment()) {
                    const minNodeHeight = getMinCanvasHeight() + 30 + getLegacyBottomGap();
                    size[1] = Math.max(size[1], minNodeHeight);
                }
                finishPointerDrag();
                debugLog("onResize", { incomingSize: [...size] });
                applyContainerSize(size[1]);
                requestAnimationFrame(() => {
                    requestDraw();
                    debugLog("onResize:after", { finalSize: [...size] });
                });
            };

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                const slotStyle = document.getElementById(slotStyleId);
                slotStyle?.remove();
                if (originalOnRemoved) {
                    originalOnRemoved.apply(this, arguments);
                }
                finishPointerDrag();
                canvas.removeEventListener("pointermove", handlePointerMove);
                canvas.removeEventListener("pointerup", handlePointerUp);
                canvas.removeEventListener("pointercancel", handlePointerCancel);
                canvas.removeEventListener("wheel", handleWheel);
            };

            this.serialize_widgets = true;

            const originalOnConfigure = this.onConfigure;
            this.onConfigure = function (info) {
                if (originalOnConfigure) {
                    originalOnConfigure.apply(this, arguments);
                }
                if (this.properties) {
                    cleanupLegacyProperties();
                }

                // Hydrate from ComfyUI widgets_values first if they exist
                if (info && Array.isArray(info.widgets_values) && this.widgets) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        const widget = this.widgets[i];
                        const wval = info.widgets_values[i];
                        if (widget.name === "X" && typeof wval === "number") {
                            this.properties.valueX = wval;
                            if (wval > this.properties.maxX) this.properties.maxX = wval;
                            if (wval < this.properties.minX) this.properties.minX = wval;
                        }
                        if (widget.name === "Y" && typeof wval === "number") {
                            this.properties.valueY = wval;
                            if (wval > this.properties.maxY) this.properties.maxY = wval;
                            if (wval < this.properties.minY) this.properties.minY = wval;
                        }
                    }
                }

                syncIntposFromProperties();
                refreshCoreState();
                updateOutputTypes();
                hideDataWidgets();
                applyContainerSize(this.size[1]);
                requestAnimationFrame(() => requestDraw());
            };

            const originalOnSerialize = this.onSerialize;
            this.onSerialize = function (info) {
                if (originalOnSerialize) {
                    originalOnSerialize.apply(this, arguments);
                }

                info.properties = info.properties || {};
                Object.assign(info.properties, this.properties);

                if (Array.isArray(info.widgets_values) && this.widgets?.length) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        const widget = this.widgets[i];
                        if (widget.name === "X") {
                            info.widgets_values[i] = this.properties.valueX;
                        }
                        if (widget.name === "Y") {
                            info.widgets_values[i] = this.properties.valueY;
                        }
                    }
                }
            };

            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.apply(this, arguments);
                }

                syncIntposFromProperties();
                refreshCoreState();
                updateOutputTypes();
            };
        };
    },
});
