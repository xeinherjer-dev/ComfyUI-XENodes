import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XENodes.Slider2D";
const NODE_NAME = "XENodes.Slider2D";

const DEFAULT_SIZE = [240, 312];
const NODES2_MIN_CANVAS_W = 50;
const NODES2_MIN_CANVAS_H = 50;
const LEGACY_MIN_CANVAS_H = 80;
const LEGACY_SIDE_MARGIN = 4;
const LEGACY_BOTTOM_GAP = 16;
const LEGACY_FALLBACK_TOP_OFFSET = 80;

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
                console.log("[Slider2D]", label, {
                    mode: isNodes2Environment() ? "nodes2" : "legacy",
                    nodeSize: this.size ? [...this.size] : null,
                    widgetY: domWidget?.y ?? null,
                    widgetH: domWidget?.computedHeight ?? null,
                    containerH: container.clientHeight,
                    canvasH: canvas.clientHeight,
                    ...extra,
                });
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
            let lastLabelUpdate = 0;

            const syncWidgets = () => {
                if (!this.widgets) return;
                for (const widget of this.widgets) {
                    if (widget.name === "X") widget.value = this.properties.valueX;
                    if (widget.name === "Y") widget.value = this.properties.valueY;
                }
            };

            const updatePortLabels = (isDragging = false) => {
                if (!this.outputs) return;

                const axisX = getAxisState("X");
                const axisY = getAxisState("Y");
                const valueX = this.properties.valueX.toFixed(axisX.decimals);
                const valueY = this.properties.valueY.toFixed(axisY.decimals);
                let changed = false;

                const now = Date.now();
                // In Nodes 2.0, we need to spread the object to trigger Vue reactivity, even during drag.
                // We throttle this to ~30fps (32ms) to balance UI responsiveness and performance (GC/re-renders).
                const shouldSpread = !isDragging || (now - lastLabelUpdate > 32);

                if (this.outputs[0] && this.outputs[0].label !== valueX) {
                    this.outputs[0].label = valueX;
                    if (shouldSpread) {
                        this.outputs[0] = { ...this.outputs[0] };
                    }
                    changed = true;
                }

                if (this.outputs[1] && this.outputs[1].label !== valueY) {
                    this.outputs[1].label = valueY;
                    if (shouldSpread) {
                        this.outputs[1] = { ...this.outputs[1] };
                    }
                    changed = true;
                }

                if (changed) {
                    if (shouldSpread) {
                        this.outputs = [...this.outputs];
                        lastLabelUpdate = now;
                    }
                    if (!isDragging) {
                        this.setDirtyCanvas?.(true, true);
                    }
                }

                syncWidgets();
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
                    container.style.margin = "0";
                    container.style.height = "100%";
                    container.style.minHeight = "0";
                    container.style.maxHeight = "100%";
                    container.style.alignSelf = "stretch";
                    debugLog("applyContainerSize:nodes2", { requestedNodeHeight: nodeHeight });
                    return;
                }

                const topOffset = domWidget?.y || LEGACY_FALLBACK_TOP_OFFSET;
                const bottomGap = getLegacyBottomGap();
                const canvasHeight = Math.max(
                    nodeHeight - topOffset - bottomGap,
                    getMinCanvasHeight()
                );

                container.style.width = "calc(100% - 8px)";
                container.style.margin = `0px ${LEGACY_SIDE_MARGIN}px ${bottomGap}px ${LEGACY_SIDE_MARGIN}px`;
                container.style.height = canvasHeight + "px";
                container.style.minHeight = canvasHeight + "px";
                container.style.maxHeight = canvasHeight + "px";
                container.style.alignSelf = "flex-start";

                debugLog("applyContainerSize:legacy", {
                    requestedNodeHeight: nodeHeight,
                    appliedHeight: canvasHeight,
                    bottomGap,
                    topOffset,
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

                updatePortLabels(isDragging);
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
                updatePortLabels(false);
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
            updatePortLabels();
            updateOutputTypes();

            let initFrames = 0;
            const initLayout = () => {
                if (initFrames++ < 2) {
                    requestAnimationFrame(initLayout);
                    return;
                }

                hideDataWidgets();
                syncIntposFromProperties();
                updatePortLabels();
                updateOutputTypes();
                applyContainerSize(this.size[1]);
                requestDraw();
            };
            requestAnimationFrame(initLayout);

            const originalOnPropertyChanged = this.onPropertyChanged;
            this.onPropertyChanged = function () {
                if (originalOnPropertyChanged) {
                    originalOnPropertyChanged.apply(this, arguments);
                }
                syncIntposFromProperties();
                updatePortLabels();
                updateOutputTypes();
                requestDraw();
                this.setDirtyCanvas?.(true, false);
            };

            this.onResize = function (size) {
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
                if (originalOnRemoved) {
                    originalOnRemoved.apply(this, arguments);
                }
                finishPointerDrag();
                canvas.removeEventListener("pointermove", handlePointerMove);
                canvas.removeEventListener("pointerup", handlePointerUp);
                canvas.removeEventListener("pointercancel", handlePointerCancel);
                canvas.removeEventListener("wheel", handleWheel);
            };

            const originalOnConfigure = this.onConfigure;
            this.onConfigure = function () {
                if (originalOnConfigure) {
                    originalOnConfigure.apply(this, arguments);
                }
                if (this.properties) {
                    cleanupLegacyProperties();
                }
                syncIntposFromProperties();
                updatePortLabels();
                updateOutputTypes();
                requestDraw();
            };

            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.apply(this, arguments);
                }

                syncIntposFromProperties();
                updatePortLabels();
                updateOutputTypes();
            };
        };
    },
});
