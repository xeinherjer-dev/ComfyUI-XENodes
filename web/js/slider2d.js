import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XENodes.Slider2D";
const NODE_NAME = "XENodes.Slider2D";

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
            this.properties.valueX = this.properties.valueX ?? 512;
            this.properties.valueY = this.properties.valueY ?? 512;
            this.properties.minX = this.properties.minX ?? 0;
            this.properties.maxX = this.properties.maxX ?? 1024;
            this.properties.minY = this.properties.minY ?? 0;
            this.properties.maxY = this.properties.maxY ?? 1024;
            this.properties.stepX = this.properties.stepX ?? 1;
            this.properties.stepY = this.properties.stepY ?? 1;
            this.properties.decimalsX = this.properties.decimalsX ?? 0;
            this.properties.decimalsY = this.properties.decimalsY ?? 0;
            this.properties.snap = this.properties.snap ?? true;
            this.properties.dots = this.properties.dots ?? true;
            this.properties.frame = this.properties.frame ?? true;
            this.properties.frameAlert = this.properties.frameAlert ?? 0;

            this.intpos = {
                x: (this.properties.valueX - this.properties.minX) / (this.properties.maxX - this.properties.minX),
                y: (this.properties.valueY - this.properties.minY) / (this.properties.maxY - this.properties.minY)
            };

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
                        margin: -40px 40px 4px 4px; 
                        width: auto;
                        height: 140px; 
                        overflow: hidden;
                        display: flex;
                        flex-direction: column;
                        backdrop-filter: blur(4px);
                    }
                    .xe-slider2d-canvas {
                        width: 100%;
                        height: 100%; /* ← 追加: 親のサイズに厳密に従わせる */
                        min-height: 0; /* ← 追加: Flexbox内での無限拡張を防ぐ */
                        display: block; /* ← 追加: 余計な隙間を消す */
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

                // Grid Dots
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

                // Handle and Frame
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

                // Update port labels
                const valXText = this.properties.valueX.toFixed(this.properties.decimalsX);
                const valYText = this.properties.valueY.toFixed(this.properties.decimalsY);
                if (this.outputs) {
                    if (this.outputs[0]) this.outputs[0].label = valXText;
                    if (this.outputs[1]) this.outputs[1].label = valYText;
                }

                // ここにあった this.setDirtyCanvas は無限ループの原因になるため削除しました
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

                const pX = Math.pow(10, this.properties.decimalsX);
                const pY = Math.pow(10, this.properties.decimalsY);
                this.properties.valueX = Math.round(this.properties.valueX * pX) / pX;
                this.properties.valueY = Math.round(this.properties.valueY * pY) / pY;

                if (this.widgets) {
                    for (const w of this.widgets) {
                        if (w.name === "Xi") w.value = Math.floor(this.properties.valueX);
                        if (w.name === "Xf") w.value = this.properties.valueX;
                        if (w.name === "Yi") w.value = Math.floor(this.properties.valueY);
                        if (w.name === "Yf") w.value = this.properties.valueY;
                    }
                }

                draw();
                // 値が更新された時だけ LiteGraph に再描画を通知する
                if (this.setDirtyCanvas) this.setDirtyCanvas(true, false);
            };

            let isDragging = false;

            const handleMouseMove = (e) => {
                if (!isDragging) return;
                updateValuesFromPos(e.clientX, e.clientY, e.shiftKey);
            };

            const handleMouseUp = () => {
                isDragging = false;
                window.removeEventListener("mousemove", handleMouseMove);
                window.removeEventListener("mouseup", handleMouseUp);
            };

            canvas.addEventListener("mousedown", (e) => {
                isDragging = true;
                e.preventDefault();
                e.stopPropagation();
                updateRect();
                updateValuesFromPos(e.clientX, e.clientY, e.shiftKey);

                window.addEventListener("mousemove", handleMouseMove);
                window.addEventListener("mouseup", handleMouseUp);
            });

            canvas.addEventListener("wheel", (e) => e.stopPropagation());

            const domWidget = this.addDOMWidget("slider2d_ui", "SLIDER2D", container);
            domWidget.draw = draw;
            domWidget.computeSize = (width) => [width, 160];

            setTimeout(() => {
                const hideWidgets = () => {
                    if (this.widgets) {
                        for (const w of this.widgets) {
                            if (w.name !== "slider2d_ui") {
                                w.type = "hidden";
                                w.hidden = true;
                                w.computeSize = () => [0, -4];
                            }
                        }
                    }
                };

                if (this.widgets) {
                    const idx = this.widgets.indexOf(domWidget);
                    if (idx > 0) {
                        this.widgets.splice(idx, 1);
                        this.widgets.unshift(domWidget);
                    }
                }
                hideWidgets();
                updateRect();
                if (this.onResize) this.onResize(this.size);
            }, 50);

            const originalOnPropertyChanged = this.onPropertyChanged;
            this.onPropertyChanged = function (name, value) {
                if (originalOnPropertyChanged) originalOnPropertyChanged.apply(this, arguments);
                this.intpos.x = (this.properties.valueX - this.properties.minX) / (this.properties.maxX - this.properties.minX);
                this.intpos.y = (this.properties.valueY - this.properties.minY) / (this.properties.maxY - this.properties.minY);
                draw();
                if (this.setDirtyCanvas) this.setDirtyCanvas(true, false);
            };

            // Resize observer
            this.onResize = function (size) {
                if (container) {
                    const targetH = Math.max(100, size[1] - 30);
                    if (container.style.height !== targetH + "px") {
                        container.style.height = targetH + "px";
                    }
                }
                requestAnimationFrame(() => {
                    updateRect();
                    draw();
                });
            };

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (originalOnRemoved) originalOnRemoved.apply(this, arguments);
                window.removeEventListener("mousemove", handleMouseMove);
                window.removeEventListener("mouseup", handleMouseUp);
            };
        };
    }
});