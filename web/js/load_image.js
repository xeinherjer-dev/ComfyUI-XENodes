import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "XENodes.LoadImageFromFolder",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "XENodes.LoadImageFromFolder") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const node = this;

                // Drag & Drop support
                this.onDragOver = function (e) {
                    if (e?.dataTransfer) {
                        return true;
                    }
                    return false;
                };

                this.onDragDrop = async function (e) {
                    if (!e?.dataTransfer) return false;

                    const pathWidget = node.widgets?.find((w) => w.name === "path");
                    let droppedPath = "";

                    // 1. Check for Electron / ComfyUI Desktop app file path
                    const files = e.dataTransfer.files;
                    if (files && files.length > 0) {
                        const file = files[0];
                        if (file.path) {
                            droppedPath = file.path;
                        }
                    }

                    // 2. Check for text/plain or text/uri-list (dragged from address bar, VSCode, terminal, etc.)
                    if (!droppedPath) {
                        const text = e.dataTransfer.getData("text/plain") || e.dataTransfer.getData("text/uri-list");
                        if (text) {
                            let cleaned = text.trim();
                            // Handle file:/// URL format
                            if (cleaned.startsWith("file://")) {
                                try {
                                    const url = new URL(cleaned);
                                    let pathname = decodeURIComponent(url.pathname);
                                    // Windows path: /C:/path -> C:/path
                                    if (/^\/[a-zA-Z]:/.test(pathname)) {
                                        pathname = pathname.slice(1);
                                    }
                                    cleaned = pathname;
                                } catch (_) {
                                    cleaned = cleaned.replace(/^file:\/\/\/?/, "");
                                    cleaned = decodeURIComponent(cleaned);
                                }
                            }
                            // Strip surrounding quotes
                            cleaned = cleaned.replace(/^["']|["']$/g, "").trim();
                            if (cleaned) {
                                droppedPath = cleaned;
                            }
                        }
                    }

                    if (droppedPath && pathWidget) {
                        pathWidget.value = droppedPath;
                        if (pathWidget.callback) {
                            pathWidget.callback(droppedPath);
                        }
                        node.setDirtyCanvas(true, true);
                        return true;
                    }

                    return false;
                };

                return r;
            };
        }
    },
});
