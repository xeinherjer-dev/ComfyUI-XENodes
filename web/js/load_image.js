import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XENodes.LoadImageFromFolder";
const NODE_NAME = "XENodes.LoadImageFromFolder";

app.registerExtension({
    name: EXTENSION_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        /**
         * Fetches preview image from backend API and displays it on the node.
         * @param {object} node - LiteGraph node instance
         */
        function fetchAndShowPreview(node) {
            if (!node || !node.widgets) return;

            const pathWidget = node.widgets.find((w) => w.name === "path");
            const indexWidget = node.widgets.find((w) => w.name === "index");
            const sortByWidget = node.widgets.find((w) => w.name === "sort_by");
            const reverseWidget = node.widgets.find((w) => w.name === "reverse");
            const subfoldersWidget = node.widgets.find((w) => w.name === "subfolders");

            const path = (pathWidget?.value || "").trim();
            if (!path) {
                if (node.imgs && node.imgs.length > 0) {
                    node.imgs = null;
                    app.graph.setDirtyCanvas(true, true);
                }
                return;
            }

            const index = Number(indexWidget?.value ?? 0);
            const sortBy = sortByWidget?.value ?? "name";
            const reverse = Boolean(reverseWidget?.value ?? false);
            const subfolders = Boolean(subfoldersWidget?.value ?? false);

            const params = new URLSearchParams({
                path: path,
                index: index.toString(),
                sort_by: sortBy,
                reverse: reverse.toString(),
                subfolders: subfolders.toString(),
                t: Date.now().toString(),
            });

            // Track request ID to discard stale responses from rapid index changing
            const reqId = (node._xe_last_req_id = (node._xe_last_req_id || 0) + 1);

            const img = new Image();
            img.onload = () => {
                if (node._xe_last_req_id !== reqId) return;

                node.imgs = [img];
                node.setSizeForImage?.();
                app.graph.setDirtyCanvas(true, true);
            };
            img.onerror = () => {
                if (node._xe_last_req_id !== reqId) return;
                // Leave previous image or clear if desired
            };
            img.src = `/xenodes/load_image/preview?${params.toString()}`;
        }

        /**
         * Sets up debounced preview updater and hooks widget callbacks.
         * @param {object} node - LiteGraph node instance
         */
        function setupPreviewHooks(node) {
            if (!node._xe_update_preview) {
                let debounceTimer = null;
                node._xe_update_preview = () => {
                    if (debounceTimer) clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(() => {
                        fetchAndShowPreview(node);
                    }, 120);
                };
            }

            // Hook widget callbacks for real-time preview updates
            for (const widget of node.widgets || []) {
                if (["path", "index", "sort_by", "reverse", "subfolders"].includes(widget.name)) {
                    if (!widget._xe_preview_hooked) {
                        widget._xe_preview_hooked = true;
                        const origCallback = widget.callback;
                        widget.callback = function () {
                            const r = origCallback ? origCallback.apply(this, arguments) : undefined;
                            node._xe_update_preview?.();
                            return r;
                        };
                    }
                }
            }
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
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

            setupPreviewHooks(this);
            // Fetch initial preview if path is set
            requestAnimationFrame(() => {
                fetchAndShowPreview(this);
            });
            return r;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
            setupPreviewHooks(this);
            // Fetch preview when workflow is loaded / configured
            requestAnimationFrame(() => {
                setTimeout(() => {
                    fetchAndShowPreview(this);
                }, 50);
            });
            return r;
        };
    },
});
