import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "XENodes.LoadImageFromFolder";
const NODE_NAME = "XENodes.LoadImageFromFolder";

function injectContextMenuStyles() {
    if (typeof document === "undefined" || document.getElementById("xe-load-image-contextmenu-styles")) return;
    const style = document.createElement("style");
    style.id = "xe-load-image-contextmenu-styles";
    style.textContent = `
        .litecontextmenu input.comfy-context-menu-filter {
            position: sticky !important;
            top: 0 !important;
            z-index: 100 !important;
            background-color: var(--comfy-input-bg, #222) !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
        }
    `;
    document.head.appendChild(style);
}

const DATE_REGEX = /(\d{4}[-_/]\d{2}[-_/]\d{2})/;

function sortFolderList(folders, sort_by = "newest_first") {
    if (!Array.isArray(folders)) return [];
    const copy = [...folders];
    if (sort_by === "oldest_first") {
        copy.sort((a, b) => {
            const ma = DATE_REGEX.exec(a);
            const mb = DATE_REGEX.exec(b);
            const da = ma ? ma[1].replace(/[/_]/g, "-") : null;
            const db = mb ? mb[1].replace(/[/_]/g, "-") : null;
            if (da && db) return da.localeCompare(db) || a.localeCompare(b);
            if (da) return -1;
            if (db) return 1;
            return a.localeCompare(b);
        });
    } else {
        // newest_first
        copy.sort((a, b) => {
            const ma = DATE_REGEX.exec(a);
            const mb = DATE_REGEX.exec(b);
            const da = ma ? ma[1].replace(/[/_]/g, "-") : null;
            const db = mb ? mb[1].replace(/[/_]/g, "-") : null;
            if (da && db) return db.localeCompare(da) || a.localeCompare(b);
            if (da) return -1;
            if (db) return 1;
            return a.localeCompare(b);
        });
    }
    return copy;
}

let isContextMenuHooked = false;
function setupContextMenuAutoScroll() {
    if (isContextMenuHooked || typeof LiteGraph === "undefined" || !LiteGraph.ContextMenu) return;
    isContextMenuHooked = true;

    const origContextMenu = LiteGraph.ContextMenu;
    LiteGraph.ContextMenu = function (values, options) {
        const ctx = new origContextMenu(values, options);
        if (ctx && ctx.root) {
            // Auto-scroll to selected entry when dropdown is opened, only if needed
            requestAnimationFrame(() => {
                setTimeout(() => {
                    if (!ctx.root) return;
                    const selected = ctx.root.querySelector('.litemenu-entry[style*="background-color"]')
                        || ctx.root.querySelector('.litemenu-entry.selected');
                    if (selected) {
                        const rect = selected.getBoundingClientRect();
                        const rootRect = ctx.root.getBoundingClientRect();
                        // Scroll only if out of visible bounds
                        if (rect.top < rootRect.top + 32 || rect.bottom > rootRect.bottom) {
                            selected.scrollIntoView({ block: "nearest", behavior: "instant" });
                        }
                    }
                }, 30);
            });

            // Follow-through scrolling when using arrow keys
            ctx.root.addEventListener("keydown", (e) => {
                if (e.key === "ArrowUp" || e.key === "ArrowDown") {
                    setTimeout(() => {
                        if (!ctx.root) return;
                        const cur = ctx.root.querySelector('.litemenu-entry[style*="background-color"]');
                        if (cur) {
                            cur.scrollIntoView({ block: "nearest", behavior: "instant" });
                        }
                    }, 10);
                }
            }, true);
        }
        return ctx;
    };
    LiteGraph.ContextMenu.prototype = origContextMenu.prototype;
}

app.registerExtension({
    name: EXTENSION_NAME,
    init() {
        injectContextMenuStyles();
        setupContextMenuAutoScroll();
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        injectContextMenuStyles();
        setupContextMenuAutoScroll();

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
         * Fetches latest folder list from backend and updates the path combo widget.
         * @param {object} node - LiteGraph node instance
         * @param {boolean} force - Whether to force bypass server-side cache
         */
        async function refreshFolders(node, force = false) {
            try {
                const folderSortWidget = node.widgets?.find((w) => w.name === "folder_sort");
                const folderSort = folderSortWidget?.value || "newest_first";
                const params = new URLSearchParams();
                if (force) params.append("force", "true");
                params.append("sort_by", folderSort);

                const url = `/xenodes/load_image/folders?${params.toString()}`;
                const resp = await fetch(url);
                if (resp.ok) {
                    const folders = await resp.json();
                    const pathWidget = node.widgets?.find((w) => w.name === "path");
                    if (pathWidget && Array.isArray(folders) && folders.length > 0) {
                        pathWidget.options = pathWidget.options || {};
                        pathWidget.options.values = folders;
                        if (!pathWidget.value || !folders.includes(pathWidget.value)) {
                            pathWidget.value = folders[0];
                        }
                        app.graph.setDirtyCanvas(true, true);
                    }
                }
            } catch (e) {
                console.warn("[xenodes] Failed to fetch folder list:", e);
            }
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

            // Hook folder_sort callback to immediately re-sort path options
            const folderSortWidget = node.widgets?.find((w) => w.name === "folder_sort");
            if (folderSortWidget && !folderSortWidget._xe_hooked) {
                folderSortWidget._xe_hooked = true;
                const origSortCallback = folderSortWidget.callback;
                folderSortWidget.callback = function () {
                    const r = origSortCallback ? origSortCallback.apply(this, arguments) : undefined;
                    const pathWidget = node.widgets?.find((w) => w.name === "path");
                    const currentPath = pathWidget?.value;
                    refreshFolders(node, false).then(() => {
                        if (pathWidget && currentPath && pathWidget.options?.values?.includes(currentPath)) {
                            pathWidget.value = currentPath;
                        }
                        fetchAndShowPreview(node);
                    });
                    return r;
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

        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const r = originalGetExtraMenuOptions ? originalGetExtraMenuOptions.apply(this, arguments) : undefined;
            options.push({
                content: "Refresh Folders",
                callback: () => {
                    refreshFolders(this, true).then(() => {
                        fetchAndShowPreview(this);
                    });
                },
            });
            return r;
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
            setupPreviewHooks(this);
            refreshFolders(this);
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
            refreshFolders(this);
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
