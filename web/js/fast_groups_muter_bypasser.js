/**
 * XENodes Fast Groups Muter & Bypasser
 *
 * Displays toggles for each group in the workflow, allowing quick muting or
 * bypassing of all nodes within a group.
 *
 * Features:
 * - Replicates ComfyUI's standard capsule toggle switch (true/false) perfectly using DOM widgets
 * - Retains the Arrow Navigation button (chevron right) to quickly jump to the group location
 * - Smooth in-place differential rendering (reconciliation) to prevent blinking/flickering
 */

import { app } from "../../../scripts/app.js";

const NODE_NAME      = "XENodes.FastGroupsMuterBypasser";
const EXTENSION_NAME = "XENodes.FastGroupsMuterBypasser";

// Node mode constants
const MODE_ALWAYS = 0; // Active
const MODE_NEVER  = 2; // Muted
const MODE_BYPASS = 4; // Bypassed

const ROW_HEIGHT = 36; // px per group row



// ------------------------------------------------------------------
// Utility helpers
// ------------------------------------------------------------------

/** All direct-child LGraphNodes of a group (flat, no recursion into Subgraphs). */
function getGroupNodes(group) {
    if (!group._children) return [];
    return Array.from(group._children).filter((c) => c instanceof LGraphNode);
}

/** Apply mode to a flat node list (no recursion). */
function applyModeToNodes(nodes, mode) {
    for (const node of nodes) node.mode = mode;
}

/** Collect groups depending on showAllGraphs property and prevent double counting. */
function getAllGroups(node) {
    const groups = new Set();
    const showAll = node.properties?.showAllGraphs === true;

    if (!showAll) {
        // Only groups in the graph where this node resides
        if (node.graph && node.graph._groups) {
            for (const g of node.graph._groups) groups.add(g);
        }
    } else {
        // All groups in the root graph and any subgraphs
        const root = app.graph;
        if (!root) return [];
        const graphsToProcess = [root];
        const processed = new Set();

        while (graphsToProcess.length > 0) {
            const g = graphsToProcess.pop();
            if (!g || processed.has(g)) continue;
            processed.add(g);

            if (g._groups) {
                for (const grp of g._groups) groups.add(grp);
            }

            if (g.subgraphs && typeof g.subgraphs.values === 'function') {
                for (const sub of g.subgraphs.values()) {
                    graphsToProcess.push(sub);
                }
            }
        }
    }
    return Array.from(groups);
}

/**
 * Re-detect which nodes lie inside a group.
 * Only updates _children (Set). Does NOT write to group.nodes (getter-only).
 */
function recomputeGroupChildren(group) {
    if (!group.graph) return;
    if (!group._children) group._children = new Set();
    group._children.clear();

    const gb = group._bounding
        ?? (group._pos && group._size
            ? [group._pos[0], group._pos[1], group._size[0], group._size[1]]
            : null);
    if (!gb) return;

    for (const node of (group.graph.nodes ?? [])) {
        const b = node.getBounding?.();
        if (!b) continue;
        const cx = b[0] + b[2] * 0.5;
        const cy = b[1] + b[3] * 0.5;
        if (cx >= gb[0] && cx < gb[0] + gb[2] &&
            cy >= gb[1] && cy < gb[1] + gb[3]) {
            group._children.add(node);
        }
    }
}

// ------------------------------------------------------------------
// DOM row builder with capsule toggle switch
// ------------------------------------------------------------------

function createRow(row, showNav, node) {
    const rowEl = document.createElement("div");
    rowEl.style.cssText = `
        display: flex; align-items: center; gap: 8px;
        padding: 4px 6px; border-radius: 6px;
        background: rgba(0, 0, 0, 0.25);
        cursor: pointer; user-select: none;
        min-height: ${ROW_HEIGHT - 6}px; box-sizing: border-box;
    `;

    // Group name label
    const label = document.createElement("span");
    label.style.cssText = `
        flex: 1; font-size: 12px; overflow: hidden;
        text-overflow: ellipsis; white-space: nowrap;
        color: #888; transition: color 0.12s;
    `;

    // Capsule toggle switch (Outer frame)
    const toggleContainer = document.createElement("div");
    toggleContainer.style.cssText = `
        display: flex; align-items: center;
        width: 48px; height: 22px; border-radius: 11px;
        background: #181818; border: 1.5px solid #333;
        padding: 0; position: relative; cursor: pointer;
        transition: background 0.15s, border-color 0.15s;
        box-sizing: border-box; flex-shrink: 0;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
    `;

    // Status text (true / false) inside toggle
    const statusText = document.createElement("span");
    statusText.style.cssText = `
        font-size: 11px; font-family: sans-serif; font-weight: 600;
        position: absolute; transition: all 0.15s ease-in-out;
        pointer-events: none; user-select: none;
    `;

    // Toggle knob (circle)
    const knob = document.createElement("div");
    knob.style.cssText = `
        width: 14px; height: 14px; border-radius: 50%;
        background: #8fa0ac; position: absolute;
        transition: left 0.15s cubic-bezier(0.25, 0.8, 0.25, 1);
        pointer-events: none;
        box-shadow: 0 1px 2px rgba(0,0,0,0.4);
    `;

    toggleContainer.appendChild(statusText);
    toggleContainer.appendChild(knob);

    const updateVisuals = (toggled) => {
        label.textContent = row.group.title ?? "(group)";
        label.style.color = toggled ? "#eee" : "#888";

        if (toggled) {
            toggleContainer.style.background = "#1c1d20";
            toggleContainer.style.borderColor = "#44464d";
            knob.style.left = "29px"; // Right-aligned
            statusText.textContent = "on";
            statusText.style.color = "#eee";
            statusText.style.left = "8px"; // Text on left
            statusText.style.opacity = "1";
        } else {
            toggleContainer.style.background = "#141416";
            toggleContainer.style.borderColor = "#2c2d30";
            knob.style.left = "4px"; // Left-aligned
            statusText.textContent = "off";
            statusText.style.color = "#555860";
            statusText.style.left = "21px"; // Text on right
            statusText.style.opacity = "1";
        }
    };
    updateVisuals(row.toggled);

    const doToggle = (e) => {
        e.stopPropagation();
        recomputeGroupChildren(row.group);
        const nodes       = getGroupNodes(row.group);
        const hasActive   = nodes.some((n) => n.mode === MODE_ALWAYS);
        let   next        = !hasActive;
        const modeOff     = (node.properties?.actionMode === "bypass") ? MODE_BYPASS : MODE_NEVER;
        const restriction = node.properties?.toggleRestriction ?? "default";

        if (restriction === "always one" && !next) {
            const othersActive = node._groupsData.some(r => r.group !== row.group && r.toggled);
            if (!othersActive) {
                return; // Cannot turn off the only active group
            }
        }

        if (next && (restriction === "max one" || restriction === "always one")) {
            for (const r of (node._groupsData ?? [])) {
                if (r.group !== row.group && r.toggled) {
                    recomputeGroupChildren(r.group);
                    applyModeToNodes(getGroupNodes(r.group), modeOff);
                    r.toggled = false;
                }
            }
            setTimeout(() => node._rebuildGroups?.(), 0);
        }

        applyModeToNodes(nodes, next ? MODE_ALWAYS : modeOff);
        row.toggled = next;
        updateVisuals(next);
        row.group.graph?.setDirtyCanvas?.(true, false);
        app.canvas?.setDirty?.(true, true);
    };

    rowEl.appendChild(label);
    rowEl.appendChild(toggleContainer);
    rowEl.addEventListener("click", doToggle);

    // Attach updater and group reference for DOM-reuse (differential update)
    rowEl.updateVisuals = updateVisuals;
    rowEl.group = row.group;

    if (showNav) {
        const navBtn = document.createElement("button");
        navBtn.title = `Go to: ${row.group.title}`;
        navBtn.style.cssText = `
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #a0a0a0;
            border-radius: 4px;
            padding: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            flex-shrink: 0;
            transition: all 0.15s ease-in-out;
        `;
        
        // Hover effects
        navBtn.addEventListener("mouseenter", () => {
            navBtn.style.background = "rgba(255, 255, 255, 0.15)";
            navBtn.style.borderColor = "rgba(255, 255, 255, 0.35)";
            navBtn.style.color = "#ffffff";
        });
        navBtn.addEventListener("mouseleave", () => {
            navBtn.style.background = "rgba(255, 255, 255, 0.05)";
            navBtn.style.borderColor = "rgba(255, 255, 255, 0.15)";
            navBtn.style.color = "#a0a0a0";
        });

        // Lucide-like Chevron Right SVG
        navBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="pointer-events: none; display: block;">
                <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
        `;

        navBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            const canvas = app.canvas;
            if (!canvas) return;
            canvas.centerOnNode?.(row.group);
            const scale = canvas.ds?.scale ?? 1;
            const zx = canvas.canvas.width  / (row.group._size?.[0] ?? 200) - 0.02;
            const zy = canvas.canvas.height / (row.group._size?.[1] ?? 200) - 0.02;
            canvas.setZoom?.(Math.min(scale, zx, zy),
                [canvas.canvas.width * 0.5, canvas.canvas.height * 0.5]);
            canvas.setDirty?.(true, true);
        });
        rowEl.appendChild(navBtn);
    }

    return rowEl;
}

// ------------------------------------------------------------------
// Extension
// ------------------------------------------------------------------

app.registerExtension({
    name: EXTENSION_NAME,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            this.properties = this.properties ?? {};
            this.properties.actionMode = this.properties.actionMode ?? "mute";
            this.properties.showAllGraphs = this.properties.showAllGraphs ?? false;
            this.properties.matchTitle = this.properties.matchTitle ?? "";
            this.properties.matchColors = this.properties.matchColors ?? "";
            this.properties.toggleRestriction = this.properties.toggleRestriction ?? "default";

            this.addProperty?.("actionMode", "mute", "enum", { values: ["mute", "bypass"] });
            this.addProperty?.("showAllGraphs", false, "boolean");
            this.addProperty?.("matchTitle", "",     "string");
            this.addProperty?.("matchColors", "", "string");
            this.addProperty?.("toggleRestriction", "default", "enum", { values: ["default", "max one", "always one"] });

            this._groupsData = [];

            // ---- DOM container ----
            const container = document.createElement("div");
            container.style.cssText = `
                display: flex; flex-direction: column; gap: 3px;
                padding: 4px 4px; box-sizing: border-box; width: 100%;
            `;

            const emptyMsg = document.createElement("div");
            emptyMsg.textContent = "No groups found";
            emptyMsg.style.cssText = `color:#666;font-size:12px;text-align:center;padding:8px;`;

            // ---- Rebuild logic ----
            this._rebuildGroups = () => {
                const groups    = getAllGroups(this);
                const matchStr  = (this.properties?.matchTitle ?? "").trim();
                let   titleRe   = null;
                if (matchStr) {
                    try { titleRe = new RegExp(matchStr, "i"); } catch (_) {}
                }

                const matchColorsStr = (this.properties?.matchColors ?? "").trim();
                const matchColorsArr = matchColorsStr
                    ? matchColorsStr.split(",").map(c => c.trim().toLowerCase()).filter(c => c)
                    : null;

                for (const g of groups) recomputeGroupChildren(g);

                const filtered = groups
                    .filter((g) => !titleRe || titleRe.exec(g.title))
                    .filter((g) => {
                        if (!matchColorsArr || matchColorsArr.length === 0) return true;
                        const gColor = (g.color ?? "").toLowerCase().trim();
                        if (!gColor) return false;

                        return matchColorsArr.some(c => {
                            // Dynamically map against LiteGraph's core LGraphCanvas.node_colors
                            if (window.LGraphCanvas && LGraphCanvas.node_colors && LGraphCanvas.node_colors[c]) {
                                const coreColorDef = LGraphCanvas.node_colors[c];
                                const coreColors = Object.values(coreColorDef).map(val => (val ?? "").toLowerCase());
                                if (coreColors.some(mc => mc && (gColor.includes(mc) || mc.includes(gColor)))) {
                                    return true;
                                }
                            }
                            return gColor.includes(c) || c.includes(gColor);
                        });
                    })
                    .sort((a, b) => {
                        const ay = Math.floor((a._pos?.[1] ?? a._bounding?.[1] ?? 0) / 30);
                        const by_ = Math.floor((b._pos?.[1] ?? b._bounding?.[1] ?? 0) / 30);
                        if (ay !== by_) return ay - by_;
                        return Math.floor((a._pos?.[0] ?? a._bounding?.[0] ?? 0) / 30)
                             - Math.floor((b._pos?.[0] ?? b._bounding?.[0] ?? 0) / 30);
                    });

                // Sync toggled state from actual node modes
                this._groupsData = filtered.map((g) => ({
                    group:   g,
                    toggled: getGroupNodes(g).some((n) => n.mode === MODE_ALWAYS),
                }));

                // Rebuild DOM with Reconciliation (reuse existing elements to prevent hover flickering)
                const currentChildren = Array.from(container.children).filter(el => el !== emptyMsg);

                // Check if number of rows and sequence of groups are identical
                let matches = currentChildren.length === filtered.length;
                if (matches) {
                    for (let i = 0; i < filtered.length; i++) {
                        if (currentChildren[i].group !== filtered[i]) {
                            matches = false;
                            break;
                        }
                    }
                }

                if (matches) {
                    // Update toggled states in-place so hover states aren't disrupted
                    for (let i = 0; i < this._groupsData.length; i++) {
                        const row = this._groupsData[i];
                        const child = currentChildren[i];
                        if (typeof child.updateVisuals === "function") {
                            child.updateVisuals(row.toggled);
                        }
                    }
                } else {
                    // Complete rebuild if the count or order of groups has changed
                    container.innerHTML = "";
                    if (filtered.length === 0) {
                        container.appendChild(emptyMsg);
                    } else {
                        for (const row of this._groupsData) {
                            container.appendChild(createRow(row, true, this));
                        }
                    }

                    // Resize node ONLY when widgets/groups actually change structurally
                    if (this.computeSize && this.setSize) {
                        const s = this.computeSize();
                        this.setSize(s);
                    }
                }
                app.canvas?.setDirty?.(true, true);
            };

            // ---- DOM widget ----
            const domWidget = this.addDOMWidget("xen_fgm_rows", "FGM_ROWS", container, {
                getValue:     () => null,
                setValue:     () => {},
                getMinHeight: () => Math.max(40, (this._groupsData?.length ?? 0) * ROW_HEIGHT + 8),
            });
            domWidget.computeLayoutSize = () => ({
                minHeight: Math.max(40, (this._groupsData?.length ?? 0) * ROW_HEIGHT + 8),
                minWidth: 0,
            });

            // Initial build + periodic refresh
            setTimeout(() => this._rebuildGroups?.(), 80);
            this._rebuildInterval = setInterval(() => this._rebuildGroups?.(), 600);
        };

        // ---- onRemoved: cleanup ----
        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            clearInterval(this._rebuildInterval);
            this._rebuildInterval = null;
            origOnRemoved?.apply(this, arguments);
        };

        // ---- computeSize ----
        const origComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (out) {
            const size = origComputeSize?.apply(this, arguments) ?? [260, 80];
            size[0] = Math.max(size[0], 260);
            size[1] = Math.max(size[1], (this._groupsData?.length ?? 0) * ROW_HEIGHT + 60);
            return size;
        };

        // ---- Right-click menu ----
        // ---- Right-click menu ----
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            origGetExtraMenuOptions?.apply(this, arguments);
            const mode = this.properties?.actionMode ?? "mute";
            const showAll = this.properties?.showAllGraphs === true;
            const restriction = this.properties?.toggleRestriction ?? "default";
            const matchTitle = this.properties?.matchTitle ?? "";
            const matchColors = this.properties?.matchColors ?? "";

            options.push(null);

            // 1. Mode switch
            options.push({
                content: `Mode: ${mode === "mute" ? "🔇 Mute" : "⏭ Bypass"} (click to switch)`,
                callback: () => {
                    this.properties = this.properties ?? {};
                    this.properties.actionMode = mode === "mute" ? "bypass" : "mute";
                    this._rebuildGroups?.();
                },
            });

            // 2. Show subgraphs switch
            options.push({
                content: `Show Subgraphs: ${showAll ? "✅ ON" : "❌ OFF"} (click to switch)`,
                callback: () => {
                    this.properties = this.properties ?? {};
                    this.properties.showAllGraphs = !showAll;
                    this._rebuildGroups?.();
                },
            });

            // 3. Restriction switch
            const nextRestrictionMap = {
                "default": "max one",
                "max one": "always one",
                "always one": "default"
            };
            options.push({
                content: `Restriction: ⚙️ ${restriction} (click to switch)`,
                callback: () => {
                    this.properties = this.properties ?? {};
                    this.properties.toggleRestriction = nextRestrictionMap[restriction] ?? "default";
                    this._rebuildGroups?.();
                },
            });

            // 4. Match title (prompt input)
            options.push({
                content: `Filter Title (RegEx): "${matchTitle || "(none)"}" (click to edit)`,
                callback: () => {
                    const val = prompt("Filter by group title (RegEx):", matchTitle);
                    if (val !== null) {
                        this.properties = this.properties ?? {};
                        this.properties.matchTitle = val;
                        this._rebuildGroups?.();
                    }
                },
            });

            // 5. Match colors (prompt input)
            options.push({
                content: `Filter Colors: "${matchColors || "(none)"}" (click to edit)`,
                callback: () => {
                    const val = prompt("Filter by group colors (comma-separated, e.g. 'red, blue, #3a2222'):", matchColors);
                    if (val !== null) {
                        this.properties = this.properties ?? {};
                        this.properties.matchColors = val;
                        this._rebuildGroups?.();
                    }
                },
            });

            options.push(null);

            options.push({
                content: "Mute / Bypass all groups",
                callback: () => {
                    const modeOff = (mode === "bypass") ? MODE_BYPASS : MODE_NEVER;
                    for (const row of (this._groupsData ?? [])) {
                        applyModeToNodes(getGroupNodes(row.group), modeOff);
                    }
                    this._rebuildGroups?.();
                },
            });
            options.push({
                content: "Enable all groups",
                callback: () => {
                    for (const row of (this._groupsData ?? [])) {
                        applyModeToNodes(getGroupNodes(row.group), MODE_ALWAYS);
                    }
                    this._rebuildGroups?.();
                },
            });
        };
    },

    loadedGraphNode(node) {
        if (node.type !== NODE_NAME) return;
        setTimeout(() => {
            node._rebuildGroups?.();
            node.setSize?.(node.computeSize?.());
        }, 200);
    },
});
