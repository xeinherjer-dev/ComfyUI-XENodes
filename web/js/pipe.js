import { app } from "../../../scripts/app.js";

const PIPE_IN_NODE = "XENodes.PipeIn";
const PIPE_OUT_NODE = "XENodes.PipeOut";
const MANAGED_INPUT_PREFIX = "slots.slot";

const getSlotIndex = (name) => {
    if (!name) return -1;
    const match = name.match(/slot(\d+)/);
    return match ? parseInt(match[1], 10) : -1;
};

const getManagedInputs = (node) =>
    (node.inputs || []).filter((inp) => inp.name?.startsWith(MANAGED_INPUT_PREFIX));

const findConnectedPipeIn = (pipeOutNode) => {
    const graph = pipeOutNode.graph || app.graph;
    if (!graph) return null;

    const pipeInput = (pipeOutNode.inputs || []).find((inp) => inp.name === "pipe");
    if (!pipeInput || pipeInput.link == null) return null;

    const link = graph.links?.[pipeInput.link];
    if (!link) return null;

    const originNode = graph.getNodeById(link.origin_id);
    if (originNode?.type === PIPE_IN_NODE) return originNode;
    return null;
};

const getConnectedType = (node, inputSlot) => {
    const graph = node.graph || app.graph;
    if (!graph || inputSlot.link == null) return null;
    const link = graph.links?.[inputSlot.link];
    if (!link) return null;
    const originNode = graph.getNodeById(link.origin_id);
    if (!originNode) return null;
    const originOutput = originNode.outputs?.[link.origin_slot];
    return originOutput?.type || null;
};

const getConnectedLabel = (node, inputSlot) => {
    const graph = node.graph || app.graph;
    if (!graph || inputSlot.link == null) return null;
    const link = graph.links?.[inputSlot.link];
    if (!link) return null;
    const originNode = graph.getNodeById(link.origin_id);
    if (!originNode) return null;
    const originOutput = originNode.outputs?.[link.origin_slot];
    return originOutput?.label || originOutput?.name || originOutput?.type || null;
};


// ─── PipeIn Extension ────────────────────────────────────────────────
app.registerExtension({
    name: "XENodes.PipeIn",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PIPE_IN_NODE) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated
                ? originalOnNodeCreated.apply(this, arguments)
                : undefined;

            const updateLabels = () => {
                for (const inp of getManagedInputs(this)) {
                    const label = getConnectedLabel(this, inp);
                    const idx = getSlotIndex(inp.name);
                    inp.label = label || (idx >= 0 ? `slot${String(idx).padStart(2, "0")}` : inp.name);
                }
            };

            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                const r = originalOnConnectionsChange?.apply(this, arguments);
                updateLabels();
                app.canvas?.setDirty(true, true);
                return r;
            };

            let lastFP = "";
            const checkLabels = () => {
                if (this.flags?.collapsed) return;
                let fp = "";
                for (const inp of getManagedInputs(this)) {
                    fp += `${inp.name}:${getConnectedLabel(this, inp) || ""}|`;
                }
                if (fp !== lastFP) { lastFP = fp; updateLabels(); app.canvas?.setDirty(true, true); }
            };
            this._pipeInInterval = setInterval(checkLabels, 1000);

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                clearInterval(this._pipeInInterval);
                return originalOnRemoved?.apply(this, arguments);
            };

            updateLabels();
            return result;
        };
    },
});


// ─── PipeOut Extension ───────────────────────────────────────────────
app.registerExtension({
    name: "XENodes.PipeOut",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PIPE_OUT_NODE) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated
                ? originalOnNodeCreated.apply(this, arguments)
                : undefined;

            this._allOutputs = null;

            // ── Slot cache (persisted in node.properties) ──────────────────
            // When PipeIn is visible (not in a Subgraph), we update both the
            // live outputs AND save the slot configuration to properties.
            // When PipeIn is NOT directly reachable (e.g., wrapped in a Subgraph),
            // we restore from the cache so outputs remain visible.
            //
            // Cache format: Array of { label, type } ordered by slotIdx.
            //   properties._pipeSlotCache = [
            //     { label: "IMAGE", type: "IMAGE" },   // slot00
            //     { label: "STRING", type: "STRING" }, // slot01
            //     null,                                // slot02 (empty but in range)
            //   ]
            const CACHE_KEY = "_pipeSlotCache";

            const saveCache = (slots) => {
                this.properties = this.properties || {};
                this.properties[CACHE_KEY] = slots;
            };

            const loadCache = () => {
                return this.properties?.[CACHE_KEY] || null;
            };

            const syncOutputs = () => {
                if (!this._allOutputs && this.outputs?.length > 0) {
                    this._allOutputs = [...this.outputs];
                }
                if (!this._allOutputs) return;

                const pipeInNode = findConnectedPipeIn(this);
                const all = this._allOutputs;
                const newOutputs = [];

                if (pipeInNode) {
                    // ── PipeIn is directly visible ──────────────────────────
                    const pipeInInputs = getManagedInputs(pipeInNode);

                    let maxIdx = -1;
                    for (const inp of pipeInInputs) {
                        if (inp.link != null) maxIdx = Math.max(maxIdx, getSlotIndex(inp.name));
                    }

                    // Build cache for this sync
                    const cacheSlots = [];

                    for (let slotIdx = 0; slotIdx <= maxIdx; slotIdx++) {
                        const outEntry = all[slotIdx];
                        if (!outEntry) continue;

                        const matchingInput = pipeInInputs.find(
                            (inp) => getSlotIndex(inp.name) === slotIdx
                        );

                        if (matchingInput && matchingInput.link != null) {
                            const type = getConnectedType(pipeInNode, matchingInput);
                            const label = getConnectedLabel(pipeInNode, matchingInput);
                            outEntry.label = label || `slot${String(slotIdx).padStart(2, "0")}`;
                            outEntry.type = (type && type !== "*") ? type : "*";
                            cacheSlots.push({ label: outEntry.label, type: outEntry.type });
                        } else {
                            outEntry.label = `slot${String(slotIdx).padStart(2, "0")}`;
                            outEntry.type = "*";
                            cacheSlots.push(null); // empty intermediate slot
                        }
                        newOutputs.push(outEntry);
                    }

                    // Persist the slot configuration
                    saveCache(cacheSlots);

                } else {
                    // ── PipeIn is NOT directly visible (e.g., in a Subgraph) ──
                    // Restore from the last known cache so outputs remain usable.
                    const cache = loadCache();
                    if (cache && cache.length > 0) {
                        for (let slotIdx = 0; slotIdx < cache.length; slotIdx++) {
                            const outEntry = all[slotIdx];
                            if (!outEntry) continue;
                            const cached = cache[slotIdx];
                            if (cached) {
                                outEntry.label = cached.label;
                                outEntry.type = cached.type;
                            } else {
                                outEntry.label = `slot${String(slotIdx).padStart(2, "0")}`;
                                outEntry.type = "*";
                            }
                            newOutputs.push(outEntry);
                        }
                    }
                    // If no cache exists yet, outputs remain empty (nothing to restore)
                }

                // Remove links on slots that are being hidden
                const graph = this.graph || app.graph;
                if (graph) {
                    const visible = new Set(newOutputs);
                    for (const o of all) {
                        if (visible.has(o)) continue;
                        if (o.links?.length > 0) {
                            for (const linkId of [...o.links]) graph.removeLink(linkId);
                        }
                    }
                }

                const prevLen = this.outputs?.length;
                this.outputs = newOutputs;

                if (prevLen !== newOutputs.length || !this._pipeSizeSet) {
                    this._pipeSizeSet = true;
                    if (this.computeSize && this.setSize) {
                        const s = this.computeSize();
                        this.setSize([Math.max(this.size?.[0] ?? 180, s[0]), s[1]]);
                    }
                }
                app.canvas?.setDirty(true, true);
            };

            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                const r = originalOnConnectionsChange?.apply(this, arguments);
                syncOutputs();
                return r;
            };

            let lastFP = "";
            const periodicSync = () => {
                if (this.flags?.collapsed) return;
                const pipeInNode = findConnectedPipeIn(this);
                let fp;
                if (pipeInNode) {
                    fp = "";
                    for (const inp of getManagedInputs(pipeInNode)) {
                        if (inp.link == null) continue;
                        fp += `${inp.name}:${getConnectedType(pipeInNode, inp)}:${getConnectedLabel(pipeInNode, inp)}|`;
                    }
                } else {
                    // Use cache fingerprint so we don't re-render unnecessarily
                    fp = `cache:${JSON.stringify(loadCache())}`;
                }
                if (fp !== lastFP) { lastFP = fp; syncOutputs(); }
            };
            this._pipeOutInterval = setInterval(periodicSync, 500);

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                clearInterval(this._pipeOutInterval);
                return originalOnRemoved?.apply(this, arguments);
            };

            syncOutputs();
            return result;
        };
    },
});
