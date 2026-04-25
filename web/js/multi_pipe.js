import { app } from "../../../scripts/app.js";

const PIPE_IN_NODE = "XENodes.MultiPipeIn";
const PIPE_OUT_NODE = "XENodes.MultiPipeOut";
const MANAGED_INPUT_PREFIX = "slots.slot";
const CACHE_KEY = "_pipeSlotCache";

// ─── Utility Functions ───────────────────────────────────────────────

const debugLog = (node, ...args) => {
    if (node?.properties?.debug) {
        console.log(`[Multi-Pipe:${node?.type}:${node?.id}]`, ...args);
    }
};


const getSlotIndex = (name) => {
    if (!name) return -1;
    const match = name.match(/slot(\d+)/);
    return match ? parseInt(match[1], 10) : -1;
};

const formatSlotName = (idx) => `${MANAGED_INPUT_PREFIX}${String(idx).padStart(2, "0")}`;
const formatOutputLabel = (idx) => `slot${String(idx).padStart(2, "0")}`;

const getManagedInputs = (node) =>
    (node.inputs || []).filter((inp) => inp.name?.startsWith(MANAGED_INPUT_PREFIX));

// Triggers a full UI update in Vue (Nodes 2.0 / V3 API) bypassing optimizations
const triggerVueUpdate = (node, slotType) => {
    requestAnimationFrame(() => {
        // Recalculate size if outputs dynamically changed
        if (node.computeSize && node.setSize) {
            const s = node.computeSize();
            node.setSize([Math.max(node.size?.[0] ?? 180, s[0]), s[1]]);
        }

        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
        app.canvas?.setDirty(true, true);

        const graph = node.graph || app.graph;
        if (graph?.trigger) {
            // Append space to oldValue to force Vue reactivity
            graph.trigger('node:property:changed', { nodeId: node.id, property: 'title', oldValue: node.title + ' ', newValue: node.title });
            graph.trigger('node:slot-label:changed', { nodeId: node.id, slotType }); 
        }
        if (graph?.change) graph.change();
    });
};

// ─── Penetration Handlers (Strategy Pattern) ─────────────────────────

const PENETRATION_HANDLERS = {
    "Reroute": (node) => node.inputs?.[0]?.link,
    "Builtin: Reroute": (node) => node.inputs?.[0]?.link,
    "XENodes.MultiSwitch": (node) => {
        let selectVal = node.widgets?.find(w => w.name === "select")?.value ?? 0;
        if (selectVal >= 0) {
            const inputSuffix = `input${String(selectVal).padStart(2, '0')}`;
            return node.inputs?.find(inp => inp.name === inputSuffix || inp.name?.endsWith(`.${inputSuffix}`))?.link;
        }
        return null;
    },
    "ComfySwitchNode": (node) => {
        let switchVal = node.widgets?.find(w => w.name === "switch" || w.name === "boolean")?.value ?? false;
        return node.inputs?.find(inp => inp.name === (switchVal ? "on_true" : "on_false"))?.link;
    }
};

// ─── Core Tracing Logic ──────────────────────────────────────────────

// Deep link tracing function (Penetrates Subgraph, Reroute, and Switches)
const traceTrueOrigin = (contextNode, graph, linkId) => {
    if (linkId == null) {
        return null;
    }
    let currentGraph = graph;
    let currentLinkId = linkId;
    const seen = new Set(); 
    

    while (currentLinkId != null) {
        if (seen.has(currentLinkId)) {
            break;
        }
        seen.add(currentLinkId);

        const findLinkGlobal = (g, lId) => {
            if (!g) return null;
            if (g.links?.[lId]) return g.links[lId];
            for (const n of (g._nodes || [])) {
                const inner = n.inner_graph || n.subgraph || n.getInnerGraph?.();
                if (inner) {
                    const found = findLinkGlobal(inner, lId);
                    if (found) return found;
                }
            }
            return null;
        };

        const link = currentGraph.links?.[currentLinkId] || app.graph.links?.[currentLinkId] || findLinkGlobal(app.graph, currentLinkId);
        if (!link) {
            break;
        }

        let originNode = currentGraph.getNodeById(link.origin_id);
        
        // Subgraph Penetration (Inward)
        if (!originNode && link.origin_id > 0) {
            const findNodeGlobal = (g, nId) => {
                if (!g) return null;
                const n = g.getNodeById(nId);
                if (n) return n;
                for (const nn of (g._nodes || [])) {
                    const inner = nn.inner_graph || nn.subgraph || nn.getInnerGraph?.();
                    if (inner) {
                        const found = findNodeGlobal(inner, nId);
                        if (found) return found;
                    }
                }
                return null;
            };
            originNode = app.graph.getNodeById(link.origin_id) || findNodeGlobal(app.graph, link.origin_id);
        }
        
        // Fallback reverse lookup
        if (!originNode) {
            originNode = currentGraph._nodes.find(n => n.outputs?.some(out => out.links?.includes(currentLinkId)));
        }

        // Subgraph Penetration (Upward)
        if (!originNode) {
            if (link.origin_id != null && link.origin_id < 0) {
                const findParentNode = (rootG, innerG) => {
                    if (!rootG || !rootG._nodes) return null;
                    for (const n of rootG._nodes) {
                        const inner = n.inner_graph || n.subgraph || n.getInnerGraph?.();
                        if (inner === innerG) return n;
                        if (inner) {
                            const found = findParentNode(inner, innerG);
                            if (found) return found;
                        }
                    }
                    return null;
                };
                const parentNode = findParentNode(app.graph, currentGraph);
                if (parentNode) {
                    const parentInput = parentNode.inputs?.[link.origin_slot];
                    if (parentInput && parentInput.link != null) {
                        currentGraph = parentNode.graph || app.graph;
                        currentLinkId = parentInput.link;
                        continue;
                    }
                }
            }
            break; // Trace interrupted
        }
        
        let originSlot = link.origin_slot;

        // 1. Target MultiPipeIn found
        if (originNode.type === PIPE_IN_NODE) {
            return { node: originNode, slot: originSlot, graph: currentGraph };
        }

        // 2. Bypass Handlers (Reroute, Switch)
        if (PENETRATION_HANDLERS[originNode.type]) {
            const nextLinkId = PENETRATION_HANDLERS[originNode.type](originNode);
            if (nextLinkId != null) {
                currentLinkId = nextLinkId;
                continue;
            }
        }

        // 3. Subgraph output bypass
        let innerGraph = originNode.subgraph || originNode.inner_graph || originNode.getInnerGraph?.();
        if (innerGraph) {
            const outSlotDef = originNode.outputs[originSlot];
            if (outSlotDef) {
                const outNode = innerGraph._nodes.find(n => 
                    (n.type === "GraphOutput" || n.type === "SubgraphOutput" || n.type === "Builtin: GraphOutput" || n.type === "Primitive") && 
                    (n.properties?.name === outSlotDef.name || n.title === outSlotDef.name || n.name === outSlotDef.name)
                );
                
                if (outNode) {
                    currentGraph = innerGraph;
                    currentLinkId = outNode.inputs?.[0]?.link || outNode.inputs?.[originSlot]?.link;
                    if (currentLinkId != null) continue;
                } else {
                    // Nodes 2.0 Subgraph: No physical GraphOutput node. 
                    // Links are made directly to a virtual output node with a negative ID (e.g., -20).
                    // The target_slot corresponds to the index of the subgraph's output pin.
                    let virtualOutputLink = null;
                    if (innerGraph.links) {
                        const linkKeys = Object.keys(innerGraph.links);
                        for (let key of linkKeys) {
                            let l = innerGraph.links[key];
                            if (l && l.target_slot === originSlot && l.target_id < 0) {
                                virtualOutputLink = l;
                                break;
                            }
                        }
                    }

                    if (virtualOutputLink) {
                        currentGraph = innerGraph;
                        currentLinkId = virtualOutputLink.id;
                        continue;
                    }
                }
                
                // Deep recursive find
                const findPipeInRecursive = (g) => {
                    if (!g || !g._nodes) return null;
                    const direct = g._nodes.find(n => n.type === PIPE_IN_NODE);
                    if (direct) return { node: direct, graph: g };
                    for (const n of g._nodes) {
                        const inner = n.inner_graph || n.subgraph || n.getInnerGraph?.();
                        if (inner) {
                            const found = findPipeInRecursive(inner);
                            if (found) return found;
                        }
                    }
                    return null;
                };
                const innerPipeInResult = findPipeInRecursive(innerGraph);
                if (innerPipeInResult) {
                    return { node: innerPipeInResult.node, slot: 0, graph: innerPipeInResult.graph };
                }
            }
        }

        return { node: originNode, slot: originSlot, graph: currentGraph };
    }
    return null;
};

// ─── Unified State Builder ───────────────────────────────────────────

const getConnectedInfo = (node, inputSlot, includeTitle = false) => {
    const graph = node.graph || app.graph;
    if (!graph || inputSlot.link == null) return null;
    
    const origin = traceTrueOrigin(node, graph, inputSlot.link);
    
    if (!origin) return null;
    
    const originOutput = origin.node.outputs?.[origin.slot];
    let label = originOutput?.label || originOutput?.name || originOutput?.type || null;
    const type = originOutput?.type || "*";

    if (includeTitle && label && origin.node) {
        const title = origin.node.title || origin.node.name || origin.node.type;
        if (title) label = `${title}.${label}`;
    }
    return { type, label };
};

// Builds the complete calculated state of the pipe at the given node.
const buildPipeState = (contextNode, includeTitle) => {
    const state = new Map(); // Map<slotIndex, { type, label }>
    
    const traceAndMerge = (node) => {
        if (!node) return;
        
        
        // 1. Trace parent pipe to inherit state first
        const pipeInput = (node.inputs || []).find(inp => inp.name === "pipe");
        if (pipeInput && pipeInput.link != null) {
            const graph = node.graph || app.graph;
            const origin = traceTrueOrigin(node, graph, pipeInput.link);
            if (origin && origin.node.type === PIPE_IN_NODE) {
                traceAndMerge(origin.node);
            }
        }
        
        // 2. Overwrite with locally connected slots
        const managed = getManagedInputs(node);
        for (const inp of managed) {
            if (inp.link != null) {
                const info = getConnectedInfo(node, inp, includeTitle);
                if (info) state.set(getSlotIndex(inp.name), info);
            }
        }
    };

    if (contextNode.type === PIPE_IN_NODE) {
        traceAndMerge(contextNode);
    } else if (contextNode.type === PIPE_OUT_NODE) {
        const pipeInput = (contextNode.inputs || []).find(inp => inp.name === "pipe");
        if (pipeInput && pipeInput.link != null) {
            const graph = contextNode.graph || app.graph;
            const origin = traceTrueOrigin(contextNode, graph, pipeInput.link);
            if (origin && origin.node.type === PIPE_IN_NODE) traceAndMerge(origin.node);
        }
    }
    
    return state;
};


// ─── MultiPipeIn Extension ───────────────────────────────────────────

app.registerExtension({
    name: "XENodes.MultiPipeIn",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PIPE_IN_NODE) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            
            this.properties = this.properties || {};
            this.properties.debug = this.properties.debug ?? false;
            this.properties.show_origin_title = this.properties.show_origin_title ?? true;

            const updateLabels = () => {
                let changed = false;
                const pipeState = buildPipeState(this, this.properties.show_origin_title);
                const activeIndices = Array.from(pipeState.keys());
                let targetMaxIdx = activeIndices.length > 0 ? Math.max(...activeIndices) : -1;

                // Add missing inputs
                for (let i = 0; i <= targetMaxIdx + 1; i++) {
                    const name = formatSlotName(i);
                    if (!this.inputs.find(inp => inp.name === name)) {
                        this.addInput(name, "*");
                        changed = true;
                    }
                }
                
                // Trim trailing empty slots
                for (let i = (this.inputs?.length || 0) - 1; i >= 0; i--) {
                    const inp = this.inputs[i];
                    if (inp.name?.startsWith(MANAGED_INPUT_PREFIX)) {
                        const idx = getSlotIndex(inp.name);
                        if (idx > targetMaxIdx + 1 && inp.link == null) {
                            this.removeInput(i);
                            changed = true;
                        }
                    }
                }

                // Update labels (with Optimized Splice Hack for Vue)
                for (let i = 0; i < (this.inputs?.length || 0); i++) {
                    const inp = this.inputs[i];
                    if (inp.name?.startsWith(MANAGED_INPUT_PREFIX)) {
                        const idx = getSlotIndex(inp.name);
                        const stateInfo = pipeState.get(idx);
                        const newLabel = stateInfo?.label || formatOutputLabel(idx);
                        
                        if (inp.label !== newLabel) {
                            inp.label = newLabel;
                            changed = true;
                            
                            // 🚀 Optimized Splice Hack: Only re-mount the specific slot that changed!
                            const temp = this.inputs.splice(i, 1)[0];
                            this.inputs.splice(i, 0, temp);
                        }
                    }
                }
                
                if (changed) triggerVueUpdate(this, 1);
            };
            
            const originalGetExtraMenuOptions = this.getExtraMenuOptions;
            this.getExtraMenuOptions = function (_, options) {
                originalGetExtraMenuOptions?.apply(this, arguments);
                options.push({
                    content: this.properties.show_origin_title ? "Hide Origin Node Title" : "Show Origin Node Title",
                    callback: () => {
                        this.properties.show_origin_title = !this.properties.show_origin_title;
                        updateLabels();
                    }
                });
            };

            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                const r = originalOnConnectionsChange?.apply(this, arguments);
                // Wait 10ms to ensure LiteGraph has fully resolved the new link internally
                setTimeout(() => {
                    updateLabels();
                    triggerVueUpdate(this, 1);
                }, 10);
                return r;
            };

            let lastFP = "";
            const checkLabels = () => {
                if (this.flags?.collapsed) return;
                const pipeState = buildPipeState(this, this.properties.show_origin_title);
                const stateStr = JSON.stringify(Array.from(pipeState.entries()));
                const fp = `${this.properties.show_origin_title ? "1" : "0"}|${stateStr}`;

                if (fp !== lastFP) {
                    lastFP = fp;
                    updateLabels();
                }
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

// ─── MultiPipeOut Extension ──────────────────────────────────────────

app.registerExtension({
    name: "XENodes.MultiPipeOut",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PIPE_OUT_NODE) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);

            this._allOutputs = null;
            this.properties = this.properties || {};
            this.properties.debug = this.properties.debug ?? false;
            this.properties.show_origin_title = this.properties.show_origin_title ?? true;

            const saveCache = (slots) => { this.properties[CACHE_KEY] = slots; };
            const loadCache = () => this.properties?.[CACHE_KEY] || null;
            
            const originalGetExtraMenuOptions = this.getExtraMenuOptions;
            this.getExtraMenuOptions = function (_, options) {
                originalGetExtraMenuOptions?.apply(this, arguments);
                options.push({
                    content: this.properties.show_origin_title ? "Hide Origin Node Title" : "Show Origin Node Title",
                    callback: () => {
                        this.properties.show_origin_title = !this.properties.show_origin_title;
                        syncOutputs();
                    }
                });
            };

            const syncOutputs = () => {
                if (!this._allOutputs && this.outputs?.length > 0) {
                    this._allOutputs = this.outputs.map(o => Object.assign({}, o));
                }
                if (!this._allOutputs) return;

                const targetOutputs = [];
                const cacheSlots = [];
                const pipeState = buildPipeState(this, this.properties.show_origin_title);
                const hasConnection = pipeState.size > 0 || (this.inputs[0]?.link != null);

                if (hasConnection) {
                    const activeIndices = Array.from(pipeState.keys());
                    const maxIdx = activeIndices.length > 0 ? Math.max(...activeIndices) : -1;

                    for (let slotIdx = 0; slotIdx <= maxIdx; slotIdx++) {
                        const baseEntry = this._allOutputs[slotIdx];
                        if (!baseEntry) continue;

                        const info = pipeState.get(slotIdx);
                        const targetLabel = info?.label || formatOutputLabel(slotIdx);
                        const targetType = (info?.type && info.type !== "*") ? info.type : "*";
                        
                        cacheSlots.push(info ? { label: targetLabel, type: targetType } : null);
                        targetOutputs.push({ baseEntry, label: targetLabel, type: targetType });
                    }
                    saveCache(cacheSlots);
                } else {
                    const cache = loadCache();
                    if (cache && cache.length > 0) {
                        for (let slotIdx = 0; slotIdx < cache.length; slotIdx++) {
                            const baseEntry = this._allOutputs[slotIdx];
                            if (!baseEntry) continue;
                            const cached = cache[slotIdx];
                            const targetLabel = cached?.label || formatOutputLabel(slotIdx);
                            const targetType = cached?.type || "*";
                            targetOutputs.push({ baseEntry, label: targetLabel, type: targetType });
                        }
                    }
                }

                let changed = false;
                const graph = this.graph || app.graph;

                // Handle output shrinkage safely
                if (this.outputs?.length > targetOutputs.length) {
                    changed = true;
                    if (graph) {
                        for (let i = targetOutputs.length; i < this.outputs.length; i++) {
                            const oldOut = this.outputs[i];
                            if (oldOut && oldOut.links?.length > 0) {
                                for (const linkId of [...oldOut.links]) graph.removeLink(linkId);
                            }
                        }
                    }
                    this.outputs.splice(targetOutputs.length); // Bulk remove excess
                }

                // Handle output growth and updates (with Optimized Splice Hack)
                for (let i = 0; i < targetOutputs.length; i++) {
                    const target = targetOutputs[i];
                    let current = this.outputs[i];

                    if (!current) {
                        changed = true;
                        current = Object.assign({}, target.baseEntry, { label: target.label, type: target.type });
                        this.outputs.push(current);
                    } else if (current.label !== target.label || current.type !== target.type) {
                        changed = true;
                        current.label = target.label;
                        current.type = target.type;
                        if (target.baseEntry) current.name = target.baseEntry.name;
                        
                        // 🚀 Optimized Splice Hack: Only re-mount the specific slot that changed!
                        const temp = this.outputs.splice(i, 1)[0];
                        this.outputs.splice(i, 0, temp);
                    }
                }

                if (changed || !this._pipeSizeSet) {
                    this._pipeSizeSet = true;
                    triggerVueUpdate(this, 2);
                }
            };

            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                const r = originalOnConnectionsChange?.apply(this, arguments);
                // Wait 10ms to ensure LiteGraph has fully resolved the new link internally
                setTimeout(() => {
                    syncOutputs();
                    triggerVueUpdate(this, 2);
                }, 10);
                return r;
            };

            let lastFP = "";
            const periodicSync = () => {
                if (this.flags?.collapsed) return;
                
                const pipeInput = (this.inputs || []).find(inp => inp.name === "pipe");
                let fp = `${this.properties.show_origin_title ? "1" : "0"}|`;

                if (pipeInput && pipeInput.link != null) {
                    const pipeState = buildPipeState(this, this.properties.show_origin_title);
                    fp += JSON.stringify(Array.from(pipeState.entries()));
                } else {
                    fp += `cache:${JSON.stringify(loadCache())}`;
                }

                if (fp !== lastFP) {
                    lastFP = fp;
                    syncOutputs();
                }
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