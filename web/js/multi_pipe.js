import { app } from "../../../scripts/app.js";

const PIPE_IN_NODE = "XENodes.MultiPipeIn";
const PIPE_OUT_NODE = "XENodes.MultiPipeOut";
const MANAGED_INPUT_PREFIX = "slots.slot";

function debugLog(node, ...args) {
    if (node?.properties?.debug) {
        console.log(`[Multi-Pipe:${node.type}:${node.id}]`, ...args);
    }
}

const getSlotIndex = (name) => {
    if (!name) return -1;
    const match = name.match(/slot(\d+)/);
    return match ? parseInt(match[1], 10) : -1;
};

const getManagedInputs = (node) =>
    (node.inputs || []).filter((inp) => inp.name?.startsWith(MANAGED_INPUT_PREFIX));

// Deep link tracing function (Penetrates Subgraph and Reroute)
const traceTrueOrigin = (contextNode, graph, linkId) => {
    if (linkId == null) return null;
    let currentGraph = graph;
    let currentLinkId = linkId;

    const seen = new Set(); // Prevent infinite loops
    let step = 0;

    while (currentLinkId != null) {
        step++;
        if (seen.has(currentLinkId)) {
            debugLog(contextNode, `Trace interrupted: Loop detected. linkId=${currentLinkId}`);
            break;
        }
        seen.add(currentLinkId);

        const findLinkGlobal = (graphToSearch, linkIdToFind) => {
            if (!graphToSearch) return null;
            if (graphToSearch.links?.[linkIdToFind]) return graphToSearch.links[linkIdToFind];
            for (const n of (graphToSearch._nodes || [])) {
                const inner = n.inner_graph || n.subgraph || (typeof n.getInnerGraph === 'function' ? n.getInnerGraph() : null);
                if (inner) {
                    const found = findLinkGlobal(inner, linkIdToFind);
                    if (found) return found;
                }
            }
            return null;
        };

        const link = currentGraph.links?.[currentLinkId] || app.graph.links?.[currentLinkId] || findLinkGlobal(app.graph, currentLinkId);
        if (!link) {
            debugLog(contextNode, `Trace interrupted: Link not found globally. linkId=${currentLinkId}`);
            break;
        }

        let originNode = currentGraph.getNodeById(link.origin_id);
        
        // Penetrate Group Node Inner Graph boundaries when Origin Node isn't found
        if (!originNode && link.origin_id > 0) {
            const findNodeGlobal = (graphToSearch, idToFind) => {
                if (!graphToSearch) return null;
                const n = graphToSearch.getNodeById(idToFind);
                if (n) return n;
                for (const nn of (graphToSearch._nodes || [])) {
                    const inner = nn.inner_graph || nn.subgraph || (typeof nn.getInnerGraph === 'function' ? nn.getInnerGraph() : null);
                    if (inner) {
                        const found = findNodeGlobal(inner, idToFind);
                        if (found) return found;
                    }
                }
                return null;
            };
            originNode = app.graph.getNodeById(link.origin_id) || findNodeGlobal(app.graph, link.origin_id);
        }
        
        // Fallback: If originNode has a negative ID or cannot be found, 
        // try to find it by searching all nodes in the graph for this link
        if (!originNode) {
            originNode = currentGraph._nodes.find(n => 
                n.outputs?.some(out => out.links?.includes(currentLinkId))
            );
            if (originNode) {
                debugLog(contextNode, `Found originNode via reverse lookup. id=${originNode.id}`);
            }
        }

        if (!originNode) {
            // Also fall back to graph boundaries 
            const isInner = currentGraph !== app.graph;
            
            // Subgraph -> Parent Graph Penetration (Upward)
            if (link.origin_id != null && link.origin_id < 0) {
                const findParentNode = (rootGraph, innerGraphToFind) => {
                    if (!rootGraph || !rootGraph._nodes) return null;
                    for (const n of rootGraph._nodes) {
                        const inner = n.inner_graph || n.subgraph || (typeof n.getInnerGraph === 'function' ? n.getInnerGraph() : null);
                        if (inner === innerGraphToFind) return n;
                        if (inner) {
                            const found = findParentNode(inner, innerGraphToFind);
                            if (found) return found;
                        }
                    }
                    return null;
                };

                const parentNode = findParentNode(app.graph, currentGraph);
                if (parentNode) {
                    const parentInput = parentNode.inputs?.[link.origin_slot];
                    if (parentInput && parentInput.link != null) {
                        debugLog(contextNode, `Penetrating OUT (Subgraph->Parent): parentInput=${parentInput.name}, nextLinkId=${parentInput.link}`);
                        currentGraph = parentNode.graph || app.graph;
                        currentLinkId = parentInput.link;
                        continue;
                    }
                }
            }

            debugLog(contextNode, `Trace interrupted: originNode not found even after reverse lookup.`);
            break;
        }
        
        let originSlot = link.origin_slot;
        debugLog(contextNode, `Step ${step}: originNode.type=${originNode.type}, originNode.id=${originNode.id}`);

        // 1. End condition: Multi Pipe In node found
        if (originNode.type === PIPE_IN_NODE) {
            return { node: originNode, slot: originSlot, graph: currentGraph };
        }

        // 2. Penetrate Reroute node
        if (originNode.type === "Reroute" || originNode.type === "Builtin: Reroute") {
            currentLinkId = originNode.inputs?.[0]?.link;
            debugLog(contextNode, `Penetrating Reroute: next linkId=${currentLinkId}`);
            continue;
        }

        // 2.5 Penetrate Switch Nodes
        if (originNode.type === "XENodes.MultiSwitch") {
            let selectVal = 0;
            const selectWidget = originNode.widgets?.find(w => w.name === "select");
            if (selectWidget) selectVal = selectWidget.value;
            
            if (selectVal >= 0) {
                const inputSuffix = `input${String(selectVal).padStart(2, '0')}`;
                const activeInput = originNode.inputs?.find(inp => inp.name === inputSuffix || inp.name?.endsWith(`.${inputSuffix}`));
                if (activeInput && activeInput.link != null) {
                    currentLinkId = activeInput.link;
                    debugLog(contextNode, `Penetrating MultiSwitch: selected ${inputSuffix}, next linkId=${currentLinkId}`);
                    continue;
                }
            }
        }

        if (originNode.type === "ComfySwitchNode") {
            let switchVal = false;
            const switchWidget = originNode.widgets?.find(w => w.name === "switch" || w.name === "boolean");
            if (switchWidget) switchVal = switchWidget.value;
            const inputName = switchVal ? "on_true" : "on_false";
            const activeInput = originNode.inputs?.find(inp => inp.name === inputName);
            if (activeInput && activeInput.link != null) {
                currentLinkId = activeInput.link;
                debugLog(contextNode, `Penetrating ComfySwitchNode: selected ${inputName}, next linkId=${currentLinkId}`);
                continue;
            }
        }

        // 3. Penetrate Subgraph / Group Node
        let innerGraph = originNode.subgraph || originNode.inner_graph;
        if (!innerGraph && typeof originNode.getInnerGraph === "function") {
            innerGraph = originNode.getInnerGraph();
        }

        if (innerGraph) {
            debugLog(contextNode, `Penetrating: innerGraph found. type=${originNode.type}`);
            const outSlotDef = originNode.outputs[originSlot];
            if (outSlotDef) {
                // Find the corresponding output node inside the Subgraph
                const outNode = innerGraph._nodes.find(n => 
                    (n.type === "GraphOutput" || n.type === "SubgraphOutput" || n.type === "Builtin: GraphOutput" || n.type === "Primitive") && 
                    (n.properties?.name === outSlotDef.name || n.title === outSlotDef.name || n.name === outSlotDef.name)
                );
                
                if (outNode) {
                    debugLog(contextNode, `Penetrating: inner node found. type=${outNode.type}, name=${outNode.title || outNode.name}`);
                    currentGraph = innerGraph;
                    currentLinkId = outNode.inputs?.[0]?.link || outNode.inputs?.[originSlot]?.link;
                    if (currentLinkId != null) {
                        continue;
                    }
                }
                
                // Fallback: Just find the MultiPipeIn inside the innerGraph (recursive for nested subgraphs)
                const findPipeInRecursive = (g) => {
                    if (!g || !g._nodes) return null;
                    const direct = g._nodes.find(n => n.type === PIPE_IN_NODE);
                    if (direct) return { node: direct, graph: g };
                    for (const n of g._nodes) {
                        const inner = n.inner_graph || n.subgraph || (typeof n.getInnerGraph === 'function' ? n.getInnerGraph() : null);
                        if (inner) {
                            const found = findPipeInRecursive(inner);
                            if (found) return found;
                        }
                    }
                    return null;
                };

                const innerPipeInResult = findPipeInRecursive(innerGraph);
                if (innerPipeInResult) {
                    debugLog(contextNode, `Penetrating: Fallback direct find inner PipeIn. id=${innerPipeInResult.node.id}`);
                    return { node: innerPipeInResult.node, slot: 0, graph: innerPipeInResult.graph };
                }

                debugLog(contextNode, `Penetration failed: inner node not found. outSlotDef.name=${outSlotDef.name}`);
            } else {
                debugLog(contextNode, `originNode.outputs[${originSlot}] is undefined`);
            }
        }

        // Return the final traceable node
        return { node: originNode, slot: originSlot, graph: currentGraph };
    }

    debugLog(contextNode, `Trace finished: currentLinkId=${currentLinkId}`);
    return null;
};

const findConnectedPipeIn = (pipeOutNode) => {
    const graph = pipeOutNode.graph || app.graph;
    if (!graph) return null;

    const pipeInput = (pipeOutNode.inputs || []).find((inp) => inp.name === "pipe");
    if (!pipeInput || pipeInput.link == null) {
        return null;
    }

    debugLog(pipeOutNode, `Find: Trace started...`);
    const origin = traceTrueOrigin(pipeOutNode, graph, pipeInput.link);
    if (origin) {
        debugLog(pipeOutNode, `Find: Trace result node type ->`, origin.node.type);
        if (origin.node.type === PIPE_IN_NODE) {
            return origin.node;
        }
    }
    return null;
};

const getConnectedType = (node, inputSlot) => {
    const graph = node.graph || app.graph;
    if (!graph || inputSlot.link == null) return null;
    const origin = traceTrueOrigin(node, graph, inputSlot.link);
    if (!origin) return null;
    const originOutput = origin.node.outputs?.[origin.slot];
    return originOutput?.type || null;
};

const getConnectedLabel = (node, inputSlot, includeTitle = false) => {
    const graph = node.graph || app.graph;
    if (!graph || inputSlot.link == null) return null;
    const origin = traceTrueOrigin(node, graph, inputSlot.link);
    if (!origin) return null;
    const originOutput = origin.node.outputs?.[origin.slot];
    let label = originOutput?.label || originOutput?.name || originOutput?.type || null;
    
    if (includeTitle && label && origin.node) {
        const title = origin.node.title || origin.node.name || origin.node.type;
        if (title) {
            label = `${title}.${label}`;
        }
    }
    
    return label;
};


// ─── MultiPipeIn Extension ───────────────────────────────────────────
app.registerExtension({
    name: "XENodes.MultiPipeIn",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PIPE_IN_NODE) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated
                ? originalOnNodeCreated.apply(this, arguments)
                : undefined;
            this.properties = this.properties || {};
            this.properties.debug = this.properties.debug ?? false;
            this.properties.show_origin_title = this.properties.show_origin_title ?? false;

            const updateLabels = () => {
                let changed = false;
                
                // Clone inputs array to force Vue to detect fully new references
                const newInputs = [];
                for (let i = 0; i < (this.inputs?.length || 0); i++) {
                    const inp = this.inputs[i];
                    if (inp.name?.startsWith(MANAGED_INPUT_PREFIX)) {
                        const label = getConnectedLabel(this, inp, this.properties.show_origin_title);
                        const idx = getSlotIndex(inp.name);
                        const newLabel = label || (idx >= 0 ? `slot${String(idx).padStart(2, "0")}` : inp.name);
                        if (inp.label !== newLabel) {
                            newInputs.push(Object.assign({}, inp, { label: newLabel }));
                            changed = true;
                            continue;
                        }
                    }
                    newInputs.push(inp);
                }

                if (changed) {
                    this.inputs = newInputs; // triggers the Vue setter
                    
                    if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
                    app.canvas?.setDirty(true, true);
                    
                    const graph = this.graph || app.graph;
                    if (graph?.trigger) {
                        // Force full Vue node UI re-evaluation by faking a property change
                        graph.trigger('node:property:changed', { nodeId: this.id, property: 'title', oldValue: this.title, newValue: this.title });
                        graph.trigger('node:slot-label:changed', { nodeId: this.id, slotType: 1 }); // INPUT
                    }
                    if (graph?.change) graph.change();
                }
            };
            
            const originalGetExtraMenuOptions = this.getExtraMenuOptions;
            this.getExtraMenuOptions = function (_, options) {
                if (originalGetExtraMenuOptions) {
                    originalGetExtraMenuOptions.apply(this, arguments);
                }
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
                // Delay to allow LiteGraph to finish registering the link before tracing
                setTimeout(() => updateLabels(), 0);
                app.canvas?.setDirty(true, true);
                return r;
            };

            let lastFP = "";
            const checkLabels = () => {
                if (this.flags?.collapsed) return;
                let fp = `${this.properties.show_origin_title ? "1" : "0"}|`;
                for (const inp of getManagedInputs(this)) {
                    fp += `${inp.name}:${getConnectedLabel(this, inp, this.properties.show_origin_title) || ""}|`;
                }
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
            const result = originalOnNodeCreated
                ? originalOnNodeCreated.apply(this, arguments)
                : undefined;

            this._allOutputs = null;
            const CACHE_KEY = "_pipeSlotCache";

            this.properties = this.properties || {};
            this.properties.debug = this.properties.debug ?? false;
            this.properties.show_origin_title = this.properties.show_origin_title ?? false;

            const saveCache = (slots) => {
                this.properties[CACHE_KEY] = slots;
            };

            const loadCache = () => {
                return this.properties?.[CACHE_KEY] || null;
            };
            
            const originalGetExtraMenuOptions = this.getExtraMenuOptions;
            this.getExtraMenuOptions = function (_, options) {
                if (originalGetExtraMenuOptions) {
                    originalGetExtraMenuOptions.apply(this, arguments);
                }
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
                    this._allOutputs = [...this.outputs];
                }
                if (!this._allOutputs) return;

                const pipeInNode = findConnectedPipeIn(this);
                const all = this._allOutputs;
                const newOutputs = [];

                if (pipeInNode) {
                    const pipeInInputs = getManagedInputs(pipeInNode);
                    let maxIdx = -1;
                    for (const inp of pipeInInputs) {
                        if (inp.link != null) maxIdx = Math.max(maxIdx, getSlotIndex(inp.name));
                    }

                    const cacheSlots = [];
                    for (let slotIdx = 0; slotIdx <= maxIdx; slotIdx++) {
                        const outEntry = all[slotIdx];
                        if (!outEntry) continue;

                        const matchingInput = pipeInInputs.find(
                            (inp) => getSlotIndex(inp.name) === slotIdx
                        );

                        if (matchingInput && matchingInput.link != null) {
                            const type = getConnectedType(pipeInNode, matchingInput);
                            const label = getConnectedLabel(pipeInNode, matchingInput, this.properties.show_origin_title);
                            outEntry.label = label || `slot${String(slotIdx).padStart(2, "0")}`;
                            outEntry.type = (type && type !== "*") ? type : "*";
                            cacheSlots.push({ label: outEntry.label, type: outEntry.type });
                        } else {
                            outEntry.label = `slot${String(slotIdx).padStart(2, "0")}`;
                            outEntry.type = "*";
                            cacheSlots.push(null);
                        }
                        newOutputs.push(outEntry);
                    }
                    saveCache(cacheSlots);

                } else {
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
                }

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
                
                let changed = false;
                if (this.outputs?.length !== newOutputs.length) {
                    changed = true;
                } else {
                    for(let i = 0; i < newOutputs.length; i++) {
                        if(this.outputs[i]?.label !== newOutputs[i].label || this.outputs[i]?.type !== newOutputs[i].type) {
                            changed = true;
                            break;
                        }
                    }
                }

                if (changed) {
                    this.outputs = newOutputs; // Complete array replacement
                }

                if (changed || !this._pipeSizeSet) {
                    this._pipeSizeSet = true;
                    if (this.computeSize && this.setSize) {
                        const s = this.computeSize();
                        this.setSize([Math.max(this.size?.[0] ?? 180, s[0]), s[1]]);
                    }
                    
                    if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
                    app.canvas?.setDirty(true, true);

                    const graph = this.graph || app.graph;
                    if (graph?.trigger) {
                        // Force full Vue node UI re-evaluation by faking a property change
                        graph.trigger('node:property:changed', { nodeId: this.id, property: 'title', oldValue: this.title, newValue: this.title });
                        graph.trigger('node:slot-label:changed', { nodeId: this.id, slotType: 2 }); // OUTPUT
                    }
                    if (graph?.change) graph.change();
                }
            };

            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                const r = originalOnConnectionsChange?.apply(this, arguments);
                // Delay to allow LiteGraph to finish registering the link before tracing
                setTimeout(() => syncOutputs(), 0);
                return r;
            };

            let lastFP = "";
            const periodicSync = () => {
                if (this.flags?.collapsed) return;
                const pipeInNode = findConnectedPipeIn(this);
                let fp = `${this.properties.show_origin_title ? "1" : "0"}|`;
                if (pipeInNode) {
                    for (const inp of getManagedInputs(pipeInNode)) {
                        if (inp.link == null) continue;
                        fp += `${inp.name}:${getConnectedType(pipeInNode, inp)}:${getConnectedLabel(pipeInNode, inp, this.properties.show_origin_title)}|`;
                    }
                } else {
                    fp += `cache:${JSON.stringify(loadCache())}`;
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
