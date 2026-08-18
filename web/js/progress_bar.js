/**
 * [XENodes.ProgressBar]
 * - Subgraph-aware Progress Bar & Node Navigator for ComfyUI.
 * - Accurately tracks executing nodes across deep subgraph hierarchies (Nodes 2.0 Subgraphs & GroupNodes).
 * - Click anywhere on the progress bar to instantly drill down into nested subgraphs and smoothly center on the active node.
 * - Provides visual pulse/glow highlighting on focused nodes for effortless tracking.
 * - Right-click to quickly return to the root graph.
 * - Cleanly enabled or disabled via Settings -> XENodes -> Progress Bar.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[XENodes.ProgressBar] Initializing Progress Bar extension...");

// ============================================================================
// 1. Subgraph Hierarchy Search & Safe Traversal Utilities
// ============================================================================

/**
 * Safely extracts inner nodes from a subgraph parent node without throwing TypeErrors.
 * @param {LGraphNode} node
 * @returns {LGraphNode[]}
 */
function getInnerNodesSafely(node) {
    if (!node) return [];

    if (Array.isArray(node.innerNodes)) {
        return node.innerNodes;
    }

    const symbols = Object.getOwnPropertySymbols(node);
    for (const sym of symbols) {
        const val = node[sym];
        if (val && typeof val === "object" && Array.isArray(val.innerNodes)) {
            return val.innerNodes;
        }
    }

    if (typeof node.getInnerNodes === "function") {
        try {
            const map = new Map();
            const res = node.getInnerNodes(map, [], [], new Set());
            if (Array.isArray(res)) {
                return res.map(item => (item && item.node ? item.node : item));
            }
        } catch (e) {
            try {
                const map = new Map();
                const res = node.getInnerNodes(map);
                if (Array.isArray(res)) {
                    return res.map(item => (item && item.node ? item.node : item));
                }
            } catch (e2) {}
        }
    }

    return [];
}

/**
 * Traverses graph hierarchy to locate a node by its ID (flat ID, composite "10:25", or UUID).
 * Returns the target node, its owner graph (LGraph/Subgraph), and the path of parent subgraphs.
 * 
 * @param {LGraph} rootGraph
 * @param {string|number} nodeId
 * @param {Array<{graph: LGraph|Subgraph, title: string}>} [currentPath=[]]
 * @returns {{ node: LGraphNode, graph: LGraph|Subgraph, path: Array<{graph: LGraph|Subgraph, title: string}> } | null}
 */
function findNodeAndHierarchy(rootGraph, nodeId, currentPath = []) {
    if (!rootGraph || nodeId == null) return null;

    const idStr = String(nodeId).trim();
    if (!idStr) return null;

    // 1. Check direct match in root / current graph
    let directNode = rootGraph.getNodeById(idStr) || rootGraph.getNodeById(Number(idStr));
    if (directNode) {
        return {
            node: directNode,
            graph: rootGraph,
            path: currentPath
        };
    }

    // 2. Composite ID parsing (e.g. "10:25" or "10.25")
    const separators = [":", "."];
    for (const sep of separators) {
        if (idStr.includes(sep)) {
            const parts = idStr.split(sep).filter(p => p.length > 0);
            let curGraph = rootGraph;
            let path = [...currentPath];
            let foundNode = null;

            for (let i = 0; i < parts.length; i++) {
                const part = parts[i];
                if (!curGraph) break;

                const n = curGraph.getNodeById(part) || curGraph.getNodeById(Number(part));
                if (!n) {
                    foundNode = null;
                    break;
                }

                foundNode = n;
                if (n.subgraph) {
                    path.push({
                        graph: n.subgraph,
                        title: n.title || n.subgraph.name || `Subgraph #${n.id}`
                    });
                    curGraph = n.subgraph;
                } else if (typeof n.getInnerNodes === "function" || Array.isArray(n.innerNodes)) {
                    const innerNodes = getInnerNodesSafely(n);
                    const nextPart = parts[i + 1];
                    if (nextPart !== undefined && Array.isArray(innerNodes)) {
                        const targetInner = innerNodes.find(item => {
                            if (!item) return false;
                            const itemId = String(item.id);
                            return itemId.endsWith(":" + nextPart) || itemId === nextPart || item.index === Number(nextPart);
                        });
                        if (targetInner) {
                            foundNode = targetInner;
                            curGraph = targetInner.graph || curGraph;
                            i++;
                        }
                    }
                }
            }

            if (foundNode) {
                return {
                    node: foundNode,
                    graph: curGraph,
                    path: path
                };
            }
        }
    }

    // 3. Search in all subgraphs registered on root graph (Nodes 2.0 centralized registry)
    if (rootGraph.subgraphs && typeof rootGraph.subgraphs.values === "function") {
        for (const sub of rootGraph.subgraphs.values()) {
            if (!sub || sub === rootGraph) continue;
            const matchInSub = sub.getNodeById(idStr) || sub.getNodeById(Number(idStr));
            if (matchInSub) {
                return {
                    node: matchInSub,
                    graph: sub,
                    path: [...currentPath, { graph: sub, title: sub.name || `Subgraph` }]
                };
            }
        }
    }

    // 4. Recursive search across child nodes in graph._nodes
    const candidateNodes = rootGraph._nodes || rootGraph.nodes || [];
    for (const n of candidateNodes) {
        if (n.subgraph) {
            const subTitle = n.title || n.subgraph.name || `Subgraph #${n.id}`;
            const subResult = findNodeAndHierarchy(n.subgraph, nodeId, [
                ...currentPath,
                { graph: n.subgraph, title: subTitle }
            ]);
            if (subResult) return subResult;
        } else if (typeof n.getInnerNodes === "function" || Array.isArray(n.innerNodes)) {
            const inners = getInnerNodesSafely(n);
            if (Array.isArray(inners)) {
                const found = inners.find(inner => {
                    if (!inner) return false;
                    const iId = String(inner.id);
                    return iId === idStr || iId.split(":").at(-1) === idStr || (inner.index != null && String(inner.index) === idStr);
                });
                if (found) {
                    return {
                        node: found,
                        graph: found.graph || rootGraph,
                        path: [...currentPath, { graph: found.graph || rootGraph, title: n.title || `GroupNode #${n.id}` }]
                    };
                }
            }
        }
    }

    return null;
}

// ============================================================================
// 2. Pulse Highlight & Navigation Logic
// ============================================================================

let activeHighlightNode = null;
let activeHighlightEnd = 0;

/**
 * Triggers a vibrant glowing pulse highlight around the focused node.
 * @param {LGraphNode} node
 */
function pulseHighlightNode(node) {
    if (!node) return;
    activeHighlightNode = node;
    activeHighlightEnd = performance.now() + 2000; // 2 seconds pulse

    if (app.canvas) {
        app.canvas.setDirty(true, true);
    }
}

/**
 * Seamlessly navigates the canvas to the target graph and focuses on the target node.
 * @param {LGraph|Subgraph} targetGraph
 * @param {LGraphNode} targetNode
 */
async function navigateAndFocusNode(targetGraph, targetNode) {
    const canvas = app.canvas;
    if (!canvas || !targetNode) return;

    const rootGraph = app.rootGraph || app.graph;

    // 1. Switch graph viewport if target graph differs from current active graph
    if (targetGraph && canvas.graph !== targetGraph) {
        const isRoot = !targetGraph || targetGraph === rootGraph || targetGraph.isRootGraph;
        canvas.subgraph = isRoot ? undefined : targetGraph;

        if (typeof canvas.setGraph === "function") {
            canvas.setGraph(targetGraph);
        }

        // Wait two animation frames for LiteGraph / Vue Nodes to mount and layout
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    }

    // 2. Center and zoom on the target node
    if (typeof canvas.centerOnNode === "function") {
        canvas.centerOnNode(targetNode);
    } else if (typeof canvas.animateToBounds === "function" && targetNode.boundingRect) {
        canvas.animateToBounds(targetNode.boundingRect);
    }

    // 3. Select node
    if (typeof canvas.selectNode === "function") {
        canvas.selectNode(targetNode, false);
    }

    // 4. Trigger pulse glow highlight
    pulseHighlightNode(targetNode);
}

/**
 * Installs canvas draw hook for rendering the glowing pulse effect around focused nodes.
 */
function setupPulseDrawHook() {
    const LGraphCanvas = globalThis.LGraphCanvas || (globalThis.LiteGraph && globalThis.LiteGraph.LGraphCanvas);
    if (!LGraphCanvas || !LGraphCanvas.prototype || LGraphCanvas.prototype.__xenodes_pulse_hooked) return;

    LGraphCanvas.prototype.__xenodes_pulse_hooked = true;
    const originalDrawNode = LGraphCanvas.prototype.drawNode;

    LGraphCanvas.prototype.drawNode = function (node, ctx) {
        originalDrawNode.apply(this, arguments);

        if (activeHighlightNode && node === activeHighlightNode) {
            const now = performance.now();
            if (now < activeHighlightEnd) {
                const remaining = (activeHighlightEnd - now) / 2000; // 1.0 -> 0.0
                const pulse = Math.sin((1 - remaining) * Math.PI * 4) * 0.3 + 0.7; // Pulse frequency

                ctx.save();
                const pad = 8 + (1 - remaining) * 6;
                const radius = 10;
                const x = -pad;
                const y = -pad;
                const w = node.size[0] + pad * 2;
                const h = node.size[1] + pad * 2;

                ctx.lineWidth = 3.5;
                ctx.strokeStyle = `rgba(16, 185, 129, ${(remaining * pulse * 0.9).toFixed(2)})`; // Emerald Glow
                ctx.shadowColor = `rgba(52, 211, 153, 0.8)`;
                ctx.shadowBlur = 16 * pulse;

                // Draw rounded rectangle
                ctx.beginPath();
                if (ctx.roundRect) {
                    ctx.roundRect(x, y, w, h, radius);
                } else {
                    ctx.rect(x, y, w, h);
                }
                ctx.stroke();
                ctx.restore();

                // Keep repainting while animating
                this.setDirty(true, true);
            } else {
                activeHighlightNode = null;
            }
        }
    };
}

// ============================================================================
// 3. Web Component: <xenodes-progress-bar>
// ============================================================================

class XENodesProgressBarElement extends HTMLElement {
    static TAG = "xenodes-progress-bar";

    constructor() {
        super();
        this.attachShadow({ mode: "open" });
        this.currentNodeId = null;
        this.currentQueue = 0;
        this.totalNodes = 0;
        this.currentStep = 0;
        this.maxSteps = 0;
        this.nodeTitle = "";
        this.hierarchyPathStr = "";
        this.isError = false;
        this.errorMessage = "";
        this.isExecuting = false;

        this.render();
    }

    connectedCallback() {
        this.setupInteractions();
    }

    setupInteractions() {
        // Left-Click: Drill-down and navigate to executing node inside subgraph
        this.addEventListener("pointerdown", async (e) => {
            if (e.button === 0) { // Left click
                e.stopPropagation();
                e.preventDefault();

                if (!this.currentNodeId) return;

                const rootGraph = app.rootGraph || app.graph;
                const result = findNodeAndHierarchy(rootGraph, this.currentNodeId);
                if (result && result.node) {
                    await navigateAndFocusNode(result.graph, result.node);
                }
            } else if (e.button === 2) { // Right click: Quick return to root graph
                e.stopPropagation();
                e.preventDefault();

                const rootGraph = app.rootGraph || app.graph;
                const canvas = app.canvas;
                if (canvas && rootGraph && canvas.graph !== rootGraph) {
                    canvas.subgraph = undefined;
                    if (typeof canvas.setGraph === "function") {
                        canvas.setGraph(rootGraph);
                    }
                }
            }
        });

        this.addEventListener("contextmenu", (e) => {
            e.stopPropagation();
            e.preventDefault();
        });
    }

    updateState({ currentNodeId, currentQueue, totalNodes, executedNodesCount, step, maxSteps, nodeTitle, hierarchyPath, isError, errorMessage, isExecuting }) {
        if (currentNodeId !== undefined) this.currentNodeId = currentNodeId;
        if (currentQueue !== undefined) this.currentQueue = currentQueue;
        if (totalNodes !== undefined) this.totalNodes = totalNodes;
        if (step !== undefined) this.currentStep = step;
        if (maxSteps !== undefined) this.maxSteps = maxSteps;
        if (nodeTitle !== undefined) this.nodeTitle = nodeTitle;
        if (hierarchyPath !== undefined) this.hierarchyPathStr = hierarchyPath;
        if (isError !== undefined) this.isError = isError;
        if (errorMessage !== undefined) this.errorMessage = errorMessage;
        if (isExecuting !== undefined) this.isExecuting = isExecuting;

        this.updateView(executedNodesCount);
    }

    updateView(executedNodesCount = 0) {
        if (!this.shadowRoot) return;

        const container = this.shadowRoot.querySelector(".progress-container");
        const barOverall = this.shadowRoot.querySelector(".bar-overall");
        const barStep = this.shadowRoot.querySelector(".bar-step");
        const textMain = this.shadowRoot.querySelector(".progress-text");

        if (!container || !barOverall || !barStep || !textMain) return;

        if (this.isError) {
            container.classList.add("-error");
            barOverall.style.width = "100%";
            barStep.style.width = "0%";
            textMain.textContent = `⚠️ ${this.errorMessage || "Execution Error"}`;
            return;
        }

        container.classList.remove("-error");

        if (!this.isExecuting) {
            if (this.currentQueue > 0) {
                textMain.textContent = `(${this.currentQueue}) In Queue...`;
            } else {
                textMain.textContent = `Idle`;
            }
            barOverall.style.width = "0%";
            barStep.style.width = "0%";
            return;
        }

        // Overall progress percentage
        let overallPercent = 0;
        if (this.totalNodes > 0) {
            overallPercent = Math.min(100, Math.round((executedNodesCount / this.totalNodes) * 100));
            barOverall.style.width = `${Math.max(2, overallPercent)}%`;
        } else {
            barOverall.style.width = "2%";
        }

        // Step progress percentage
        let stepPercent = 0;
        let stepText = "";
        if (this.maxSteps > 0 && this.currentStep != null) {
            stepPercent = Math.min(100, Math.round((this.currentStep / this.maxSteps) * 100));
            barStep.style.width = `${stepPercent}%`;
            stepText = ` (${this.currentStep}/${this.maxSteps} - ${stepPercent}%)`;
        } else {
            barStep.style.width = "0%";
        }

        const pathPrefix = this.hierarchyPathStr ? `[${this.hierarchyPathStr}] ` : "";
        const queuePart = `(${this.currentQueue || 1}) `;
        const percentPart = `${overallPercent}%`;
        const nodePart = this.nodeTitle ? ` - ${pathPrefix}${this.nodeTitle}${stepText}` : "";

        textMain.textContent = `${queuePart}${percentPart}${nodePart}`;
    }

    render() {
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    width: 100%;
                    user-select: none;
                    z-index: 1000;
                    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }
                :host([hidden]) {
                    display: none !important;
                }
                .progress-container {
                    position: relative;
                    width: 100%;
                    height: 16px;
                    background: rgba(15, 23, 42, 0.92);
                    backdrop-filter: blur(8px);
                    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                    cursor: pointer;
                    overflow: hidden;
                    transition: background-color 0.2s ease;
                }
                .progress-container:hover {
                    background: rgba(30, 41, 59, 0.96);
                }
                /* Track Divider Line */
                .bar-divider {
                    position: absolute;
                    top: 50%;
                    left: 0;
                    width: 100%;
                    height: 1px;
                    background: rgba(0, 0, 0, 0.4);
                    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.05);
                    z-index: 2;
                    pointer-events: none;
                }
                /* Top Bar: Overall Workflow Progress */
                .bar-overall {
                    position: absolute;
                    top: 0;
                    left: 0;
                    height: 50%;
                    width: 0%;
                    background: linear-gradient(90deg, #059669 0%, #10b981 100%);
                    box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.2);
                    transition: width 0.15s ease-out;
                    pointer-events: none;
                    z-index: 1;
                }
                /* Bottom Bar: Current Node Step Progress */
                .bar-step {
                    position: absolute;
                    top: 50%;
                    left: 0;
                    height: 50%;
                    width: 0%;
                    background: linear-gradient(90deg, #0284c7 0%, #06b6d4 100%);
                    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
                    transition: width 0.1s ease-out;
                    pointer-events: none;
                    z-index: 1;
                }
                .progress-container.-error .bar-overall,
                .progress-container.-error .bar-step {
                    background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%) !important;
                    opacity: 0.95;
                }
                .progress-text {
                    position: relative;
                    z-index: 3;
                    display: flex;
                    align-items: center;
                    justify-content: flex-start;
                    height: 100%;
                    padding: 0 8px;
                    font-size: 10.5px;
                    font-weight: 700;
                    line-height: 16px;
                    color: #ffffff;
                    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.95), 0 0 4px rgba(0, 0, 0, 0.6);
                    letter-spacing: 0.2px;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
            </style>
            <div class="progress-container" title="XENodes Progress Bar&#013;• Top Bar (Green): Overall Workflow Progress&#013;• Bottom Bar (Cyan): Active Node Step Progress&#013;• Left-Click: Jump into subgraph and focus active node&#013;• Right-Click: Return to root graph">
                <div class="bar-overall"></div>
                <div class="bar-divider"></div>
                <div class="bar-step"></div>
                <div class="progress-text">Idle</div>
            </div>
        `;
    }
}

if (!customElements.get(XENodesProgressBarElement.TAG)) {
    customElements.define(XENodesProgressBarElement.TAG, XENodesProgressBarElement);
}

// ============================================================================
// 4. Extension Core Lifecycle & Execution State Tracking
// ============================================================================

let progressBarInstance = null;
let currentPromptId = null;
let totalNodesInPrompt = 0;
let executedNodeIds = new Set();
let currentExecutingNodeId = null;
let promptApiData = null;

/**
 * Checks whether the Progress Bar is enabled by the user.
 * @param {boolean} [explicitVal]
 * @returns {boolean}
 */
function isProgressBarEnabled(explicitVal) {
    if (explicitVal !== undefined) {
        return explicitVal !== false && explicitVal !== "false" && explicitVal !== 0;
    }
    const val = app.ui?.settings?.getSettingValue("XENodes.ProgressBar.Enabled");
    if (val === false || val === "false" || val === 0) {
        return false;
    }
    return true;
}

/**
 * Resolves node label and its subgraph hierarchy path string.
 * @param {string|number} nodeId
 * @returns {{ title: string, pathStr: string }}
 */
function resolveNodeDisplayInfo(nodeId) {
    if (!nodeId) return { title: "", pathStr: "" };

    const rootGraph = app.rootGraph || app.graph;
    const res = findNodeAndHierarchy(rootGraph, nodeId);

    let title = "";
    let pathStr = "";

    if (res && res.node) {
        title = res.node.title || res.node.type || `Node #${res.node.id}`;
        if (Array.isArray(res.path) && res.path.length > 0) {
            pathStr = res.path.map(p => p.title).join(" > ");
        }
    } else if (promptApiData && promptApiData[String(nodeId)]) {
        const apiNode = promptApiData[String(nodeId)];
        title = apiNode._meta?.title || apiNode.class_type || `Node #${nodeId}`;
    } else {
        title = `Node #${nodeId}`;
    }

    return { title, pathStr };
}

/**
 * Mounts, updates, or completely removes the progress bar DOM element.
 * @param {boolean} [explicitEnabled] Optional explicit boolean value from onChange
 */
function syncProgressBarDOM(explicitEnabled) {
    const isEnabled = isProgressBarEnabled(explicitEnabled);

    if (!isEnabled) {
        // Completely hide and remove all instances from DOM
        if (progressBarInstance) {
            progressBarInstance.style.display = "none";
            progressBarInstance.hidden = true;
            if (progressBarInstance.parentNode) {
                progressBarInstance.parentNode.removeChild(progressBarInstance);
            }
        }
        document.querySelectorAll(XENodesProgressBarElement.TAG).forEach(el => {
            el.style.display = "none";
            el.hidden = true;
            if (el.parentNode) el.parentNode.removeChild(el);
        });
        return;
    }

    if (!progressBarInstance) {
        progressBarInstance = document.createElement(XENodesProgressBarElement.TAG);
    }

    progressBarInstance.style.display = "block";
    progressBarInstance.hidden = false;

    // Mount to ComfyUI body slot if available, otherwise document.body
    const topSlot = document.querySelector(".comfyui-body-top");

    if (topSlot) {
        if (progressBarInstance.parentNode !== topSlot) {
            topSlot.prepend(progressBarInstance);
        }
    } else {
        progressBarInstance.style.position = "fixed";
        progressBarInstance.style.top = "0";
        progressBarInstance.style.left = "0";
        progressBarInstance.style.width = "100%";
        if (progressBarInstance.parentNode !== document.body) {
            document.body.prepend(progressBarInstance);
        }
    }
}

// ============================================================================
// 5. Register Extension with ComfyUI
// ============================================================================

app.registerExtension({
    name: "XENodes.ProgressBar",

    // Expose extension settings array for standard ComfyUI registration
    settings: [
        {
            id: "XENodes.ProgressBar.Enabled",
            category: ["XENodes", "Progress Bar"],
            name: "Display Progress Bar",
            type: "boolean",
            defaultValue: true,
            onChange: (newValue) => {
                syncProgressBarDOM(newValue);
            }
        }
    ],

    async setup() {
        console.log("[XENodes.ProgressBar] Setting up Progress Bar hooks...");

        // Ensure setting is also registered via addSetting for fallback compatibility
        app.ui?.settings?.addSetting({
            id: "XENodes.ProgressBar.Enabled",
            category: ["XENodes", "Progress Bar"],
            name: "Display Progress Bar",
            type: "boolean",
            defaultValue: true,
            onChange: (newValue) => {
                syncProgressBarDOM(newValue);
            }
        });

        // Setup pulse glow rendering hook on LiteGraphCanvas
        setupPulseDrawHook();

        // Initial sync of Progress Bar DOM based on user settings
        syncProgressBarDOM();

        // Hook API queuePrompt to capture totalNodes & prompt structure
        const originalQueuePrompt = api.queuePrompt;
        if (typeof originalQueuePrompt === "function") {
            api.queuePrompt = async function (num, prompt, ...args) {
                const response = await originalQueuePrompt.apply(api, [num, prompt, ...args]);
                if (response && response.prompt_id && prompt && prompt.output) {
                    promptApiData = prompt.output;
                    totalNodesInPrompt = Object.keys(prompt.output).length;
                }
                return response;
            };
        }

        // Connect API Listeners
        api.addEventListener("status", (e) => {
            if (!isProgressBarEnabled()) {
                syncProgressBarDOM(false);
                return;
            }
            const queueRemaining = e.detail?.exec_info?.queue_remaining || 0;
            if (progressBarInstance) {
                progressBarInstance.updateState({
                    currentQueue: queueRemaining,
                    isExecuting: queueRemaining > 0 && currentExecutingNodeId != null
                });
            }
        });

        api.addEventListener("execution_start", (e) => {
            if (!isProgressBarEnabled()) {
                syncProgressBarDOM(false);
                return;
            }
            currentPromptId = e.detail?.prompt_id;
            executedNodeIds.clear();
            currentExecutingNodeId = null;

            if (progressBarInstance) {
                progressBarInstance.updateState({
                    currentNodeId: null,
                    totalNodes: totalNodesInPrompt,
                    executedNodesCount: 0,
                    step: 0,
                    maxSteps: 0,
                    nodeTitle: "",
                    hierarchyPath: "",
                    isError: false,
                    isExecuting: true
                });
            }
        });

        api.addEventListener("executing", (e) => {
            if (!isProgressBarEnabled()) {
                syncProgressBarDOM(false);
                return;
            }

            let nodeId = null;
            if (e.detail !== null && e.detail !== undefined) {
                if (typeof e.detail === "object") {
                    nodeId = e.detail.node || e.detail.display_node || null;
                } else {
                    nodeId = e.detail;
                }
            }

            if (nodeId == null) {
                currentExecutingNodeId = null;
                if (progressBarInstance) {
                    progressBarInstance.updateState({
                        currentNodeId: null,
                        step: 0,
                        maxSteps: 0,
                        nodeTitle: "",
                        hierarchyPath: "",
                        isExecuting: false
                    });
                }
                return;
            }

            if (currentExecutingNodeId && currentExecutingNodeId !== nodeId) {
                executedNodeIds.add(currentExecutingNodeId);
            }

            currentExecutingNodeId = nodeId;
            const info = resolveNodeDisplayInfo(nodeId);

            if (progressBarInstance) {
                progressBarInstance.updateState({
                    currentNodeId: nodeId,
                    executedNodesCount: executedNodeIds.size,
                    step: 0,
                    maxSteps: 0,
                    nodeTitle: info.title,
                    hierarchyPath: info.pathStr,
                    isError: false,
                    isExecuting: true
                });
            }
        });

        api.addEventListener("progress", (e) => {
            if (!isProgressBarEnabled()) {
                syncProgressBarDOM(false);
                return;
            }
            if (!progressBarInstance || !e.detail) return;

            const nodeId = e.detail.node || currentExecutingNodeId;
            const step = e.detail.value;
            const maxSteps = e.detail.max;

            if (nodeId && nodeId !== currentExecutingNodeId) {
                if (currentExecutingNodeId) executedNodeIds.add(currentExecutingNodeId);
                currentExecutingNodeId = nodeId;
            }

            const info = resolveNodeDisplayInfo(currentExecutingNodeId);

            progressBarInstance.updateState({
                currentNodeId: currentExecutingNodeId,
                executedNodesCount: executedNodeIds.size,
                step: step,
                maxSteps: maxSteps,
                nodeTitle: info.title,
                hierarchyPath: info.pathStr,
                isExecuting: true
            });
        });

        api.addEventListener("execution_cached", (e) => {
            if (!isProgressBarEnabled()) return;
            if (e.detail?.nodes && Array.isArray(e.detail.nodes)) {
                for (const c of e.detail.nodes) {
                    executedNodeIds.add(String(c));
                }
                if (progressBarInstance) {
                    progressBarInstance.updateState({
                        executedNodesCount: executedNodeIds.size
                    });
                }
            }
        });

        api.addEventListener("execution_error", (e) => {
            if (!isProgressBarEnabled()) return;
            if (progressBarInstance && e.detail) {
                const errDetail = e.detail;
                const errNodeId = errDetail.node_id;
                const errMsg = errDetail.exception_message || errDetail.exception_type || "Error occurred";

                currentExecutingNodeId = errNodeId;
                progressBarInstance.updateState({
                    currentNodeId: errNodeId,
                    isError: true,
                    errorMessage: `${errMsg} (Node #${errNodeId})`,
                    isExecuting: false
                });
            }
        });

        console.log("[XENodes.ProgressBar] Extension setup completed successfully.");
    }
});
