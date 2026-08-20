/**
 * [XENodes.ProgressBar]
 * - Subgraph-aware Progress Bar & Node Navigator for ComfyUI.
 * - Deeply tracks executing nodes across complex subgraph hierarchies (Nodes 2.0 Subgraphs & GroupNodes).
 * - Multi-Workflow Tab Aware: Automatically switches to the executing workflow tab if another tab is focused.
 * - Multi-Queue Robust: Isolates job progress per prompt_id so progress never resets when queuing new runs.
 * - Click to smoothly jump/drill-down into executing subgraphs and highlight nodes with an animated glow.
 * - Right-click to quickly return to the root graph.
 * - Cleanly enabled/disabled via Settings -> XENodes -> Progress Bar.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ============================================================================
// 1. Subgraph Hierarchy Search & Safe Traversal Utilities
// ============================================================================

/**
 * Safely extracts inner nodes from a subgraph/group parent node.
 * @param {LGraphNode} node
 * @returns {LGraphNode[]}
 */
function getInnerNodesSafely(node) {
    if (!node) return [];
    if (Array.isArray(node.innerNodes)) return node.innerNodes;

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
            if (Array.isArray(res)) return res.map(item => (item?.node ? item.node : item));
        } catch {
            try {
                const res = node.getInnerNodes(new Map());
                if (Array.isArray(res)) return res.map(item => (item?.node ? item.node : item));
            } catch {}
        }
    }
    return [];
}

/**
 * Traverses graph hierarchy to locate a node by ID (flat ID, composite "10:25", or UUID).
 * @param {LGraph} rootGraph
 * @param {string|number} nodeId
 * @param {Array<{graph: LGraph|Subgraph, title: string}>} [currentPath=[]]
 * @returns {{ node: LGraphNode, graph: LGraph|Subgraph, path: Array<{graph: LGraph|Subgraph, title: string}> } | null}
 */
function findNodeAndHierarchy(rootGraph, nodeId, currentPath = []) {
    if (!rootGraph || nodeId == null) return null;
    const idStr = String(nodeId).trim();
    if (!idStr) return null;

    // 1. Direct match in root / current graph
    const directNode = rootGraph.getNodeById(idStr) || rootGraph.getNodeById(Number(idStr));
    if (directNode) {
        return { node: directNode, graph: rootGraph, path: currentPath };
    }

    // 2. Composite ID parsing (e.g. "10:25" or "10.25")
    for (const sep of [":", "."]) {
        if (idStr.includes(sep)) {
            const parts = idStr.split(sep).filter(p => p.length > 0);
            let curGraph = rootGraph;
            let path = [...currentPath];
            let foundNode = null;

            for (let i = 0; i < parts.length; i++) {
                const part = parts[i];
                if (!curGraph) break;
                const n = curGraph.getNodeById(part) || curGraph.getNodeById(Number(part));
                if (!n) { foundNode = null; break; }

                foundNode = n;
                if (n.subgraph) {
                    path.push({ graph: n.subgraph, title: n.title || n.subgraph.name || `Subgraph #${n.id}` });
                    curGraph = n.subgraph;
                } else if (typeof n.getInnerNodes === "function" || Array.isArray(n.innerNodes)) {
                    const inners = getInnerNodesSafely(n);
                    const nextPart = parts[i + 1];
                    if (nextPart !== undefined && Array.isArray(inners)) {
                        const targetInner = inners.find(item => {
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
            if (foundNode) return { node: foundNode, graph: curGraph, path };
        }
    }

    // 3. Search in all subgraphs registered on root graph
    if (rootGraph.subgraphs && typeof rootGraph.subgraphs.values === "function") {
        for (const sub of rootGraph.subgraphs.values()) {
            if (!sub || sub === rootGraph) continue;
            const matchInSub = sub.getNodeById(idStr) || sub.getNodeById(Number(idStr));
            if (matchInSub) {
                return { node: matchInSub, graph: sub, path: [...currentPath, { graph: sub, title: sub.name || "Subgraph" }] };
            }
        }
    }

    // 4. Recursive search across child nodes
    const candidateNodes = rootGraph._nodes || rootGraph.nodes || [];
    for (const n of candidateNodes) {
        if (n.subgraph) {
            const subTitle = n.title || n.subgraph.name || `Subgraph #${n.id}`;
            const subResult = findNodeAndHierarchy(n.subgraph, nodeId, [...currentPath, { graph: n.subgraph, title: subTitle }]);
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
                    return { node: found, graph: found.graph || rootGraph, path: [...currentPath, { graph: found.graph || rootGraph, title: n.title || `GroupNode #${n.id}` }] };
                }
            }
        }
    }
    return null;
}

// ============================================================================
// 2. Pulse Highlight & Multi-Tab Navigation Logic
// ============================================================================

let activeHighlightNode = null;
let activeHighlightEnd = 0;

function simulateFullClick(el) {
    if (!el) return;
    const opts = { bubbles: true, cancelable: true, view: window };
    el.dispatchEvent(new PointerEvent("pointerdown", opts));
    el.dispatchEvent(new MouseEvent("mousedown", opts));
    el.dispatchEvent(new PointerEvent("pointerup", opts));
    el.dispatchEvent(new MouseEvent("mouseup", opts));
    el.dispatchEvent(new MouseEvent("click", opts));
    if (typeof el.click === "function") el.click();
}

function getActiveWorkflowTabName() {
    const activeLabel = document.querySelector(".workflow-tabs .p-togglebutton-checked .workflow-label")
        || document.querySelector(".workflow-tab.active .workflow-label")
        || document.querySelector(".p-togglebutton-checked .workflow-label");
    return activeLabel?.textContent ? activeLabel.textContent.trim() : null;
}

function switchTabToExecutingWorkflow(targetTabName) {
    const tabButtons = Array.from(document.querySelectorAll(".workflow-tabs .p-togglebutton, .workflow-tabs-container .p-togglebutton, .workflow-tab"));
    if (tabButtons.length === 0) return false;

    let targetTabBtn = null;

    // Match by target tab name
    if (targetTabName) {
        for (const el of tabButtons) {
            const toggleBtn = el.closest(".p-togglebutton") || el;
            if (toggleBtn.classList.contains("p-togglebutton-checked") || el.classList.contains("active")) continue;
            const label = el.querySelector(".workflow-label") || el;
            if (label?.textContent?.trim() === targetTabName) {
                targetTabBtn = toggleBtn;
                break;
            }
        }
    }

    // Match by active execution spinner
    if (!targetTabBtn) {
        for (const el of tabButtons) {
            const toggleBtn = el.closest(".p-togglebutton") || el;
            if (toggleBtn.classList.contains("p-togglebutton-checked") || el.classList.contains("active")) continue;
            if (el.querySelector(".animate-spin, [class*='animate-spin'], [class*='loader'], [class*='spin'], .pi-spin, [role='img']")) {
                targetTabBtn = toggleBtn;
                break;
            }
        }
    }

    // Fallback: solitary inactive tab
    if (!targetTabBtn) {
        const inactive = tabButtons.filter(el => {
            const toggleBtn = el.closest(".p-togglebutton") || el;
            return !toggleBtn.classList.contains("p-togglebutton-checked") && !el.classList.contains("active");
        });
        if (inactive.length === 1) targetTabBtn = inactive[0].closest(".p-togglebutton") || inactive[0];
    }

    if (targetTabBtn) {
        simulateFullClick(targetTabBtn);
        simulateFullClick(targetTabBtn.querySelector(".workflow-tab") || targetTabBtn);
        return true;
    }
    return false;
}

function pulseHighlightNode(node) {
    if (!node) return;
    activeHighlightNode = node;
    activeHighlightEnd = performance.now() + 2000;
    if (app.canvas) app.canvas.setDirty(true, true);
}

async function navigateAndFocusNode(targetGraph, targetNode) {
    const canvas = app.canvas;
    if (!canvas || !targetNode) return;
    const rootGraph = app.rootGraph || app.graph;

    if (targetGraph && canvas.graph !== targetGraph) {
        const isRoot = !targetGraph || targetGraph === rootGraph || targetGraph.isRootGraph;
        canvas.subgraph = isRoot ? undefined : targetGraph;
        if (typeof canvas.setGraph === "function") canvas.setGraph(targetGraph);
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    }

    if (typeof canvas.centerOnNode === "function") {
        canvas.centerOnNode(targetNode);
    } else if (typeof canvas.animateToBounds === "function" && targetNode.boundingRect) {
        canvas.animateToBounds(targetNode.boundingRect);
    }

    if (typeof canvas.selectNode === "function") {
        canvas.selectNode(targetNode, false);
    }
    pulseHighlightNode(targetNode);
}

function setupPulseDrawHook() {
    const LGraphCanvas = globalThis.LGraphCanvas || (globalThis.LiteGraph && globalThis.LiteGraph.LGraphCanvas);
    if (!LGraphCanvas?.prototype || LGraphCanvas.prototype.__xenodes_pulse_hooked) return;

    LGraphCanvas.prototype.__xenodes_pulse_hooked = true;
    const originalDrawNode = LGraphCanvas.prototype.drawNode;

    LGraphCanvas.prototype.drawNode = function (node, ctx) {
        originalDrawNode.apply(this, arguments);

        if (activeHighlightNode && node === activeHighlightNode) {
            const now = performance.now();
            if (now < activeHighlightEnd) {
                const remaining = (activeHighlightEnd - now) / 2000;
                const pulse = Math.sin((1 - remaining) * Math.PI * 4) * 0.3 + 0.7;

                ctx.save();
                const pad = 8 + (1 - remaining) * 6;
                const x = -pad;
                const y = -pad;
                const w = node.size[0] + pad * 2;
                const h = node.size[1] + pad * 2;

                ctx.lineWidth = 3.5;
                ctx.strokeStyle = `rgba(16, 185, 129, ${(remaining * pulse * 0.9).toFixed(2)})`;
                ctx.shadowColor = "rgba(52, 211, 153, 0.8)";
                ctx.shadowBlur = 16 * pulse;

                ctx.beginPath();
                if (ctx.roundRect) ctx.roundRect(x, y, w, h, 10);
                else ctx.rect(x, y, w, h);
                ctx.stroke();
                ctx.restore();

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
        this.executedNodesCount = 0;
        this.currentStep = 0;
        this.maxSteps = 0;
        this.nodeTitle = "";
        this.hierarchyPathStr = "";
        this.isError = false;
        this.errorMessage = "";
        this.isExecuting = false;
        this.currentExecutingTabName = null;

        this.render();
    }

    connectedCallback() {
        this.setupInteractions();
    }

    setupInteractions() {
        this.addEventListener("pointerdown", async (e) => {
            if (e.button === 0) { // Left click: Focus node
                e.stopPropagation();
                e.preventDefault();
                if (!this.currentNodeId) return;

                let rootGraph = app.rootGraph || app.graph;
                let result = findNodeAndHierarchy(rootGraph, this.currentNodeId);

                if (!result || !result.node) {
                    const switchAttempted = switchTabToExecutingWorkflow(this.currentExecutingTabName);
                    if (switchAttempted) {
                        const start = performance.now();
                        while (performance.now() - start < 1000) {
                            await new Promise(r => setTimeout(r, 50));
                            rootGraph = app.rootGraph || app.graph;
                            result = findNodeAndHierarchy(rootGraph, this.currentNodeId);
                            if (result?.node) break;
                        }
                    }
                }

                if (result?.node) {
                    await navigateAndFocusNode(result.graph, result.node);
                }
            } else if (e.button === 2) { // Right click: Return to root
                e.stopPropagation();
                e.preventDefault();
                const rootGraph = app.rootGraph || app.graph;
                const canvas = app.canvas;
                if (canvas && rootGraph && canvas.graph !== rootGraph) {
                    canvas.subgraph = undefined;
                    if (typeof canvas.setGraph === "function") canvas.setGraph(rootGraph);
                }
            }
        });

        this.addEventListener("contextmenu", (e) => {
            e.stopPropagation();
            e.preventDefault();
        });
    }

    updateState({ currentNodeId, currentQueue, totalNodes, executedNodesCount, step, maxSteps, nodeTitle, hierarchyPath, isError, errorMessage, isExecuting, tabName }) {
        if (currentNodeId !== undefined) this.currentNodeId = currentNodeId;
        if (currentQueue !== undefined) this.currentQueue = currentQueue;
        if (totalNodes !== undefined) this.totalNodes = totalNodes;
        if (executedNodesCount !== undefined) this.executedNodesCount = executedNodesCount;
        if (step !== undefined) this.currentStep = step;
        if (maxSteps !== undefined) this.maxSteps = maxSteps;
        if (nodeTitle !== undefined) this.nodeTitle = nodeTitle;
        if (hierarchyPath !== undefined) this.hierarchyPathStr = hierarchyPath;
        if (isError !== undefined) this.isError = isError;
        if (errorMessage !== undefined) this.errorMessage = errorMessage;
        if (isExecuting !== undefined) this.isExecuting = isExecuting;
        if (tabName !== undefined) this.currentExecutingTabName = tabName;

        this.updateView();
    }

    updateView() {
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
            textMain.textContent = this.currentQueue > 0 ? `(${this.currentQueue}) In Queue...` : "Idle";
            barOverall.style.width = "0%";
            barStep.style.width = "0%";
            return;
        }

        // Overall progress percentage
        let overallPercent = 0;
        if (this.totalNodes > 0) {
            overallPercent = Math.min(100, Math.round((this.executedNodesCount / this.totalNodes) * 100));
            barOverall.style.width = `${Math.max(2, overallPercent)}%`;
        } else {
            barOverall.style.width = "2%";
        }

        // Step progress percentage
        let stepText = "";
        const hasStep = this.maxSteps > 0 && this.currentStep != null;
        if (hasStep) {
            const stepPercent = Math.min(100, Math.round((this.currentStep / this.maxSteps) * 100));
            barStep.style.width = `${stepPercent}%`;
            barStep.style.height = "50%";
            barOverall.style.height = "50%";
            stepText = ` (${stepPercent}%)`;
        } else {
            barStep.style.width = "0%";
            barStep.style.height = "0%";
            barOverall.style.height = "100%";
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
                :host([hidden]) { display: none !important; }
                .progress-container {
                    position: relative;
                    width: 100%;
                    height: 14px;
                    background: rgba(15, 23, 42, 0.92);
                    backdrop-filter: blur(8px);
                    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                    cursor: pointer;
                    overflow: hidden;
                    transition: background-color 0.2s ease;
                }
                .progress-container:hover { background: rgba(30, 41, 59, 0.96); }
                .bar-overall {
                    position: absolute;
                    top: 0;
                    left: 0;
                    height: 100%;
                    width: 0%;
                    background: linear-gradient(90deg, #059669 0%, #10b981 100%);
                    opacity: 0.85;
                    transition: width 0.15s ease-out, height 0.15s ease-out;
                    pointer-events: none;
                    z-index: 1;
                }
                .bar-step {
                    position: absolute;
                    top: 50%;
                    left: 0;
                    height: 50%;
                    width: 0%;
                    background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%);
                    opacity: 0.85;
                    transition: width 0.1s ease-out;
                    pointer-events: none;
                    z-index: 2;
                }
                .progress-container.-error .bar-overall {
                    background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%) !important;
                    opacity: 0.9;
                }
                .progress-text {
                    position: relative;
                    z-index: 2;
                    display: flex;
                    align-items: center;
                    justify-content: flex-start;
                    height: 100%;
                    padding: 0 6px;
                    font-size: 10px;
                    font-weight: 700;
                    line-height: 14px;
                    color: #ffffff;
                    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.85);
                    letter-spacing: 0.2px;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
            </style>
            <div class="progress-container" title="XENodes Progress Bar&#013;Left-Click: Jump to active node&#013;Right-Click: Return to root graph">
                <div class="bar-overall"></div>
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
// 4. Extension Core Lifecycle & Multi-Prompt State Tracking
// ============================================================================

let progressBarInstance = null;
let currentPromptId = null;
let currentExecutingNodeId = null;
const promptsMap = new Map();

function getOrInitPrompt(promptId) {
    if (!promptsMap.has(promptId)) {
        promptsMap.set(promptId, {
            promptApiData: null,
            totalNodes: 0,
            executedNodeIds: new Set(),
            tabName: null
        });
    }
    return promptsMap.get(promptId);
}

function isProgressBarEnabled(explicitVal) {
    if (explicitVal !== undefined) {
        return explicitVal !== false && explicitVal !== "false" && explicitVal !== 0;
    }
    const val = app.ui?.settings?.getSettingValue("XENodes.ProgressBar.Enabled");
    return val !== false && val !== "false" && val !== 0;
}

function resolveNodeDisplayInfo(nodeId, promptApiData) {
    if (!nodeId) return { title: "", pathStr: "" };
    const rootGraph = app.rootGraph || app.graph;
    const res = findNodeAndHierarchy(rootGraph, nodeId);

    let title = "";
    let pathStr = "";
    if (res?.node) {
        title = res.node.title || res.node.type || `Node #${res.node.id}`;
        if (Array.isArray(res.path) && res.path.length > 0) {
            pathStr = res.path.map(p => p.title).join(" > ");
        }
    } else if (promptApiData?.[String(nodeId)]) {
        const apiNode = promptApiData[String(nodeId)];
        title = apiNode._meta?.title || apiNode.class_type || `Node #${nodeId}`;
    } else {
        title = `Node #${nodeId}`;
    }
    return { title, pathStr };
}

function syncProgressBarDOM(explicitEnabled) {
    const isEnabled = isProgressBarEnabled(explicitEnabled);

    if (!isEnabled) {
        if (progressBarInstance) {
            progressBarInstance.style.display = "none";
            progressBarInstance.hidden = true;
            progressBarInstance.remove();
        }
        document.querySelectorAll(XENodesProgressBarElement.TAG).forEach(el => {
            el.style.display = "none";
            el.hidden = true;
            el.remove();
        });
        return;
    }

    if (!progressBarInstance) {
        progressBarInstance = document.createElement(XENodesProgressBarElement.TAG);
    }
    progressBarInstance.style.display = "block";
    progressBarInstance.hidden = false;

    const topSlot = document.querySelector(".comfyui-body-top");
    if (topSlot) {
        if (progressBarInstance.parentNode !== topSlot) topSlot.prepend(progressBarInstance);
    } else {
        progressBarInstance.style.position = "fixed";
        progressBarInstance.style.top = "0";
        progressBarInstance.style.left = "0";
        progressBarInstance.style.width = "100%";
        if (progressBarInstance.parentNode !== document.body) document.body.prepend(progressBarInstance);
    }
}

// ============================================================================
// 5. Register Extension with ComfyUI
// ============================================================================

app.registerExtension({
    name: "XENodes.ProgressBar",

    settings: [
        {
            id: "XENodes.ProgressBar.Enabled",
            category: ["XENodes", "Progress Bar"],
            name: "Display Progress Bar",
            type: "boolean",
            defaultValue: true,
            onChange: (newValue) => syncProgressBarDOM(newValue)
        }
    ],

    async setup() {
        app.ui?.settings?.addSetting({
            id: "XENodes.ProgressBar.Enabled",
            category: ["XENodes", "Progress Bar"],
            name: "Display Progress Bar",
            type: "boolean",
            defaultValue: true,
            onChange: (newValue) => syncProgressBarDOM(newValue)
        });

        setupPulseDrawHook();
        syncProgressBarDOM();

        // Hook queuePrompt to capture per-job prompt structure and tab name
        const originalQueuePrompt = api.queuePrompt;
        if (typeof originalQueuePrompt === "function") {
            api.queuePrompt = async function (num, prompt, ...args) {
                const currentTab = getActiveWorkflowTabName();
                const response = await originalQueuePrompt.apply(api, [num, prompt, ...args]);
                if (response?.prompt_id) {
                    const record = getOrInitPrompt(response.prompt_id);
                    if (currentTab) record.tabName = currentTab;
                    if (prompt?.output) {
                        record.promptApiData = prompt.output;
                        record.totalNodes = Object.keys(prompt.output).length;
                    }
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
            const queueRemaining = e.detail?.exec_info?.queue_remaining ?? 0;
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
            const pid = e.detail?.prompt_id;
            currentPromptId = pid;
            currentExecutingNodeId = null;

            const record = getOrInitPrompt(pid);
            const currentTab = getActiveWorkflowTabName();
            if (currentTab && !record.tabName) record.tabName = currentTab;

            if (progressBarInstance) {
                progressBarInstance.updateState({
                    currentNodeId: null,
                    totalNodes: record.totalNodes || 0,
                    executedNodesCount: 0,
                    step: 0,
                    maxSteps: 0,
                    nodeTitle: "",
                    hierarchyPath: "",
                    isError: false,
                    isExecuting: true,
                    tabName: record.tabName
                });
            }
        });

        api.addEventListener("executing", (e) => {
            if (!isProgressBarEnabled()) {
                syncProgressBarDOM(false);
                return;
            }

            let nodeId = null;
            if (e.detail != null) {
                nodeId = typeof e.detail === "object" ? (e.detail.node || e.detail.display_node || null) : e.detail;
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

            const promptState = currentPromptId ? getOrInitPrompt(currentPromptId) : null;
            if (promptState && currentExecutingNodeId && currentExecutingNodeId !== nodeId) {
                promptState.executedNodeIds.add(currentExecutingNodeId);
            }

            currentExecutingNodeId = nodeId;
            const info = resolveNodeDisplayInfo(nodeId, promptState?.promptApiData);

            if (progressBarInstance) {
                progressBarInstance.updateState({
                    currentNodeId: nodeId,
                    totalNodes: promptState?.totalNodes,
                    executedNodesCount: promptState?.executedNodeIds?.size,
                    step: 0,
                    maxSteps: 0,
                    nodeTitle: info.title,
                    hierarchyPath: info.pathStr,
                    isError: false,
                    isExecuting: true,
                    tabName: promptState?.tabName
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

            const promptState = currentPromptId ? getOrInitPrompt(currentPromptId) : null;
            if (nodeId && nodeId !== currentExecutingNodeId) {
                if (currentExecutingNodeId && promptState) {
                    promptState.executedNodeIds.add(currentExecutingNodeId);
                }
                currentExecutingNodeId = nodeId;
            }

            const info = resolveNodeDisplayInfo(currentExecutingNodeId, promptState?.promptApiData);

            progressBarInstance.updateState({
                currentNodeId: currentExecutingNodeId,
                totalNodes: promptState?.totalNodes,
                executedNodesCount: promptState?.executedNodeIds?.size,
                step,
                maxSteps,
                nodeTitle: info.title,
                hierarchyPath: info.pathStr,
                isExecuting: true,
                tabName: promptState?.tabName
            });
        });

        api.addEventListener("execution_cached", (e) => {
            if (!isProgressBarEnabled()) return;
            const promptState = currentPromptId ? getOrInitPrompt(currentPromptId) : null;
            if (promptState && Array.isArray(e.detail?.nodes)) {
                for (const c of e.detail.nodes) promptState.executedNodeIds.add(String(c));
                if (progressBarInstance) {
                    progressBarInstance.updateState({
                        executedNodesCount: promptState.executedNodeIds.size
                    });
                }
            }
        });

        api.addEventListener("execution_error", (e) => {
            if (!isProgressBarEnabled()) return;
            if (progressBarInstance && e.detail) {
                const errNodeId = e.detail.node_id;
                const errMsg = e.detail.exception_message || e.detail.exception_type || "Error occurred";
                currentExecutingNodeId = errNodeId;
                progressBarInstance.updateState({
                    currentNodeId: errNodeId,
                    isError: true,
                    errorMessage: `${errMsg} (Node #${errNodeId})`,
                    isExecuting: false
                });
            }
        });
    }
});
