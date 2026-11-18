/**
 * [XENodes.ProgressBar]
 * - Subgraph-aware Progress Bar & Node Navigator for ComfyUI.
 * - Deeply tracks executing nodes across complex subgraph hierarchies (Nodes 2.0 Subgraphs & GroupNodes).
 * - Multi-Workflow Tab Aware: Automatically switches to the executing workflow tab if another tab is focused.
 * - Multi-Queue Robust: Isolates job progress per prompt_id so progress never resets when queuing new runs.
 * - Click to smoothly jump/drill-down into executing subgraphs and focus on the active node.
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
// 2. Multi-Tab Navigation & Node Focus Logic
// ============================================================================

function isRoleTabElement(el) {
    return !!(el && (el.matches?.("[role='tab']") || el.getAttribute?.("role") === "tab"));
}

function simulateFullClick(el) {
    if (!el) return;
    const opts = { bubbles: true, cancelable: true, view: window, button: 0, buttons: 1, ctrlKey: false };
    if (typeof PointerEvent === "function") el.dispatchEvent(new PointerEvent("pointerdown", opts));
    if (typeof MouseEvent === "function") el.dispatchEvent(new MouseEvent("mousedown", opts));
    if (typeof PointerEvent === "function") el.dispatchEvent(new PointerEvent("pointerup", opts));
    if (typeof MouseEvent === "function") el.dispatchEvent(new MouseEvent("mouseup", opts));

    const isRekaTabTrigger = isRoleTabElement(el) && !el.closest?.(".p-togglebutton");
    if (!isRekaTabTrigger) {
        if (typeof MouseEvent === "function") el.dispatchEvent(new MouseEvent("click", opts));
        if (typeof el.click === "function") el.click();
    }
}

function getWorkflowTabButtons() {
    const primary = Array.from(document.querySelectorAll(
        ".workflow-tabs [role='tab'], .workflow-tabs-container [role='tab'], [data-testid='workflow-tab-strip'] [role='tab']"
    ));
    const raw = primary.length > 0
        ? primary
        : Array.from(document.querySelectorAll(
            ".workflow-tabs .p-togglebutton, .workflow-tabs-container .p-togglebutton, .workflow-tab, [role='tablist'] [role='tab'], .workflow-tabs-list .workflow-tab-item"
        ));

    const unique = new Set();
    for (const el of raw) {
        if (!el) continue;
        const normalized = isRoleTabElement(el) ? el : (el.querySelector?.("[role='tab']") || el);
        unique.add(normalized);
    }
    return Array.from(unique);
}

function isTabActive(el) {
    if (!el) return false;
    const roleTab = isRoleTabElement(el) ? el : (el.querySelector?.("[role='tab']") || el.closest?.("[role='tab']"));
    const container = el.closest?.(".workflow-tab, .p-togglebutton") || el;
    const targets = [roleTab, container, el].filter(Boolean);
    return targets.some(t => (
        t.getAttribute?.("aria-selected") === "true" ||
        t.getAttribute?.("data-state") === "active" ||
        t.getAttribute?.("data-p-highlight") === "true" ||
        t.classList?.contains?.("p-togglebutton-checked") ||
        t.classList?.contains?.("active")
    ));
}

function hasExecutingSpinner(el) {
    if (!el) return false;
    const container = el.closest?.(".workflow-tab, .p-togglebutton") || el;
    return !!container.querySelector?.(
        ".animate-spin, [class*='animate-spin'], [class*='loader'], [class*='spin'], .pi-spin, .pi-spinner, [data-icon*='spin'], [role='status']"
    );
}

function getTabLabelText(el) {
    if (!el) return null;
    const container = el.closest?.(".workflow-tab, .p-togglebutton") || el;
    const labelEl = el.querySelector?.(".workflow-label")
        || container.querySelector?.(".workflow-label")
        || el;
    return labelEl?.textContent ? labelEl.textContent.trim() : null;
}

function getActiveWorkflowTabName() {
    const storeFilename = app.extensionManager?.workflow?.activeWorkflow?.filename;
    if (typeof storeFilename === "string" && storeFilename.trim()) {
        return storeFilename.trim();
    }
    const activeLabel = document.querySelector(".workflow-tabs [role='tab'][aria-selected='true'] .workflow-label")
        || document.querySelector(".workflow-tabs [data-state='active'] .workflow-label")
        || document.querySelector("[data-testid='workflow-tab-strip'] [role='tab'][aria-selected='true'] .workflow-label")
        || document.querySelector(".workflow-tabs .p-togglebutton-checked .workflow-label")
        || document.querySelector(".workflow-tab.active .workflow-label")
        || document.querySelector(".p-togglebutton-checked .workflow-label");
    return activeLabel?.textContent ? activeLabel.textContent.trim() : null;
}

function isCurrentlyOnExecutingTab(targetTabName) {
    const tabButtons = getWorkflowTabButtons();
    if (tabButtons.length === 0) return true;

    const activeTab = tabButtons.find(isTabActive);
    if (activeTab && hasExecutingSpinner(activeTab)) {
        return true;
    }

    const otherRunningTab = tabButtons.find(el => !isTabActive(el) && hasExecutingSpinner(el));
    if (otherRunningTab) {
        return false;
    }

    if (targetTabName && activeTab) {
        const currentName = getTabLabelText(activeTab);
        if (currentName && currentName !== targetTabName) {
            return false;
        }
    }

    return true;
}

function switchTabToExecutingWorkflow(targetTabName) {
    const tabButtons = getWorkflowTabButtons();
    if (tabButtons.length === 0) return false;

    let targetTabBtn = null;

    // 1. Priority: Inactive tab with executing spinner
    for (const el of tabButtons) {
        if (isTabActive(el)) continue;
        if (hasExecutingSpinner(el)) {
            targetTabBtn = el;
            break;
        }
    }

    // 2. Match by target tab name
    if (!targetTabBtn && targetTabName) {
        for (const el of tabButtons) {
            if (isTabActive(el)) continue;
            if (getTabLabelText(el) === targetTabName) {
                targetTabBtn = el;
                break;
            }
        }
    }

    // 3. Fallback: solitary inactive tab
    if (!targetTabBtn) {
        const inactive = tabButtons.filter(el => !isTabActive(el));
        if (inactive.length === 1) targetTabBtn = inactive[0];
    }

    if (targetTabBtn) {
        const triggerEl = (isRoleTabElement(targetTabBtn) ? targetTabBtn : targetTabBtn.querySelector?.("[role='tab']")) || targetTabBtn;
        simulateFullClick(triggerEl);
        return true;
    }
    return false;
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

                // 1. If currently active tab is not the executing tab, switch first
                if (!isCurrentlyOnExecutingTab(this.currentExecutingTabName)) {
                    const prevGraph = app.rootGraph || app.graph;
                    const prevGraphState = prevGraph?.state;
                    const prevActivePath = app.extensionManager?.workflow?.activeWorkflow?.path;
                    const prevTabName = getActiveWorkflowTabName();

                    const switched = switchTabToExecutingWorkflow(this.currentExecutingTabName);
                    if (switched) {
                        const maxWait = 1500;
                        const start = performance.now();
                        while (performance.now() - start < maxWait) {
                            await new Promise(r => setTimeout(r, 30));
                            const curGraph = app.rootGraph || app.graph;
                            const curPath = app.extensionManager?.workflow?.activeWorkflow?.path;
                            const curTabName = getActiveWorkflowTabName();
                            const tabOrGraphChanged =
                                isCurrentlyOnExecutingTab(this.currentExecutingTabName) ||
                                curPath !== prevActivePath ||
                                curTabName !== prevTabName ||
                                curGraph !== prevGraph ||
                                (prevGraphState !== undefined && curGraph?.state !== prevGraphState);
                            if (tabOrGraphChanged && !app.configuringGraph) {
                                break;
                            }
                        }
                        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
                    }
                }

                let rootGraph = app.rootGraph || app.graph;
                let result = findNodeAndHierarchy(rootGraph, this.currentNodeId);

                // 2. Fallback if node is not found yet (wait for graph load / retry switch)
                if (!result || !result.node) {
                    const switchAttempted = switchTabToExecutingWorkflow(this.currentExecutingTabName);
                    const maxWait = switchAttempted ? 1000 : 400;
                    const start = performance.now();
                    while (performance.now() - start < maxWait) {
                        await new Promise(r => setTimeout(r, 50));
                        rootGraph = app.rootGraph || app.graph;
                        result = findNodeAndHierarchy(rootGraph, this.currentNodeId);
                        if (result?.node && !app.configuringGraph) break;
                    }
                    if (switchAttempted && result?.node) {
                        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
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

    completeExecution(queueRemaining = 0) {
        if (!this.shadowRoot) return;
        const barOverall = this.shadowRoot.querySelector(".bar-overall");
        const barStep = this.shadowRoot.querySelector(".bar-step");
        const textMain = this.shadowRoot.querySelector(".progress-text");
        if (!barOverall || !barStep || !textMain) return;

        barOverall.style.width = "100%";
        barStep.style.width = "100%";
        const queuePart = queueRemaining > 0 ? `(${queueRemaining}) ` : "";
        textMain.textContent = `${queuePart}100% - Done`;

        setTimeout(() => {
            if (!this.isExecuting) {
                this.updateView();
            }
        }, 500);
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

        // Overall progress percentage (including fractional step progress of currently executing node)
        let overallPercent = 0;
        let stepFraction = 0;
        if (this.maxSteps > 0 && this.currentStep != null) {
            stepFraction = Math.min(1, this.currentStep / this.maxSteps);
        }
        const effectiveCompleted = this.executedNodesCount + stepFraction;
        const total = Math.max(this.totalNodes || 1, Math.ceil(effectiveCompleted));

        if (total > 0) {
            overallPercent = Math.min(100, Math.round((effectiveCompleted / total) * 100));
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
        const styleEl = document.createElement("style");
        styleEl.textContent = `
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
        `;

        const container = document.createElement("div");
        container.className = "progress-container";
        container.title = "XENodes Progress Bar\rLeft-Click: Jump to active node\rRight-Click: Return to root graph";

        const barOverall = document.createElement("div");
        barOverall.className = "bar-overall";
        const barStep = document.createElement("div");
        barStep.className = "bar-step";
        const progressText = document.createElement("div");
        progressText.className = "progress-text";
        progressText.textContent = "Idle";

        container.appendChild(barOverall);
        container.appendChild(barStep);
        container.appendChild(progressText);

        if (typeof this.shadowRoot.replaceChildren === "function") {
            this.shadowRoot.replaceChildren(styleEl, container);
        } else {
            while (this.shadowRoot.firstChild) {
                this.shadowRoot.removeChild(this.shadowRoot.firstChild);
            }
            this.shadowRoot.appendChild(styleEl);
            this.shadowRoot.appendChild(container);
        }
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
let currentQueueRemaining = 0;
const promptsMap = new Map();
const pendingPromptQueue = [];

const NON_EXECUTABLE_NODE_TYPES = new Set([
    "Reroute",
    "Builtin: Reroute",
    "Note",
    "PrimitiveNode",
    "GraphInput",
    "GraphOutput",
    "SubgraphInput",
    "SubgraphOutput",
    "XENodes.FastGroupsMuterBypasser",
]);

/**
 * Checks whether a node represents an active, executable node.
 * @param {LGraphNode} node
 * @returns {boolean}
 */
function isExecutableNode(node) {
    if (!node || node.mode === 2 || node.mode === 4) return false;
    return !NON_EXECUTABLE_NODE_TYPES.has(node.type);
}

/**
 * Counts active executable nodes directly from the frontend LiteGraph hierarchy
 * (excluding muted/bypassed nodes and non-executable routing/UI nodes).
 * @param {LGraph|Subgraph} graph
 * @param {Set<any>} [visitedGraphs=new Set()]
 * @returns {number}
 */
function countExecutableNodesInGraph(graph, visitedGraphs = new Set()) {
    if (!graph || visitedGraphs.has(graph)) return 0;
    visitedGraphs.add(graph);

    const visitedNodes = new Set();
    const countNode = (node) => {
        if (!node || visitedNodes.has(node) || node.mode === 2 || node.mode === 4) return 0;
        visitedNodes.add(node);

        if (node.subgraph) {
            return countExecutableNodesInGraph(node.subgraph, visitedGraphs);
        }
        if (typeof node.getInnerNodes === "function" || Array.isArray(node.innerNodes)) {
            let innerCount = 0;
            for (const inner of getInnerNodesSafely(node)) {
                innerCount += countNode(inner);
            }
            return innerCount;
        }
        return isExecutableNode(node) ? 1 : 0;
    };

    let count = 0;
    const nodes = graph._nodes || graph.nodes || [];
    for (const node of nodes) {
        count += countNode(node);
    }
    return count;
}

const MAX_TRACKED_PROMPTS = 50;

function getOrInitPrompt(promptId) {
    if (!promptsMap.has(promptId)) {
        while (promptsMap.size >= MAX_TRACKED_PROMPTS) {
            const oldestKey = promptsMap.keys().next().value;
            if (oldestKey === undefined) break;
            promptsMap.delete(oldestKey);
        }
        promptsMap.set(promptId, {
            totalNodes: countExecutableNodesInGraph(app.rootGraph || app.graph),
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

function resolveNodeDisplayInfo(nodeId) {
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
        syncProgressBarDOM();

        // Connect API Listeners
        api.addEventListener("promptQueueing", () => {
            if (!isProgressBarEnabled()) return;
            while (pendingPromptQueue.length >= MAX_TRACKED_PROMPTS) {
                pendingPromptQueue.shift();
            }
            pendingPromptQueue.push({
                tabName: getActiveWorkflowTabName(),
                totalNodes: countExecutableNodesInGraph(app.rootGraph || app.graph)
            });
        });

        api.addEventListener("status", (e) => {
            if (!isProgressBarEnabled()) {
                syncProgressBarDOM(false);
                return;
            }
            const queueRemaining = e.detail?.exec_info?.queue_remaining ?? 0;
            currentQueueRemaining = queueRemaining;

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
            const workflowId = e.detail?.workflow_id;
            if (workflowId) {
                const openWorkflows = app.extensionManager?.workflow?.openWorkflows;
                if (Array.isArray(openWorkflows)) {
                    const matchedWorkflow = openWorkflows.find(
                        w => (w?.activeState?.id ?? w?.initialState?.id) === workflowId
                    );
                    if (matchedWorkflow?.filename) {
                        record.tabName = matchedWorkflow.filename;
                    }
                }
            }

            const queuedMeta = pendingPromptQueue.shift();
            if (queuedMeta) {
                if (!record.tabName && queuedMeta.tabName) {
                    record.tabName = queuedMeta.tabName;
                }
                if (queuedMeta.totalNodes > 0) {
                    record.totalNodes = queuedMeta.totalNodes;
                }
            }

            if (!record.tabName) {
                const currentTab = getActiveWorkflowTabName();
                if (currentTab) record.tabName = currentTab;
            }
            if (!record.totalNodes) {
                record.totalNodes = countExecutableNodesInGraph(app.rootGraph || app.graph);
            }

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
                    progressBarInstance.completeExecution(currentQueueRemaining);
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
            const info = resolveNodeDisplayInfo(nodeId);

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

            const info = resolveNodeDisplayInfo(currentExecutingNodeId);

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

        api.addEventListener("executed", (e) => {
            if (!isProgressBarEnabled()) return;
            const promptState = currentPromptId ? getOrInitPrompt(currentPromptId) : null;
            if (promptState && e.detail?.node) {
                promptState.executedNodeIds.add(String(e.detail.node));
                if (progressBarInstance) {
                    progressBarInstance.updateState({
                        executedNodesCount: promptState.executedNodeIds.size
                    });
                }
            }
        });

        api.addEventListener("execution_success", () => {
            if (!isProgressBarEnabled()) return;
            if (progressBarInstance) {
                progressBarInstance.completeExecution(currentQueueRemaining);
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
