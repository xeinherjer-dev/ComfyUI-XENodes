/**
 * [XENodes.NodeExecutionTime]
 * - Measures and records execution duration for all nodes in the workflow.
 * - Dynamically sums up and draws aggregated execution time badges on parent subgraph nodes.
 * - Works standalone or alongside other extensions (like EasyUse), automatically preventing double counting.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[XENodes.NodeExecutionTime] Script loaded and initializing...");

/**
 * Safely retrieves the list of inner nodes from a subgraph parent node (e.g., GroupNode)
 * without raising TypeErrors when running inside different ComfyUI versions.
 * @param {LGraphNode} node - The parent node.
 * @returns {LGraphNode[]} An array of inner nodes.
 */
function getInnerNodesSafely(node) {
    if (!node) return [];
    
    // 1. Check resolved inner nodes cache if available
    if (Array.isArray(node.innerNodes)) {
        return node.innerNodes;
    }
    
    // 2. Loop through symbol properties to retrieve from the handler (GroupNodeHandler)
    const symbols = Object.getOwnPropertySymbols(node);
    for (const sym of symbols) {
        const val = node[sym];
        if (val && typeof val === "object" && Array.isArray(val.innerNodes)) {
            return val.innerNodes;
        }
    }
    
    // 3. Fallback to calling getInnerNodes method with a map as required by SubgraphNode.ts
    if (typeof node.getInnerNodes === "function") {
        try {
            const map = new Map();
            const res = node.getInnerNodes(map, [], [], new Set());
            if (Array.isArray(res)) {
                return res.map(item => item && item.node ? item.node : item);
            }
        } catch (e) {
            try {
                const map = new Map();
                const res = node.getInnerNodes(map);
                if (Array.isArray(res)) {
                    return res.map(item => item && item.node ? item.node : item);
                }
            } catch (e2) {}
        }
    }
    return [];
}

/**
 * Recursively resolves a node within standard LiteGraph subgraphs and ComfyUI GroupNodes,
 * supporting dotted/composite IDs (e.g. "33.0" or "33:0").
 * @param {LGraph} graph - The parent graph context.
 * @param {string|number} id - Target node ID.
 * @returns {LGraphNode|null}
 */
function findNodeRecursive(graph, id) {
    if (!graph || id == null) return null;
    
    const idStr = String(id);
    
    // 1. Direct match check
    let node = graph.getNodeById(idStr) || graph.getNodeById(Number(id));
    if (node) return node;
 
    // 2. Composite ID parsing (e.g. "33.0" or "33:0")
    const separators = [".", ":"];
    for (const sep of separators) {
        if (idStr.includes(sep)) {
            const parts = idStr.split(sep);
            let currentGraph = graph;
            let currentNode = null;
            
            for (let i = 0; i < parts.length; i++) {
                const part = parts[i];
                if (!currentGraph) break;
                
                currentNode = currentGraph.getNodeById(part) || currentGraph.getNodeById(Number(part));
                if (currentNode) {
                    if (currentNode.subgraph) {
                        currentGraph = currentNode.subgraph;
                    } else if (typeof currentNode.getInnerNodes === "function") {
                        try {
                            const innerNodes = getInnerNodesSafely(currentNode);
                            const nextPart = parts[i + 1];
                            if (nextPart !== undefined && Array.isArray(innerNodes)) {
                                const foundInner = innerNodes.find(itemNode => {
                                    if (!itemNode) return false;
                                    const itemId = String(itemNode.id);
                                    return itemId.endsWith(":" + nextPart) || itemId === nextPart || itemNode.index === Number(nextPart);
                                });
                                if (foundInner) {
                                    currentNode = foundInner;
                                    currentGraph = null;
                                    i++;
                                }
                            }
                        } catch (e) {}
                    } else {
                        currentGraph = null;
                    }
                } else {
                    currentNode = null;
                    break;
                }
            }
            if (currentNode) return currentNode;
        }
    }
 
    // 3. Recursive fallback search
    if (graph._nodes) {
        for (const n of graph._nodes) {
            // A. Standard LiteGraph Subgraph
            if (n.subgraph) {
                node = findNodeRecursive(n.subgraph, id);
                if (node) return node;
            }
            // B. ComfyUI GroupNode / SubgraphNode
            else if (typeof n.getInnerNodes === "function") {
                try {
                    const innerNodes = getInnerNodesSafely(n);
                    if (Array.isArray(innerNodes)) {
                        const found = innerNodes.find(itemNode => {
                            if (!itemNode) return false;
                            const itemId = String(itemNode.id);
                            return itemId === idStr || itemId.split(":").at(-1) === idStr;
                        });
                        if (found) return found;
                    }
                } catch (e) {}
            }
        }
    }
     
    return null;
}

/**
 * Calculates the total execution time of a node and its nested children.
 * For subgraph parent nodes (GroupNode or standard subgraphs), it sums up only the 
 * inner nodes' durations and ignores the duration directly recorded on the parent 
 * itself to prevent drift from transition overheads.
 * @param {LGraphNode} node - The node to measure.
 * @param {boolean} [isRoot=true] - Whether this is the entry node of summation.
 * @returns {number} Summed duration in seconds.
 */
function getNodeExecutionDuration(node, isRoot = true) {
    const isParent = typeof node.getInnerNodes === "function" || !!node.subgraph;
    
    if (isParent) {
        let duration = 0;
        // A. ComfyUI GroupNode / SubgraphNode inner nodes summation
        if (typeof node.getInnerNodes === "function") {
            try {
                const innerNodes = getInnerNodesSafely(node);
                if (Array.isArray(innerNodes)) {
                    for (const innerNode of innerNodes) {
                        if (innerNode) {
                            duration += getNodeExecutionDuration(innerNode, false);
                        }
                    }
                }
            } catch (e) {}
        }
        // B. Standard LiteGraph subgraph summation
        else if (node.subgraph && node.subgraph._nodes) {
            for (const child of node.subgraph._nodes) {
                duration += getNodeExecutionDuration(child, false);
            }
        }
        return duration;
    }
    
    // For atomic leaf nodes, return their own recorded duration
    return node.executionDuration || 0;
}

/**
 * Safely adds a dynamic execution time badge getter to the node's badges array.
 * @param {LGraphNode} node - The LiteGraph node.
 */
function addTimeBadgeToNode(node) {
    if (!node) return;
    
    if (!node.badges) {
        node.badges = [];
    }
    
    // Prevent duplicate registration of the time badge
    const hasTimeBadge = node.badges.some(b => b && b.isXENodesTimeBadge);
    if (!hasTimeBadge) {
        const timeBadgeGetter = () => {
            // Try to retrieve standard LGraphBadge class (might be global or on LiteGraph)
            const BadgeClass = globalThis.LGraphBadge || (globalThis.LiteGraph && globalThis.LiteGraph.LGraphBadge);
            if (!BadgeClass) return null;

            const isEnabled = app.ui.settings.getSettingValue("XENodes.NodeExecutionTime.Enabled") !== false;
            const duration = getNodeExecutionDuration(node);

            // Set text to empty string when disabled or has no duration.
            // LiteGraph and ComfyUI automatically handle empty text badges as invisible (visible = false).
            // Returning null here causes Uncaught TypeError: Cannot read properties of null in LGraphNode.ts:drawBadges.
            const text = (isEnabled && duration > 0)
                ? (duration < 1 ? `${Math.round(duration * 1000)}ms` : `${duration.toFixed(2)}s`)
                : "";

            const badge = new BadgeClass({
                text: text,
                fgColor: "rgba(148, 163, 184, 0.85)", // Subtle Slate 400 gray for non-intrusive metadata display
                bgColor: "rgba(15, 23, 42, 0.85)",
                fontSize: 12,
                padding: 6,
                height: 20,
                cornerRadius: 5
            });
            
            // Identify this badge specifically as our time badge
            badge.isXENodesTimeBadge = true;
            return badge;
        };
        
        // Mark the getter function itself as well
        timeBadgeGetter.isXENodesTimeBadge = true;
        node.badges.push(timeBadgeGetter);
    }
}

app.registerExtension({
    name: "XENodes.NodeExecutionTime",
    nodeCreated(node) {
        addTimeBadgeToNode(node);
    },
    setup() {
        // Register the setting in ComfyUI Settings menu
        app.ui.settings.addSetting({
            id: "XENodes.NodeExecutionTime.Enabled",
            category: ["XENodes", "Node Execution Time"],
            name: "Display Execution Time Badge",
            type: "boolean",
            defaultValue: true,
        });

        let lastNodeId = null;
        let startTime = 0;

        console.log("[XENodes.NodeExecutionTime] Extension registered successfully. Hooking up API events.");

        // Clear execution durations recursively when a new run starts
        api.addEventListener("execution_start", () => {
            const clearRecursive = (graph) => {
                if (!graph) return;
                (graph._nodes || []).forEach(node => {
                    delete node.executionDuration;
                    if (node.subgraph) {
                        clearRecursive(node.subgraph);
                    } else if (typeof node.getInnerNodes === "function") {
                        try {
                            const innerNodes = getInnerNodesSafely(node);
                            if (Array.isArray(innerNodes)) {
                                innerNodes.forEach(itemNode => {
                                    if (itemNode) delete itemNode.executionDuration;
                                });
                            }
                        } catch (e) {}
                    }
                });
            };
            clearRecursive(app.graph);
            lastNodeId = null;
            startTime = 0;
        });

        // Track executed node durations
        api.addEventListener("executing", (event) => {
            // Guard: Skip measurement if the feature is disabled in settings
            const isEnabled = app.ui.settings.getSettingValue("XENodes.NodeExecutionTime.Enabled") !== false;
            if (!isEnabled) {
                lastNodeId = null;
                startTime = 0;
                return;
            }

            let currentNodeId = null;
            if (event.detail !== null && event.detail !== undefined) {
                if (typeof event.detail === "object") {
                    currentNodeId = event.detail.node || event.detail.display_node || null;
                } else if (typeof event.detail === "string" || typeof event.detail === "number") {
                    currentNodeId = event.detail;
                }
            }
            if (currentNodeId === null) {
                currentNodeId = event.node || null;
            }

            const now = performance.now();

            if (lastNodeId !== null && startTime > 0) {
                const elapsed = (now - startTime) / 1000;
                
                const prevNode = findNodeRecursive(app.graph, lastNodeId);
                if (prevNode) {
                    // Guard: If another extension (like comfyui-easy-use) has already measured 
                    // and set a duration, skip our manual calculation to prevent double counting.
                    if (!prevNode.executionDuration) {
                        const oldDuration = prevNode.executionDuration || 0;
                        prevNode.executionDuration = oldDuration + elapsed;
                        
                        // Trigger canvas redraw to update the dynamic badges real-time
                        prevNode.setDirtyCanvas(true, true);
                    }
                }
            }

            lastNodeId = currentNodeId;
            startTime = now;
        });

        // Add badges to existing nodes already loaded in the graph
        if (app.graph && app.graph._nodes) {
            app.graph._nodes.forEach(node => {
                addTimeBadgeToNode(node);
            });
        }
    }
});
