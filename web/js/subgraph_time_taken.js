/**
 * [XENodes.SubgraphTimeTaken]
 * - Records execution times for nodes nested inside subgraphs.
 * - Sums up and draws the accumulated execution duration badge on parent subgraph nodes.
 * - Note: Displaying badges on individual nodes depends on comfyui-easy-use,
 *   but this script works standalone to record inner times and display parent total badges.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[XENodes.SubgraphTimeTaken] Script loaded and initializing...");

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
 * Renders a glassmorphism style execution time badge on the top of the node.
 * @param {CanvasRenderingContext2D} ctx - Canvas context.
 * @param {LGraphNode} node - Node on which to draw.
 * @param {string} text - Formatted time text.
 */
function drawTimeBadge(ctx, node, text) {
    ctx.save();
    
    ctx.font = "bold 10px Inter, system-ui, -apple-system, sans-serif";
    const textWidth = ctx.measureText(text).width;
    const badgeWidth = textWidth + 12;
    const badgeHeight = 16;
    
    const titleHeight = node.constructor.title_height || LiteGraph.NODE_TITLE_HEIGHT || 30;
    const x = 6;
    const y = -titleHeight - badgeHeight - 2;
    
    // Glassmorphism background: semi-transparent Slate 900
    ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
    ctx.beginPath();
    if (ctx.roundRect) {
        ctx.roundRect(x, y, badgeWidth, badgeHeight, 4);
    } else {
        ctx.rect(x, y, badgeWidth, badgeHeight);
    }
    ctx.fill();
    
    // Thin subtle border
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Text drawing: clean off-white
    ctx.fillStyle = "rgba(241, 245, 249, 0.95)";
    ctx.fillText(text, x + 6, y + 11);
    
    ctx.restore();
}

app.registerExtension({
    name: "XENodes.SubgraphTimeTaken",
    setup() {
        let lastNodeId = null;
        let startTime = 0;

        console.log("[XENodes.SubgraphTimeTaken] Extension registered successfully. Hooking up API events.");

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
                    // Only track and modify duration for nodes INSIDE subgraphs.
                    // Root-level nodes are already perfectly managed by EasyUse/ComfyUI core.
                    const isInnerNode = prevNode.graph && prevNode.graph !== app.graph;
                    
                    if (isInnerNode) {
                        // Guard: If another extension (like comfyui-easy-use) has already measured 
                        // and set a duration, skip our manual calculation to prevent double counting.
                        if (!prevNode.executionDuration) {
                            const oldDuration = prevNode.executionDuration || 0;
                            prevNode.executionDuration = oldDuration + elapsed;
                        }
                    }
                }
            }

            lastNodeId = currentNodeId;
            startTime = now;
        });

        // Patch LGraphNode globally to draw parent subgraph badges
        if (typeof LGraphNode !== "undefined" && LGraphNode.prototype) {
            const originalOnDrawForeground = LGraphNode.prototype.onDrawForeground;
            
            LGraphNode.prototype.onDrawForeground = function(ctx) {
                const result = originalOnDrawForeground ? originalOnDrawForeground.apply(this, arguments) : undefined;
                
                // Identify if this is a parent node of standard or custom subgraphs
                const isGroupNode = typeof this.getInnerNodes === "function" || 
                                    (this.type && this.type.startsWith("workflow>"));
                const isSubgraphParent = isGroupNode || !!this.subgraph;
                
                if (isSubgraphParent) {
                    const duration = getNodeExecutionDuration(this);
                    if (duration > 0) {
                        const formatted = duration < 1 
                            ? `${Math.round(duration * 1000)}ms` 
                            : `${duration.toFixed(2)}s`;
                        
                        drawTimeBadge(ctx, this, formatted);
                    }
                }
                
                return result;
            };
        } else {
            console.error("[XENodes.SubgraphTimeTaken] Failed to patch LGraphNode prototype: LGraphNode is undefined.");
        }
    }
});
