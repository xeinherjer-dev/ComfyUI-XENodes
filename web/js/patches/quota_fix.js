/**
 * Monkey-patch: QuotaExceededError Fix Patch for ComfyUI
 * 
 * This monkey patch intercepts localStorage.setItem calls for the V1 draft storage keys.
 * By suppressing these high-volume writes, we prevent the 5MB localStorage quota 
 * from being exceeded, while allowing the more efficient V2 storage (per-workflow keys)
 * to continue functioning.
 */
(function() {
    const V1_DRAFT_KEY = 'Comfy.Workflow.Drafts';
    const V1_ORDER_KEY = 'Comfy.Workflow.DraftOrder';

    const originalSetItem = localStorage.setItem;
    let hasWarnedStorageFull = false;
    let hasLoggedV1Skip = false;

    function isQuotaExceededError(e) {
        return Boolean(
            e &&
            (e.name === 'QuotaExceededError' ||
             e.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
             e.code === 22 ||
             e.code === 1014 ||
             (typeof e.message === 'string' && e.message.toLowerCase().includes('quota')))
        );
    }

    localStorage.setItem = function(key, value) {
        // Intercept and ignore V1 storage keys
        if (key === V1_DRAFT_KEY || key === V1_ORDER_KEY || (key && key.startsWith(V1_DRAFT_KEY + ':'))) {
            // Silently skip the write to avoid QuotaExceededError
            // V2 persistence (Comfy.Workflow.DraftPayload:*) remains unaffected.
            if (!hasLoggedV1Skip) {
                console.log("[XENodes/QuotaFix] Skipping writes to V1 storage keys (subsequent notices suppressed).");
                hasLoggedV1Skip = true;
            }
            return;
        }

        try {
            return originalSetItem.apply(this, arguments);
        } catch (e) {
            if (isQuotaExceededError(e)) {
                // Emergency Garbage Collection to save the current data
                let freedSpace = false;

                // 1. Delete legacy 'workflow' key
                if (localStorage.getItem('workflow') !== null) {
                    localStorage.removeItem('workflow');
                    freedSpace = true;
                    console.log("[XENodes/QuotaFix] Deleted legacy 'workflow' key to free up space.");
                }

                // 2. Delete any lingering V1 draft keys
                const keysToRemove = [];
                for (let i = 0; i < localStorage.length; i++) {
                    const k = localStorage.key(i);
                    if (k === V1_DRAFT_KEY || k === V1_ORDER_KEY || (k && k.startsWith(V1_DRAFT_KEY + ':'))) {
                        keysToRemove.push(k);
                    }
                }

                // 3. Delete orphaned V2 draft keys
                const activeDrafts = new Set();
                for (let i = 0; i < localStorage.length; i++) {
                    const k = localStorage.key(i);
                    if (k && k.startsWith('Comfy.Workflow.DraftIndex.v2:')) {
                        const workspaceId = k.slice('Comfy.Workflow.DraftIndex.v2:'.length);
                        try {
                            const indexData = JSON.parse(localStorage.getItem(k) || '{}');
                            if (Array.isArray(indexData)) {
                                indexData.forEach(entry => {
                                    if (entry && entry.id) {
                                        activeDrafts.add(`Comfy.Workflow.Draft.v2:${entry.id}`);
                                    }
                                });
                            } else if (indexData && typeof indexData === 'object') {
                                const keys = Array.isArray(indexData.order)
                                    ? indexData.order
                                    : (indexData.entries ? Object.keys(indexData.entries) : []);
                                for (const draftKey of keys) {
                                    activeDrafts.add(`Comfy.Workflow.Draft.v2:${workspaceId}:${draftKey}`);
                                }
                            }
                        } catch(err) {}
                    }
                }

                for (let i = 0; i < localStorage.length; i++) {
                    const k = localStorage.key(i);
                    if (k && k.startsWith('Comfy.Workflow.Draft.v2:')) {
                        if (!activeDrafts.has(k)) {
                            keysToRemove.push(k);
                        }
                    }
                }

                if (keysToRemove.length > 0) {
                    keysToRemove.forEach(k => localStorage.removeItem(k));
                    freedSpace = true;
                    console.log(`[XENodes/QuotaFix] Deleted ${keysToRemove.length} orphaned/legacy draft key(s) to free up space.`);
                }

                // 4. Retry the save after first pass
                if (freedSpace) {
                    try {
                        return originalSetItem.apply(this, arguments);
                    } catch (retryError) {
                        // Failed again, fall through to extreme GC
                    }
                }

                // 5. EXTREME GC: Sacrifice valid V2 drafts one by one to make space for the CURRENT draft
                const allDraftKeys = [];
                for (let i = 0; i < localStorage.length; i++) {
                    const k = localStorage.key(i);
                    if (k && k.startsWith('Comfy.Workflow.Draft.v2:') && k !== key) {
                        allDraftKeys.push(k);
                    }
                }

                // Delete one by one and retry until it fits
                for (const k of allDraftKeys) {
                    localStorage.removeItem(k);
                    console.log(`[XENodes/QuotaFix] Extreme GC: Deleted valid draft ${k} to force space for new save.`);
                    try {
                        return originalSetItem.apply(this, arguments);
                    } catch (retryError2) {
                        // Still fails, continue loop to delete another one
                    }
                }

                // 6. If we deleted EVERYTHING and it STILL fails (e.g. the single draft is > 5MB)
                // Report warning only ONCE across the entire session to prevent console spam
                if (!hasWarnedStorageFull) {
                    console.warn(`[XENodes/QuotaFix] Storage full for key: ${key}. Write skipped to prevent crash. (Subsequent quota warnings suppressed)`);
                    hasWarnedStorageFull = true;
                }
                return;
            }
            throw e;
        }
    };

    console.log("[XENodes/QuotaFix] Monkey Patch applied to localStorage.setItem. V1 drafts are virtualized. Emergency GC enabled.");
})();
