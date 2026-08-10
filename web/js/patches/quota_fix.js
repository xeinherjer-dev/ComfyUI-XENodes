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
    const reportedErrors = new Set();

    localStorage.setItem = function(key, value) {
        // Intercept and ignore V1 storage keys
        if (key === V1_DRAFT_KEY || key === V1_ORDER_KEY || key.startsWith(V1_DRAFT_KEY + ':')) {
            // Silently skip the write to avoid QuotaExceededError
            // V2 persistence (Comfy.Workflow.DraftPayload:*) remains unaffected.
            console.log("[XENodes/QuotaFix] skip the write V1 storage keys.");
            return;
        }

        try {
            return originalSetItem.apply(this, arguments);
        } catch (e) {
            if (e.name === 'QuotaExceededError' || e.code === 22) {
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
                    if (k === V1_DRAFT_KEY || k === V1_ORDER_KEY || k && k.startsWith(V1_DRAFT_KEY + ':')) {
                        keysToRemove.push(k);
                    }
                }

                // 3. Delete orphaned V2 draft keys
                const activeDrafts = new Set();
                for (let i = 0; i < localStorage.length; i++) {
                    const k = localStorage.key(i);
                    if (k && k.startsWith('Comfy.Workflow.DraftIndex.v2:')) {
                        try {
                            const indexData = JSON.parse(localStorage.getItem(k) || '[]');
                            indexData.forEach(entry => {
                                if (entry && entry.id) {
                                    activeDrafts.add(`Comfy.Workflow.Draft.v2:${entry.id}`);
                                }
                            });
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
                // Use a generic prefix for reporting so we don't spam the console when the UUID changes
                const prefix = key.substring(0, key.lastIndexOf(':')) || key; 
                if (!reportedErrors.has(prefix)) {
                    console.warn(`[XENodes/QuotaFix] Storage full for key: ${key}. Write skipped to prevent crash. (Subsequent warnings for ${prefix} suppressed)`);
                    reportedErrors.add(prefix);
                }
                return;
            }
            throw e;
        }
    };

    console.log("[XENodes/QuotaFix] Monkey Patch applied to localStorage.setItem. V1 drafts are virtualized. Emergency GC enabled.");
})();
