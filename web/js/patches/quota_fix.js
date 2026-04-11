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
            // If another key triggers a quota error, log it specifically
            if (e.name === 'QuotaExceededError' || e.code === 22) {
                if (!reportedErrors.has(key)) {
                    console.warn(`[XENodes/QuotaFix] Storage full for key: ${key}. Write skipped to prevent crash. (Subsequent warnings suppressed)`);
                    reportedErrors.add(key);
                }
                return;
            }
            throw e;
        }
    };

    console.log("[XENodes/QuotaFix] Monkey Patch applied to localStorage.setItem. V1 drafts are now virtualized.");
})();
