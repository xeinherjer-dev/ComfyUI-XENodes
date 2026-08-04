import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "XENodes.SaveVideo",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "XENodes.SaveVideo") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				// Intercept DOM widget creation to prevent video preview from forcing large node size
				const origAddDOMWidget = this.addDOMWidget;
				if (origAddDOMWidget) {
					this.addDOMWidget = function(name, type, element, options) {
						const widget = origAddDOMWidget.apply(this, arguments);
						if (name === "video-preview") {
							// Permanently fix computeLayoutSize to return small minimum dimensions.
							Object.defineProperty(widget, 'computeLayoutSize', {
								configurable: true,
								get() {
									return () => ({ minWidth: 50, minHeight: 200 });
								},
								set(_fn) {
									// Intentionally ignore attempts to override
								}
							});
						}
						return widget;
					};
				}

				return r;
			};
		}
	},
});
