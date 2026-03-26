import { app } from "../../../scripts/app.js";
import { applyTextReplacements } from "../../../scripts/utils.js";

app.registerExtension({
	name: "XENodes.SaveImage",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "XENodes.SaveImage") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const prefixWidget = this.widgets.find((w) => w.name === "filename_prefix");
				const formatWidget = this.widgets.find((w) => w.name === "format");
				const losslessWidget = this.widgets.find((w) => w.name === "lossless");
				const qualityWidget = this.widgets.find((w) => w.name === "quality");

				// Apply text replacements for filename_prefix (same as Save Video)
				if (prefixWidget) {
					prefixWidget.serializeValue = () => {
						return applyTextReplacements(app, prefixWidget.value);
					};
				}

				const updateWebpWidgets = (format, isInitial = false) => {
					if (!qualityWidget || !losslessWidget) return;

					// Helper to find current index
					const getWidgetIndex = (name) => this.widgets.findIndex(w => w.name === name);

					if (format === "png") {
						losslessWidget.hidden = true;
						qualityWidget.label = "compression";
						if (qualityWidget.options) {
							qualityWidget.options.min = 0;
							qualityWidget.options.max = 9;
							qualityWidget.options.step = 10;
						}
						// Manual change or switching from WebP: set to default 6
						if (!isInitial && (qualityWidget.value > 9 || formatWidget._lastValue === "webp")) {
							qualityWidget.value = 6;
						}
					} else {
						losslessWidget.hidden = false;
						qualityWidget.label = "quality";
						if (qualityWidget.options) {
							qualityWidget.options.min = 0;
							qualityWidget.options.max = 100;
							qualityWidget.options.step = 10;
						}
						// Manual change or switching from PNG: set to default 90
						if (!isInitial && (qualityWidget.value <= 9 || formatWidget._lastValue === "png")) {
							qualityWidget.value = 90;
							losslessWidget.value = false;
						}

						// Reorder: quality then lossless
						const qIdx = getWidgetIndex("quality");
						const lIdx = getWidgetIndex("lossless");
						if (qIdx !== -1 && lIdx !== -1 && lIdx < qIdx) {
							this.widgets.splice(lIdx, 1);
							const newQIdx = getWidgetIndex("quality");
							this.widgets.splice(newQIdx + 1, 0, losslessWidget);
						}
					}
					formatWidget._lastValue = format;
					this.graph?.setDirtyCanvas(true);
				};

				if (formatWidget) {
					formatWidget._lastValue = formatWidget.value;
					// Initial state (respect loaded values)
					updateWebpWidgets(formatWidget.value, true);

					const origCallback = formatWidget.callback;
					formatWidget.callback = function (v) {
						updateWebpWidgets(v, false);
						return origCallback ? origCallback.apply(this, arguments) : undefined;
					};
				}

				const onConfigure = nodeType.prototype.onConfigure;
				nodeType.prototype.onConfigure = function () {
					const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
					if (formatWidget) {
						updateWebpWidgets(formatWidget.value, true);
					}
					return r;
				};

				this.onResize = function (size) {
					const minSize = this.computeSize ? this.computeSize() : [200, 100];
					size[1] = minSize[1]; // Force height to be minimum
				};

				return r;
			};
		}
	},
});
