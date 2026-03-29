import { app } from "../../../scripts/app.js";
import { applyTextReplacements } from "../../../scripts/utils.js";

app.registerExtension({
	name: "XENodes.SaveVideo",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "XENodes.SaveVideo") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const formatWidget = this.widgets.find((w) => w.name === "format");
				const codecWidget = this.widgets.find((w) => w.name === "codec");
				const crfWidget = this.widgets.find((w) => w.name === "crf");
				const prefixWidget = this.widgets.find((w) => w.name === "filename_prefix");

				if (prefixWidget) {
					prefixWidget.serializeValue = () => {
						return applyTextReplacements(app, prefixWidget.value);
					};
				}

				const CODEC_CRF_MAP = {
					'h264': 23,
					'h265': 28,
					'av1': 46
				};

				const updateCrf = () => {
					const codec = codecWidget?.value;
					const crfValue = CODEC_CRF_MAP[codec];
					if (crfValue !== undefined && crfWidget) {
						crfWidget.value = crfValue;
					}
				};

				if (formatWidget && codecWidget && crfWidget) {
					const origFormatCallback = formatWidget.callback;
					formatWidget.callback = function (v) {
						if (v === "webm") {
							codecWidget.value = "av1";
							if (codecWidget.callback) {
								codecWidget.callback("av1");
							}
						}
						return origFormatCallback ? origFormatCallback.apply(this, arguments) : undefined;
					};

					const origCodecCallback = codecWidget.callback;
					codecWidget.callback = function (v) {
						if (formatWidget.value === "webm" && v !== "av1") {
							// Force AV1 for WebM if they try to change it
							setTimeout(() => { codecWidget.value = "av1"; }, 1);
						} else {
							updateCrf();
						}
						return origCodecCallback ? origCodecCallback.apply(this, arguments) : undefined;
					};

					// Initial sync
					updateCrf();
				}

				this.onResize = function (size) {
					const minSize = this.computeSize ? this.computeSize() : [200, 100];
					size[0] = Math.max(size[0], minSize[0]);
					size[1] = Math.max(size[1], minSize[1]);
				};

				return r;
			};
		}
	},
});
