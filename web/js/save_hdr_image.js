import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "XENodes.SaveHDRImage",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "XENodes.SaveHDRImage") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const codecWidget = this.widgets.find((w) => w.name === "codec");
				const crfWidget = this.widgets.find((w) => w.name === "crf");
				const prefixWidget = this.widgets.find((w) => w.name === "filename_prefix");


				const CODEC_CRF_MAP = {
					'av1': 2,
					'av1_nvenc': 2 
				};

				const updateCrf = () => {
					const codec = codecWidget?.value;
					const crfValue = CODEC_CRF_MAP[codec];
					if (crfValue !== undefined && crfWidget) {
						crfWidget.value = crfValue;
					}
				};

				if (codecWidget && crfWidget) {
					const origCodecCallback = codecWidget.callback;
					codecWidget.callback = function (v) {
						updateCrf();
						return origCodecCallback ? origCodecCallback.apply(this, arguments) : undefined;
					};

					// Initial sync
					updateCrf();
				}

				return r;
			};
		}
	},
});
