import { app } from "../../../scripts/app.js";
import { applyTextReplacements } from "../../../scripts/utils.js";

app.registerExtension({
	name: "XENodes.SaveAudio",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "XENodes.SaveAudio") {
			nodeData.input ??= {};
			nodeData.input.required ??= {};
			if (!nodeData.input.required.audioUI) {
				nodeData.input.required.audioUI = ["AUDIO_UI", {}];
			}

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				this.previewMediaType = "audio";

				const codecWidget = this.widgets.find((w) => w.name === "audio_codec");
				const bitrateWidget = this.widgets.find((w) => w.name === "audio_bitrate");
				const prefixWidget = this.widgets.find((w) => w.name === "filename_prefix");

				const syncTasks = [];

				if (prefixWidget) {
					prefixWidget.serializeValue = () => {
						return applyTextReplacements(app, prefixWidget.value);
					};
				}

				if (codecWidget && bitrateWidget) {
					const CODEC_CONFIG = {
						"mp3": { supports_bitrate: true, bitrates: ["V0", "64k", "128k", "192k", "256k", "320k"], default_bitrate: "V0" },
						"opus": { supports_bitrate: true, bitrates: ["64k", "128k", "192k", "256k", "320k"], default_bitrate: "128k" },
						"flac": { supports_bitrate: false, bitrates: [], default_bitrate: null }
					};

					// Save Original Layout
					const origBitrateType = bitrateWidget.type;
					const origBitrateComputeSize = bitrateWidget.computeSize;

					const updateBitrateVisibility = () => {
						const config = CODEC_CONFIG[codecWidget.value] || CODEC_CONFIG["mp3"];

						if (!config.supports_bitrate) {
							bitrateWidget.hidden = true;
							bitrateWidget.disabled = true;
						} else {
							bitrateWidget.hidden = false;
							bitrateWidget.disabled = false;
							bitrateWidget.options.values = config.bitrates;

							if (!config.bitrates.includes(bitrateWidget.value)) {
								bitrateWidget.value = config.default_bitrate;
								if (bitrateWidget.callback) bitrateWidget.callback(bitrateWidget.value);
							}
						}
					};
					syncTasks.push(updateBitrateVisibility);

					const origCodecCallback = codecWidget.callback;
					codecWidget.callback = function (v) {
						updateBitrateVisibility();
                        
						// Trigger layout reflow
						if (this.setDirtyCanvas) {
							requestAnimationFrame(() => {
								this.setDirtyCanvas(true, true);
							});
						}

						return origCodecCallback ? origCodecCallback.apply(this, arguments) : undefined;
					}.bind(this);

					// Initial sync
				}

				this.onResize = function (size) {
					const minSize = this.computeSize ? this.computeSize() : [150, 80];
					size[0] = Math.max(size[0], minSize[0]);
					size[1] = Math.max(size[1], minSize[1]);
				};

				const origOnConfigure = this.onConfigure;
				this.onConfigure = function() {
					const res = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
					for (const task of syncTasks) {
						task();
					}
					return res;
				};

				return r;
			};
		}
	},
});
