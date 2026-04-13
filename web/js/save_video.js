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
				const audioCodecWidget = this.widgets.find((w) => w.name === "audio_codec");
				const audioBitrateWidget = this.widgets.find((w) => w.name === "audio_bitrate");

				if (prefixWidget) {
					prefixWidget.serializeValue = () => {
						return applyTextReplacements(app, prefixWidget.value);
					};
				}

				const CODEC_CRF_MAP = {
					'h264': 23,
					'h265': 28,
					'av1': 42
				};

				const updateCrf = () => {
					const codec = codecWidget?.value;
					const crfValue = CODEC_CRF_MAP[codec];
					if (crfValue !== undefined && crfWidget) {
						crfWidget.value = crfValue;
					}
				};

				if (formatWidget && codecWidget && crfWidget) {
					const originalCodecOptions = [...codecWidget.options.values];

					const updateCodecs = () => {
						if (formatWidget.value === "webm") {
							codecWidget.options.values = ["av1"];
							if (codecWidget.value !== "av1") {
								codecWidget.value = "av1";
								if (codecWidget.callback) codecWidget.callback("av1");
							}
						} else {
							codecWidget.options.values = originalCodecOptions;
						}
					};

					const origFormatCallback = formatWidget.callback;
					formatWidget.callback = function (v) {
						const res = origFormatCallback ? origFormatCallback.apply(this, arguments) : undefined;
						updateCodecs();
						updateCrf();
						return res;
					};

					const origCodecCallback = codecWidget.callback;
					codecWidget.callback = function (v) {
						updateCrf();
						return origCodecCallback ? origCodecCallback.apply(this, arguments) : undefined;
					};

					// Initial sync
					updateCodecs();
					updateCrf();
				}

				if (audioCodecWidget && audioBitrateWidget) {
					const updateAudioBitrateVisibility = () => {
						if (audioCodecWidget.value === "flac") {
							// Hide bitrate widget for flac
							audioBitrateWidget.hidden = true;
							audioBitrateWidget.disabled = true;
						} else {
							// Restore it
							audioBitrateWidget.hidden = false;
							audioBitrateWidget.disabled = false;
						}
					};

					const origAudioCodecCallback = audioCodecWidget.callback;
					audioCodecWidget.callback = function (v) {
						updateAudioBitrateVisibility();
                        
						// Trigger layout reflow
						if (this.setDirtyCanvas) {
							requestAnimationFrame(() => {
								this.setDirtyCanvas(true, true);
							});
						}

						return origAudioCodecCallback ? origAudioCodecCallback.apply(this, arguments) : undefined;
					}.bind(this);

					// Initial sync
					updateAudioBitrateVisibility();
				}

				// Intercept DOM widget creation to prevent video preview from forcing large node size
				const origAddDOMWidget = this.addDOMWidget;
				if (origAddDOMWidget) {
					this.addDOMWidget = function(name, type, element, options) {
						const widget = origAddDOMWidget.apply(this, arguments);
						if (name === "video-preview") {
							widget.computeLayoutSize = () => ({
								minWidth: 150,
								minHeight: 150
							});
							// Ensure container takes full height of the widget slot
							if (element && element.style) {
								element.style.height = "100%";
								element.style.display = "flex";
								element.style.justifyContent = "center";
								element.style.alignItems = "center";
							}
						}
						return widget;
					};
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
