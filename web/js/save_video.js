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

				const syncTasks = [];

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
				syncTasks.push(updateCrf);

				if (formatWidget && codecWidget && crfWidget) {
					const originalCodecOptions = [...codecWidget.options.values];
					const originalAudioCodecOptions = audioCodecWidget ? [...audioCodecWidget.options.values] : [];

					// Defines restricted codec choices based on format
					const FORMAT_RESTRICTIONS = {
						'webm': {
							video: ['av1'],
							audio: ['opus']
						}
					};

					const updateCodecs = () => {
						const restrictions = FORMAT_RESTRICTIONS[formatWidget.value];

						if (restrictions) {
							if (restrictions.video) {
								codecWidget.options.values = restrictions.video;
								if (!restrictions.video.includes(codecWidget.value)) {
									const newCodec = restrictions.video[0];
									codecWidget.value = newCodec;
									if (codecWidget.callback) codecWidget.callback(newCodec);
								}
							}

							if (audioCodecWidget && restrictions.audio) {
								audioCodecWidget.options.values = restrictions.audio;
								if (!restrictions.audio.includes(audioCodecWidget.value)) {
									const newAudioCodec = restrictions.audio[0];
									audioCodecWidget.value = newAudioCodec;
									if (audioCodecWidget.callback) audioCodecWidget.callback(newAudioCodec);
								}
							}
						} else {
							codecWidget.options.values = originalCodecOptions;
							if (audioCodecWidget) {
								audioCodecWidget.options.values = originalAudioCodecOptions;
							}
						}
					};
					syncTasks.push(updateCodecs);

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
					syncTasks.push(updateAudioBitrateVisibility);

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
								minWidth: 50,
								minHeight: 50
							});
							// Ensure container takes full height of the widget slot
							if (element) {
								Object.assign(element.style, {
									width: "100%",
									height: "100%",
									maxWidth: "100%",
									maxHeight: "100%",
									display: "flex",
									justifyContent: "center",
									alignItems: "center",
									overflow: "hidden"
								});

								// If element is not the video itself, ensure inner video obeys container bounds
								if (element.tagName !== "VIDEO") {
									const constrainVideo = () => {
										element.querySelectorAll("video").forEach(v => {
											Object.assign(v.style, {
												width: "100%",
												height: "100%",
												maxWidth: "100%",
												maxHeight: "100%",
												objectFit: "contain"
											});
										});
									};
									constrainVideo();
									const observer = new MutationObserver(constrainVideo);
									observer.observe(element, { childList: true, subtree: true });
								} else {
									element.style.objectFit = "contain";
								}
							}
						}
						return widget;
					};
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
