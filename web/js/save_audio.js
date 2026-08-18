import { app } from "../../../scripts/app.js";

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

				this.onResize = function (size) {
					const minSize = this.computeSize ? this.computeSize() : [150, 80];
					size[0] = Math.max(size[0], minSize[0]);
					size[1] = Math.max(size[1], minSize[1]);
				};

				return r;
			};
		}
	},
});
