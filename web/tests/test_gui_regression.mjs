import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const JS_DIR = path.resolve(__dirname, "../js");

/**
 * Creates an isolated mock environment for ComfyUI web extensions.
 */
function createMockEnvironment() {
  const registeredExtensions = [];
  const eventListeners = new Map();

  const mockApp = {
    registerExtension(ext) {
      registeredExtensions.push(ext);
    },
    graph: {
      _nodes: [],
      findNodesByType() { return []; },
      setDirtyCanvas() {},
    },
    ui: {
      settings: {
        addSetting() {},
        getSettingValue() { return true; },
      },
    },
  };

  const mockApi = {
    addEventListener(event, handler) {
      if (!eventListeners.has(event)) eventListeners.set(event, []);
      eventListeners.get(event).push(handler);
    },
    removeEventListener() {},
    apiURL(endpoint) { return endpoint; },
    fetchApi() { return Promise.resolve(new Response("[]")); },
  };

  const mockComfyWidgets = {
    STRING(node, inputName, inputData) {
      const widget = {
        name: inputName,
        type: "customtext",
        value: "",
        element: { style: {} },
      };
      node.widgets = node.widgets || [];
      node.widgets.push(widget);
      return { widget };
    },
  };

  class MockHTMLElement {
    constructor() {
      this.style = {};
      this.classList = { add() {}, remove() {}, toggle() {} };
    }
    appendChild() {}
    remove() {}
    addEventListener() {}
    closest() { return null; }
    getBoundingClientRect() {
      return { left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100 };
    }
    getContext() {
      const dummyCtx = { canvas: this };
      return new Proxy(dummyCtx, {
        get(target, prop) {
          if (!(prop in target)) {
            return () => {};
          }
          return target[prop];
        },
        set(target, prop, value) {
          target[prop] = value;
          return true;
        },
      });
    }
  }

  // Minimal LiteGraph mock
  class MockLGraphNode {
    constructor() {
      this.widgets = [];
      this.outputs = [{ type: "COMBO", name: "COMBO" }, { type: "STRING", name: "STRING" }];
      this.inputs = [];
      this.properties = {};
      this.flags = {};
      this.size = [100, 100];
    }
    addWidget(type, name, value, callback, options) {
      const widget = { type, name, value, callback, options: options || {} };
      this.widgets.push(widget);
      return widget;
    }
    addDOMWidget(name, type, element, options) {
      const widget = { name, type, element, options: options || {} };
      this.widgets.push(widget);
      return widget;
    }
    addInput(name, type) {
      const input = { name, type };
      this.inputs.push(input);
      return input;
    }
    addOutput(name, type) {
      const output = { name, type };
      this.outputs.push(output);
      return output;
    }
    removeInput(index) {
      this.inputs.splice(index, 1);
    }
    removeOutput(index) {
      this.outputs.splice(index, 1);
    }
    computeSize() {
      return [150, 100];
    }
    setDirtyCanvas() {}
    setSize() {}
  }

  const mockLiteGraph = {
    LGraphNode: MockLGraphNode,
    LGraphCanvas: class {},
    registered_node_types: {},
  };

  const context = vm.createContext({
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    requestAnimationFrame(cb) { return setTimeout(cb, 0); },
    Math,
    Date,
    JSON,
    Array,
    Object,
    String,
    Number,
    Boolean,
    RegExp,
    Error,
    Map,
    Set,
    Promise,
    HTMLElement: MockHTMLElement,
    customElements: { define() {}, get() {} },
    LiteGraph: mockLiteGraph,
    window: {
      addEventListener() {},
      removeEventListener() {},
      localStorage: { getItem() { return null; }, setItem() {} },
    },
    document: {
      createElement() {
        return new MockHTMLElement();
      },
      getElementById() {
        return null;
      },
      head: new MockHTMLElement(),
      body: new MockHTMLElement(),
    },
    fetch() {
      return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
    },
  });

  return {
    context,
    mockApp,
    mockApi,
    mockComfyWidgets,
    registeredExtensions,
    MockLGraphNode,
  };
}

/**
 * Loads an ESM script using vm.SourceTextModule with intercepted imports.
 */
async function loadScript(filename, env) {
  const filePath = path.join(JS_DIR, filename);
  const code = fs.readFileSync(filePath, "utf-8");

  const module = new vm.SourceTextModule(code, {
    context: env.context,
    identifier: filePath,
    initializeImportMeta(meta) {
      meta.url = `file://${filePath}`;
    },
    importModuleDynamically() {
      throw new Error("Dynamic import not supported in test");
    },
  });

  await module.link(async (specifier) => {
    if (specifier.endsWith("app.js")) {
      return new vm.SyntheticModule(["app"], function () {
        this.setExport("app", env.mockApp);
      }, { context: env.context });
    }
    if (specifier.endsWith("api.js")) {
      return new vm.SyntheticModule(["api"], function () {
        this.setExport("api", env.mockApi);
      }, { context: env.context });
    }
    if (specifier.endsWith("widgets.js")) {
      return new vm.SyntheticModule(["ComfyWidgets"], function () {
        this.setExport("ComfyWidgets", env.mockComfyWidgets);
      }, { context: env.context });
    }
    throw new Error(`Unhandled import specifier: ${specifier}`);
  });

  await module.evaluate();
  return env.registeredExtensions;
}

test("GUI Regression: combo_selector.js extension registers and configures node", async () => {
  const env = createMockEnvironment();
  await loadScript("combo_selector.js", env);
  assert.equal(env.registeredExtensions.length, 1);
  const ext = env.registeredExtensions[0];
  assert.equal(ext.name, "XENodes.ComboSelector");

  // Verify beforeRegisterNodeDef hooks
  const nodeType = class extends env.MockLGraphNode {};
  const nodeData = { name: "XENodes.ComboSelector" };
  await ext.beforeRegisterNodeDef(nodeType, nodeData, env.mockApp);

  const instance = new nodeType();
  instance.widgets = [{ name: "index", value: 0 }, { name: "hidden_list", value: "[]" }];
  instance.onNodeCreated();

  assert.equal(instance.outputs[0].type, "COMBO");
  assert.ok(instance.comboWidget, "comboWidget must be added to node");
});

test("GUI Regression: fast_groups_muter_bypasser.js extension registers", async () => {
  const env = createMockEnvironment();
  await loadScript("fast_groups_muter_bypasser.js", env);
  assert.ok(env.registeredExtensions.length >= 1);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.FastGroupsMuterBypasser");
  assert.ok(ext, "Extension XENodes.FastGroupsMuterBypasser must be registered");
});

test("GUI Regression: slider.js extension registers and configures node", async () => {
  const env = createMockEnvironment();
  await loadScript("slider.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.Slider");
  assert.ok(ext, "Extension XENodes.Slider must be registered");

  const nodeType = class extends env.MockLGraphNode {};
  const nodeData = { name: "XENodes.Slider" };
  await ext.beforeRegisterNodeDef(nodeType, nodeData, env.mockApp);

  const instance = new nodeType();
  instance.widgets = [{ name: "value", value: 20 }];
  instance.onNodeCreated();
  assert.ok(instance.widgets.length > 0);
});

test("GUI Regression: slider2d.js extension registers and configures node", async () => {
  const env = createMockEnvironment();
  await loadScript("slider2d.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.Slider2D");
  assert.ok(ext, "Extension XENodes.Slider2D must be registered");

  const nodeType = class extends env.MockLGraphNode {};
  const nodeData = { name: "XENodes.Slider2D" };
  await ext.beforeRegisterNodeDef(nodeType, nodeData, env.mockApp);

  const instance = new nodeType();
  instance.widgets = [{ name: "X", value: 512 }, { name: "Y", value: 512 }];
  instance.onNodeCreated();
  assert.ok(instance.widgets.length > 0);
});

test("GUI Regression: multi_switch.js extension registers", async () => {
  const env = createMockEnvironment();
  await loadScript("multi_switch.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.MultiSwitch");
  assert.ok(ext, "Extension XENodes.MultiSwitch must be registered");
});

test("GUI Regression: multi_pipe.js extensions register", async () => {
  const env = createMockEnvironment();
  await loadScript("multi_pipe.js", env);
  const extIn = env.registeredExtensions.find((e) => e.name === "XENodes.MultiPipeIn");
  const extOut = env.registeredExtensions.find((e) => e.name === "XENodes.MultiPipeOut");
  assert.ok(extIn, "Extension XENodes.MultiPipeIn must be registered");
  assert.ok(extOut, "Extension XENodes.MultiPipeOut must be registered");
});

test("GUI Regression: show_any.js extension registers and handles execution", async () => {
  const env = createMockEnvironment();
  await loadScript("show_any.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.ShowAny");
  assert.ok(ext, "Extension XENodes.ShowAny must be registered");

  const nodeType = class extends env.MockLGraphNode {};
  const nodeData = { name: "XENodes.ShowAny" };
  await ext.beforeRegisterNodeDef(nodeType, nodeData, env.mockApp);

  const instance = new nodeType();
  assert.ok(instance.onExecuted, "onExecuted handler must be attached to prototype");

  // Call onExecuted with sample message
  instance.onExecuted({ text: ["Sample Text Value"] });
  assert.ok(instance.widgets.length > 0, "widgets must be populated by onExecuted");
});

test("GUI Regression: load_image.js extension registers", async () => {
  const env = createMockEnvironment();
  await loadScript("load_image.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.LoadImageFromFolder");
  assert.ok(ext, "Extension XENodes.LoadImageFromFolder must be registered");
});

test("GUI Regression: save_audio sets previewMediaType and resize", async () => {
  const env = createMockEnvironment();
  await loadScript("save_audio.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.SaveAudio");
  assert.ok(ext, "Extension XENodes.SaveAudio must be registered");

  const nodeType = class extends env.MockLGraphNode {};
  const nodeData = { name: "XENodes.SaveAudio" };
  await ext.beforeRegisterNodeDef(nodeType, nodeData, env.mockApp);

  const instance = new nodeType();
  instance.onNodeCreated();
  assert.equal(instance.previewMediaType, "audio");
  assert.ok(instance.onResize);
});

test("GUI Regression: save_video and save_hdr_video intercept addDOMWidget", async () => {
  for (const filename of ["save_video.js", "save_hdr_video.js"]) {
    const env = createMockEnvironment();
    await loadScript(filename, env);
    const nodeName = filename === "save_video.js" ? "XENodes.SaveVideo" : "XENodes.SaveHDRVideo";
    const ext = env.registeredExtensions.find((e) => e.name === nodeName);
    assert.ok(ext, `Extension ${nodeName} must be registered`);

    const nodeType = class extends env.MockLGraphNode {};
    const nodeData = { name: nodeName };
    await ext.beforeRegisterNodeDef(nodeType, nodeData, env.mockApp);

    const instance = new nodeType();
    instance.onNodeCreated();
    assert.ok(instance.addDOMWidget, "addDOMWidget should be intercepted");

    // Add video-preview DOM widget and test layout override
    const widget = instance.addDOMWidget("video-preview", "video", {});
    assert.ok(widget.computeLayoutSize);
    const layout = widget.computeLayoutSize();
    assert.equal(layout.minWidth, 50);
    assert.equal(layout.minHeight, 200);
  }
});

test("GUI Regression: ui extensions register correctly", async () => {
  const uiScripts = [
    { file: "node_execution_time.js", name: "XENodes.NodeExecutionTime" },
    { file: "progress_bar.js", name: "XENodes.ProgressBar" },
  ];

  for (const item of uiScripts) {
    const env = createMockEnvironment();
    await loadScript(item.file, env);
    const ext = env.registeredExtensions.find((e) => e.name === item.name);
    assert.ok(ext, `Extension ${item.name} must be registered`);
  }
});
