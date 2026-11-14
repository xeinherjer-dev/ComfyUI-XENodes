import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const JS_DIR = path.resolve(__dirname, "../../web/js");

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
      _groups: [],
      links: new Map(),
      getNodeById() { return null; },
      findNodesByType() { return []; },
      setDirtyCanvas() {},
    },
    canvas: {
      graph: null,
      setGraph() {},
      centerOnNode() {},
      selectNode() {},
      setDirty() {},
    },
    ui: {
      settings: {
        addSetting() {},
        getSettingValue() { return true; },
      },
    },
  };
  mockApp.canvas.graph = mockApp.graph;

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
      this.dataset = {};
      this.children = [];
      this.parentNode = null;
      this.attributes = {};
      this._textContent = "";
      this._listeners = new Map();
      this.classList = { add() {}, remove() {}, toggle() {} };
    }
    get firstChild() {
      return this.children[0] || null;
    }
    get textContent() {
      return this._textContent;
    }
    set textContent(val) {
      this._textContent = val == null ? "" : String(val);
      this.children = [];
    }
    setAttribute(k, v) {
      this.attributes[k] = String(v);
    }
    getAttribute(k) {
      return this.attributes[k] ?? null;
    }
    appendChild(child) {
      if (child) {
        child.parentNode = this;
        this.children.push(child);
      }
      return child;
    }
    prepend(child) {
      if (child) {
        child.parentNode = this;
        this.children.unshift(child);
      }
      return child;
    }
    removeChild(child) {
      const idx = this.children.indexOf(child);
      if (idx !== -1) {
        this.children.splice(idx, 1);
        child.parentNode = null;
      }
      return child;
    }
    replaceChildren(...nodes) {
      for (const c of this.children) c.parentNode = null;
      this.children = [];
      for (const n of nodes) this.appendChild(n);
    }
    remove() {
      if (this.parentNode && Array.isArray(this.parentNode.children)) {
        const idx = this.parentNode.children.indexOf(this);
        if (idx !== -1) this.parentNode.children.splice(idx, 1);
      }
      this.parentNode = null;
    }
    addEventListener(type, fn) {
      if (!this._listeners.has(type)) this._listeners.set(type, []);
      this._listeners.get(type).push(fn);
    }
    removeEventListener(type, fn) {
      const list = this._listeners.get(type);
      if (list) {
        const idx = list.indexOf(fn);
        if (idx !== -1) list.splice(idx, 1);
      }
    }
    dispatchEvent(evt) {
      for (const fn of this._listeners.get(evt?.type) || []) fn(evt);
      return true;
    }
    querySelector() { return null; }
    querySelectorAll() { return []; }
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
      this.badges = [];
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
    addProperty(name, defaultValue) {
      this.properties = this.properties || {};
      if (!(name in this.properties)) {
        this.properties[name] = defaultValue;
      }
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
    serialize() {
      return { id: this.id, type: this.type, widgets_values: [] };
    }
    setDirtyCanvas() {}
    setSize() {}
  }

  const mockLiteGraph = {
    LGraphNode: MockLGraphNode,
    LGraphCanvas: class {},
    LGraphBadge: class {
      constructor(opts) {
        Object.assign(this, opts);
      }
    },
    NODE_TITLE_HEIGHT: 30,
    INPUT: 1,
    OUTPUT: 2,
    registered_node_types: {},
  };

  const docHead = new MockHTMLElement();
  const docBody = new MockHTMLElement();
  const findByIdInTree = (root, id) => {
    for (const child of root.children || []) {
      if (child.id === id) return child;
      const found = findByIdInTree(child, id);
      if (found) return found;
    }
    return null;
  };

  const context = vm.createContext({
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    requestAnimationFrame(cb) { return setTimeout(cb, 0); },
    cancelAnimationFrame(id) { clearTimeout(id); },
    performance: { now: () => Date.now() },
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
    URLSearchParams,
    Promise,
    HTMLElement: MockHTMLElement,
    MutationObserver: class {
      observe() {}
      disconnect() {}
      takeRecords() { return []; }
    },
    customElements: { define() {}, get() {} },
    LiteGraph: mockLiteGraph,
    window: {
      addEventListener() {},
      removeEventListener() {},
      localStorage: { getItem() { return null; }, setItem() {} },
    },
    document: {
      createElement(tag) {
        const el = new MockHTMLElement();
        el.tagName = String(tag).toUpperCase();
        return el;
      },
      createElementNS(ns, tag) {
        const el = new MockHTMLElement();
        el.namespaceURI = ns;
        el.tagName = tag;
        return el;
      },
      getElementById(id) {
        return findByIdInTree(docHead, id) || findByIdInTree(docBody, id);
      },
      querySelector() { return null; },
      querySelectorAll() { return []; },
      addEventListener() {},
      removeEventListener() {},
      head: docHead,
      body: docBody,
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
    eventListeners,
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

test("GUI Regression: fast_groups_muter_bypasser.js lifecycle, collapse guard, and cleanup", async () => {
  const env = createMockEnvironment();
  await loadScript("fast_groups_muter_bypasser.js", env);
  assert.ok(env.registeredExtensions.length >= 1);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.FastGroupsMuterBypasser");
  assert.ok(ext, "Extension XENodes.FastGroupsMuterBypasser must be registered");

  const nodeType = class extends env.MockLGraphNode {};
  await ext.beforeRegisterNodeDef(nodeType, { name: "XENodes.FastGroupsMuterBypasser" }, env.mockApp);

  const instance = new nodeType();
  instance.graph = env.mockApp.graph;
  try {
    instance.onNodeCreated();
    assert.ok(instance._rebuildGroups, "_rebuildGroups helper must be initialized");
    assert.ok(instance._rebuildInterval, "periodic rebuild interval must be scheduled");

    instance._rebuildGroups();
    const rowWidget = instance.widgets.find((w) => w.name === "xen_fgm_rows");
    assert.ok(rowWidget, "xen_fgm_rows DOM widget must be attached");

    // When collapsed, _rebuildGroups should short-circuit without altering existing children
    const initialCount = rowWidget.element.children.length;
    instance.flags.collapsed = true;
    rowWidget.element.appendChild(env.context.document.createElement("div"));
    instance._rebuildGroups();
    assert.equal(rowWidget.element.children.length, initialCount + 1);
  } finally {
    instance.onRemoved?.();
  }
  assert.equal(instance._rebuildInterval, null, "onRemoved must clear _rebuildInterval");
});

test("GUI Regression: slider.js clamps values, protects DOM styles by id, and cleans up", async () => {
  const env = createMockEnvironment();
  await loadScript("slider.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.Slider");
  assert.ok(ext, "Extension XENodes.Slider must be registered");

  const nodeType = class extends env.MockLGraphNode {};
  const nodeData = { name: "XENodes.Slider" };
  await ext.beforeRegisterNodeDef(nodeType, nodeData, env.mockApp);

  const instance = new nodeType();
  instance.id = -1;
  instance.widgets = [{ name: "value", value: 20 }];
  instance.onNodeCreated();
  assert.ok(instance.widgets.length > 0);
  // Node ID -1 must NOT inject a per-node <style id="xe-slider-style--1">
  assert.equal(env.context.document.getElementById("xe-slider-style--1"), null);

  instance.id = 101;
  instance.onAdded();
  const styleEl = env.context.document.getElementById("xe-slider-style-101");
  assert.ok(styleEl, "onAdded must attach per-node <style> once real node id is assigned");
  assert.ok(styleEl.textContent.includes('[data-node-id="101"]'));

  // onConfigure must clamp out-of-range values to [min, max]
  instance.properties.min = 10;
  instance.properties.max = 50;
  instance.properties.value = 999;
  instance.onConfigure();
  assert.equal(instance.properties.value, 50);

  // Disconnected output must default to "INT,FLOAT"
  instance.outputs = [{ type: "*", name: "value", links: [] }];
  instance.onConnectionsChange(env.context.LiteGraph.OUTPUT, 0, false, null);
  assert.equal(instance.outputs[0].type, "INT,FLOAT");

  instance.onRemoved();
  assert.equal(env.context.document.getElementById("xe-slider-style-101"), null, "onRemoved must remove per-node <style>");
});

test("GUI Regression: slider2d.js style lifecycle, Map links, and default INT,FLOAT type", async () => {
  const env = createMockEnvironment();
  await loadScript("slider2d.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.Slider2D");
  assert.ok(ext, "Extension XENodes.Slider2D must be registered");

  const nodeType = class extends env.MockLGraphNode {};
  const nodeData = { name: "XENodes.Slider2D" };
  await ext.beforeRegisterNodeDef(nodeType, nodeData, env.mockApp);

  const instance = new nodeType();
  instance.id = -1;
  instance.widgets = [{ name: "X", value: 512 }, { name: "Y", value: 512 }];
  instance.onNodeCreated();
  assert.ok(instance.widgets.length > 0);
  assert.equal(env.context.document.getElementById("xe-slider2d-slot-style--1"), null);

  instance.id = 102;
  instance.onAdded();
  const styleEl = env.context.document.getElementById("xe-slider2d-slot-style-102");
  assert.ok(styleEl, "onAdded must attach per-node <style>");

  instance.outputs = [{ type: "*", name: "X", links: [] }, { type: "*", name: "Y", links: [] }];
  instance.onConnectionsChange?.(env.context.LiteGraph.OUTPUT, 0, false, null);
  assert.equal(instance.outputs[0].type, "INT,FLOAT");

  instance.onRemoved();
  assert.equal(env.context.document.getElementById("xe-slider2d-slot-style-102"), null);
});

test("GUI Regression: multi_switch.js hides connections safely and cleans up timers/styles", async () => {
  const env = createMockEnvironment();
  await loadScript("multi_switch.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.MultiSwitch");
  assert.ok(ext, "Extension XENodes.MultiSwitch must be registered");

  const nodeType = class extends env.MockLGraphNode {};
  await ext.beforeRegisterNodeDef(nodeType, { name: "XENodes.MultiSwitch" }, env.mockApp);

  const instance = new nodeType();
  instance.id = -1;
  instance.graph = env.mockApp.graph;
  instance.properties = { hide_connections: true };
  instance.inputs = [{ name: "inputs.input1", type: "*", link: null }];
  instance.widgets = [{ name: "select", value: 0 }, { name: "selected_label", value: "" }];
  try {
    instance.onNodeCreated();
    assert.equal(env.context.document.getElementById("multi-switch-style--1"), null);

    instance.id = 201;
    instance.onAdded();
    const styleEl = env.context.document.getElementById("multi-switch-style-201");
    assert.ok(styleEl, "onAdded must add multi-switch style tag");
    assert.ok(styleEl.textContent.includes('[data-node-id="201"]'));
  } finally {
    instance.onRemoved();
  }
  assert.equal(env.context.document.getElementById("multi-switch-style-201"), null);
  assert.equal(instance.title_check_interval, null);
  assert.equal(instance.sync_interval, null);
});

test("GUI Regression: multi_pipe.js extensions register and initialize nodes", async () => {
  const env = createMockEnvironment();
  await loadScript("multi_pipe.js", env);
  const extIn = env.registeredExtensions.find((e) => e.name === "XENodes.MultiPipeIn");
  const extOut = env.registeredExtensions.find((e) => e.name === "XENodes.MultiPipeOut");
  assert.ok(extIn, "Extension XENodes.MultiPipeIn must be registered");
  assert.ok(extOut, "Extension XENodes.MultiPipeOut must be registered");

  for (const [ext, name] of [[extIn, "XENodes.MultiPipeIn"], [extOut, "XENodes.MultiPipeOut"]]) {
    const nodeType = class extends env.MockLGraphNode {};
    await ext.beforeRegisterNodeDef(nodeType, { name }, env.mockApp);
    const instance = new nodeType();
    instance.graph = env.mockApp.graph;
    try {
      instance.onNodeCreated();
    } finally {
      instance.onRemoved?.();
    }
  }
});

test("GUI Regression: show_any.js extension registers, executes, and serializes preview", async () => {
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

  const serialized = instance.serialize();
  assert.ok(Array.isArray(serialized?.widgets_values) && serialized.widgets_values.length > 0);
});

test("GUI Regression: load_image.js extension registers and chains mode descriptor", async () => {
  const env = createMockEnvironment();
  await loadScript("load_image.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.LoadImageFromFolder");
  assert.ok(ext, "Extension XENodes.LoadImageFromFolder must be registered");

  const nodeType = class extends env.MockLGraphNode {};
  await ext.beforeRegisterNodeDef(nodeType, { name: "XENodes.LoadImageFromFolder" }, env.mockApp);

  const instance = new nodeType();
  let upstreamSetCalled = false;
  let upstreamVal = 0;
  Object.defineProperty(instance, "mode", {
    get() { return upstreamVal; },
    set(v) { upstreamSetCalled = true; upstreamVal = v; },
    configurable: true,
    enumerable: true,
  });
  instance.widgets = [{ name: "path", value: "output" }, { name: "index", value: 0 }];
  instance.onNodeCreated();
  instance.mode = 2;
  assert.ok(upstreamSetCalled, "chained mode descriptor must preserve upstream setter");
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
  let origCalled = false;
  instance.onResize = () => { origCalled = true; };
  instance.onNodeCreated();
  assert.equal(instance.previewMediaType, "audio");
  assert.ok(instance.onResize);
  const sz = [50, 40];
  instance.onResize(sz);
  assert.ok(origCalled, "onResize must chain any prior onResize handler");
  assert.equal(sz[0], 150);
  assert.equal(sz[1], 100);
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

test("GUI Regression: node_execution_time.js accumulates repeated execution and resets on error", async () => {
  const env = createMockEnvironment();
  await loadScript("node_execution_time.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.NodeExecutionTime");
  assert.ok(ext, "Extension XENodes.NodeExecutionTime must be registered");

  const node = new env.MockLGraphNode();
  node.id = 7;
  env.mockApp.graph._nodes = [node];
  env.mockApp.graph.getNodeById = (id) => (Number(id) === 7 ? node : null);

  ext.setup();
  ext.nodeCreated(node);
  assert.equal(node.badges.length, 1);

  // Badge instance must be cached while text doesn't change
  const b1 = node.badges[0]();
  const b2 = node.badges[0]();
  assert.equal(b1, b2, "LGraphBadge instance should be reused when text is unchanged");

  const startListeners = env.eventListeners.get("execution_start") || [];
  const execListeners = env.eventListeners.get("executing") || [];
  const errListeners = env.eventListeners.get("execution_error") || [];
  assert.ok(startListeners.length > 0 && execListeners.length > 0 && errListeners.length > 0);

  startListeners.forEach((fn) => fn({}));
  execListeners.forEach((fn) => fn({ detail: "7" }));
  await new Promise((r) => setTimeout(r, 15));
  execListeners.forEach((fn) => fn({ detail: null }));
  const firstDuration = node.executionDuration;
  assert.ok(firstDuration > 0 && node._xeMeasured === true);

  // Repeating the same node within the same run must accumulate duration
  execListeners.forEach((fn) => fn({ detail: "7" }));
  await new Promise((r) => setTimeout(r, 15));
  execListeners.forEach((fn) => fn({ detail: null }));
  assert.ok(node.executionDuration > firstDuration, "repeat execution must accumulate duration");

  // execution_error must reset active timer state so subsequent executing does not add phantom time
  execListeners.forEach((fn) => fn({ detail: "7" }));
  errListeners.forEach((fn) => fn({}));
  const afterErr = node.executionDuration;
  execListeners.forEach((fn) => fn({ detail: null }));
  assert.equal(node.executionDuration, afterErr);
});

test("GUI Regression: progress_bar.js registers cleanly without duplicate settings", async () => {
  const env = createMockEnvironment();
  let addSettingCalls = 0;
  env.mockApp.ui.settings.addSetting = () => { addSettingCalls++; };
  await loadScript("progress_bar.js", env);
  const ext = env.registeredExtensions.find((e) => e.name === "XENodes.ProgressBar");
  assert.ok(ext, "Extension XENodes.ProgressBar must be registered");
  assert.ok(Array.isArray(ext.settings) && ext.settings.length === 1);
  await ext.setup?.();
  assert.equal(addSettingCalls, 0, "setup() must not re-register setting already defined in settings array");
});
