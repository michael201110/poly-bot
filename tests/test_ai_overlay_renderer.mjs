import assert from "node:assert/strict";
import { test } from "node:test";

import { HUD_MODES, importantInputs, installPolyBotHudOverlay } from "../pml-mod/0.1.41/hud_renderer.mjs";

class FakeNode {
  constructor(tagName) {
    this.tagName = tagName;
    this.className = "";
    this.children = [];
    this.parentNode = null;
    this.style = {};
    this.attributes = {};
    this.listeners = {};
    this.hidden = false;
    this.textContent = "";
  }

  appendChild(child) {
    if (child.parentNode) {
      child.parentNode.children = child.parentNode.children.filter((item) => item !== child);
    }
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  setAttribute(name, value) {
    this.attributes[name] = value;
  }

  getAttribute(name) {
    return this.attributes[name] ?? null;
  }

  addEventListener(name, callback) {
    this.listeners[name] = callback;
  }

  click() {
    this.listeners.click?.({
      preventDefault() {},
      stopPropagation() {},
    });
  }
}

function createRenderer({ readyState = "complete" } = {}) {
  const document = {
    readyState,
    body: new FakeNode("body"),
    head: new FakeNode("head"),
    listeners: {},
    createElement: (tag) => new FakeNode(tag),
    addEventListener(name, callback) {
      this.listeners[name] = callback;
    },
  };
  const host = {
    document,
  };
  return { document, update: installPolyBotHudOverlay(host) };
}

function sampleFrame(overrides = {}) {
  return {
    schema: "polybot.ai-overlay-frame.v1",
    enabled: true,
    mode: "training",
    status: "running",
    track: "Summer 1",
    algorithm: "grtqc",
    training_step: 25_000,
    episode: 4,
    episode_total: 10,
    decision_count: 100,
    simulator_tick: 3_000,
    elapsed_simulation_s: 25,
    frame_skip: 30,
    settings: {
      enabled: true,
      preset: "full",
      scale: 1,
      show_episode_status: true,
      show_labels: true,
      show_observations: true,
      show_controls: true,
      show_reward_breakdown: true,
      show_event_popups: true,
      lookahead_points: 12,
    },
    observation_schema: "polybot.observation.v2",
    feature_schema: "polybot.observation-features.v1",
    features: Array.from({ length: 100 }, (_, index) => ({
      index,
      key: `feature.${index}`,
      label: `Feature ${index}`,
      group: "vehicle",
      unit: "normalized",
      raw_value: index / 10,
      value: index / 100,
      minimum: -1,
      maximum: 1,
    })),
    controls: {
      model_output: [{ label: "Steer", value: 0.25 }],
      transformed_output: [{ label: "Steer", value: 0.2 }],
      applied: { steer: 0.25, throttle: 1, brake: 0 },
    },
    reward: {
      learner_step: 0.5,
      episode_learner: 10,
      groups: { Progress: 0.5 },
      terms: { progress: 0.5 },
    },
    events: [],
    ...overrides,
  };
}

function sectionBody(root, index) {
  return root.children[index].children.at(-1);
}

function hudRoot(document) {
  return document.body.children.find((node) =>
    node.className === "polybot-ai-hud" || node.className.startsWith("polybot-ai-hud "),
  );
}

test("plugin toggle controls the overlay and telemetry updates reuse feature rows", () => {
  const { document, update } = createRenderer();
  const toggle = document.body.children.find((node) => node.className.includes("polybot-ai-hud-toggle"));
  const root = hudRoot(document);
  assert.ok(toggle);
  assert.ok(root);
  assert.equal(toggle.textContent, "Show AI HUD");
  assert.equal(root.hidden, true);
  toggle.click();
  assert.equal(toggle.textContent, "Hide AI HUD");
  assert.equal(root.hidden, false);

  const frame = sampleFrame();
  update(frame);
  assert.equal(root.attributes["aria-hidden"], "false");
  assert.equal(root.style.transform, "scale(1.00)");
  const firstInput = sectionBody(root, 3).children[0];
  const firstValue = firstInput.children[1];
  assert.match(firstValue.textContent, /^0\.000/);

  update(sampleFrame({
    training_step: 25_001,
    features: frame.features.map((item) => ({ ...item, value: 0.75 })),
  }));
  assert.equal(
    document.body.children.filter((node) =>
      node.className === "polybot-ai-hud" || node.className.startsWith("polybot-ai-hud "),
    ).length,
    1,
  );
  assert.equal(sectionBody(root, 3).children[0], firstInput);
  assert.match(firstValue.textContent, /^0\.750/);

  update({ enabled: false });
  assert.equal(root.hidden, false);
  toggle.click();
  assert.equal(root.hidden, true);
  toggle.click();
  assert.equal(root.hidden, false);
});

test("HUD control exists before Python telemetry is available", () => {
  const { document, update } = createRenderer();
  update();
  const toggle = document.body.children.find((node) => node.className.includes("polybot-ai-hud-toggle"));
  const root = hudRoot(document);
  assert.ok(toggle);
  assert.ok(root);
  toggle.click();
  assert.equal(root.hidden, false);
  assert.match(root.children[1].textContent, /Waiting for Python telemetry/);
});

test("HUD control waits for the page body when PML initializes early", () => {
  const { document, update } = createRenderer({ readyState: "loading" });
  assert.equal(document.body.children.length, 0);
  document.listeners.DOMContentLoaded();
  update();
  assert.ok(document.body.children.some((node) => node.className.includes("polybot-ai-hud-toggle")));
});

test("unsupported feature schemas are made explicit instead of being mislabeled", () => {
  const { document, update } = createRenderer();
  update(sampleFrame({ observation_schema: "polybot.observation.v3" }));
  const root = hudRoot(document);
  const inputSection = root.children[3];
  assert.match(inputSection.children[1].textContent, /Unsupported observation schema/);
  assert.equal(inputSection.children.at(-1).hidden, true);
});

test("renderer mock-DOM update work stays below one millisecond per full frame", () => {
  const { update } = createRenderer();
  const frame = sampleFrame();
  update(frame);
  const started = process.hrtime.bigint();
  for (let index = 0; index < 1_000; index += 1) {
    update(frame);
  }
  const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
  const meanMs = elapsedMs / 1_000;
  console.info(`AI HUD mock-DOM average: ${meanMs.toFixed(4)} ms/update`);
  assert.ok(meanMs < 1, `average update took ${meanMs.toFixed(4)} ms`);
});


test("all requested modes switch live and OFF hides the overlay", () => {
  const {document, update} = createRenderer();
  const root = hudRoot(document);
  const selector = document.body.children.find(n => n.className === "polybot-ai-hud-mode");
  assert.equal(selector.children.length, 15);
  for (const mode of HUD_MODES) {
    selector.value = mode;
    selector.listeners.change();
    update(sampleFrame());
    assert.equal(root.hidden, mode === "OFF", mode);
    assert.equal(root.children[3].hidden, !["NEURAL_NET","RL_DEBUG","OBSERVATIONS"].includes(mode), mode);
    assert.equal(root.children[5].hidden, !["RL_DEBUG","REWARD"].includes(mode), mode);
    assert.equal(root.children[4].hidden, ["OFF","TRAINING","TRAINING_GRAPH"].includes(mode), mode);
  }
});

test("input curation retains driving signals and leaves the complete vector unchanged", () => {
  const features = sampleFrame().features;
  features[99] = {...features[99], key:"velocity.forward"};
  features[98] = {...features[98], key:"route.heading_error"};
  const original = JSON.stringify(features);
  const selected = importantInputs(features);
  assert.equal(selected.length,60);
  assert.ok(selected.some(f=>f.key === "velocity.forward"));
  assert.ok(selected.some(f=>f.key === "route.heading_error"));
  assert.equal(JSON.stringify(features), original);
});

test("missing WR trace is explicit and a target never becomes a fake live delta", () => {
  const {document, update} = createRenderer();
  update(sampleFrame({settings:{display_mode:"WR_CHASE",wr_target_s:22}}));
  const texts = hudRoot(document).children.at(-1).children.at(-1).children.map(n=>n.textContent).join(" ");
  assert.match(texts,/WR target 22.000 s/);
  assert.match(texts,/WR split trace unavailable/);
  assert.doesNotMatch(texts,/Live WR delta \+?3.000/);
});
