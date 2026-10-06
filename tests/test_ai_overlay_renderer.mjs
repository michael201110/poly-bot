import assert from "node:assert/strict";
import { test } from "node:test";

import { installPolyBotHudOverlay } from "../pml-mod/0.1.39/hud_renderer.mjs";

class FakeNode {
  constructor(tagName) {
    this.tagName = tagName;
    this.className = "";
    this.children = [];
    this.parentNode = null;
    this.style = {};
    this.attributes = {};
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
}

function createRenderer() {
  const document = {
    body: new FakeNode("body"),
    head: new FakeNode("head"),
    createElement: (tag) => new FakeNode(tag),
  };
  const host = {
    document,
    setTimeout: () => 1,
    clearTimeout: () => {},
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

test("renderer creates a non-interactive overlay and reuses feature rows", () => {
  const { document, update } = createRenderer();
  const frame = sampleFrame();
  update(frame);
  const root = document.body.children.find((node) => node.className.includes("polybot-ai-hud"));
  assert.ok(root);
  assert.equal(root.attributes["aria-hidden"], "true");
  assert.equal(root.style.transform, "scale(1.00)");
  const firstInput = sectionBody(root, 3).children[0];
  const firstValue = firstInput.children[1];
  assert.match(firstValue.textContent, /^0\.000/);

  update(sampleFrame({
    training_step: 25_001,
    features: frame.features.map((item) => ({ ...item, value: 0.75 })),
  }));
  assert.equal(
    document.body.children.filter((node) => node.className.includes("polybot-ai-hud")).length,
    1,
  );
  assert.equal(sectionBody(root, 3).children[0], firstInput);
  assert.match(firstValue.textContent, /^0\.750/);

  update({ enabled: false });
  assert.equal(root.hidden, true);
});

test("unsupported feature schemas are made explicit instead of being mislabeled", () => {
  const { document, update } = createRenderer();
  update(sampleFrame({ observation_schema: "polybot.observation.v3" }));
  const root = document.body.children.find((node) => node.className.includes("polybot-ai-hud"));
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
