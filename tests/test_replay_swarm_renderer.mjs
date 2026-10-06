import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const template = readFileSync(new URL("../pml-mod/0.1.38/main.template.js", import.meta.url), "utf8");
const start = template.indexOf("    const replayState = {");
const end = template.indexOf("    pml.registerSimWorkerMixin(", start);
assert.notEqual(start, -1, "renderer state block must exist");
assert.notEqual(end, -1, "renderer state block must have an end");
const rendererSource = template.slice(start, end);

function makeHarness() {
  const cars = [];
  let clock = 0;
  const context = {
    performance: { now: () => (clock += 0.01) },
    TextEncoder,
    console,
  };
  context.globalThis = context;
  vm.runInNewContext(`(() => { ${rendererSource} })()`, context);
  const owner = {};
  let gameContext = {
    owner,
    deltaSeconds: 0,
    player: {
      getCarState: () => ({
        position: { x: 0, y: 0, z: 0 },
        quaternion: { x: 0, y: 0, z: 0, w: 1 },
      }),
      getCarStyle: () => ({ clone: () => ({ primaryHex: 0xffffff }) }),
    },
    createCar: () => {
      const car = {
        states: [],
        visible: false,
        disposed: false,
        opacity: 1,
        style: null,
        setCarState(state) {
          this.states.push(JSON.parse(JSON.stringify(state)));
        },
        setVisible(value) {
          this.visible = value;
        },
        setOpacity(value) {
          this.opacity = value;
        },
        setCarStyle(value) {
          this.style = value;
        },
        update() {},
        dispose() {
          this.disposed = true;
        },
      };
      cars.push(car);
      return car;
    },
  };
  const bind = (deltaSeconds = 0) => {
    gameContext.deltaSeconds = deltaSeconds;
    context.__polybotBindVisualReplayRenderer(gameContext);
  };
  const changeOwner = () => {
    gameContext = { ...gameContext, owner: {} };
    bind(0);
  };
  const dispatch = (op, params = {}) => {
    let response;
    context.__polybotVisualReplayDispatch({ op, params }, (message) => {
      response = message;
    });
    assert.ok(response, `${op} should respond synchronously`);
    if (!response.ok) throw new Error(response.error);
    return response.result;
  };
  bind();
  return { cars, bind, changeOwner, dispatch };
}

const identity = [0, 0, 0, 1];
function sample(tick, seconds, x, quaternion = identity) {
  return [tick, seconds, x, 0, 0, ...quaternion];
}

function loadSwarm(harness, episodes, { autoplay = true, behavior = "fade" } = {}) {
  const samplesTotal = episodes.reduce((sum, episode) => sum + episode.samples.length, 0);
  const chunks = episodes.map((episode) => {
    const result = [];
    for (let start = 0; start < episode.samples.length; start += 2) {
      result.push(episode.samples.slice(start, start + 2));
    }
    return result;
  });
  const payloadBytes = chunks.flat().reduce((sum, rows) => sum + Buffer.byteLength(JSON.stringify(rows)), 0);
  harness.dispatch("swarm_begin", {
    ghost_count: episodes.length,
    total_sample_count: samplesTotal,
    payload_bytes: payloadBytes,
    speed: 1,
    opacity: 0.5,
    end_behavior: behavior,
    fade_duration_s: 1,
  });
  for (let index = 0; index < episodes.length; index += 1) {
    const episode = episodes[index];
    harness.dispatch("swarm_episode_begin", {
      episode_id: episode.id,
      training_step_start: episode.step,
      sample_count: episode.samples.length,
      color: episode.color,
    });
    let chunkStart = 0;
    for (const rows of chunks[index]) {
      harness.dispatch("swarm_chunk", {
        episode_id: episode.id,
        start: chunkStart,
        samples: rows,
      });
      chunkStart += rows.length;
    }
    harness.dispatch("swarm_episode_commit", { episode_id: episode.id });
  }
  return harness.dispatch("swarm_commit", { autoplay });
}

test("loads a synchronized swarm with per-episode colors and SLERP interpolation", () => {
  const harness = makeHarness();
  const status = loadSwarm(harness, [
    { id: "run:a", step: 0, color: "#ff0000", samples: [
      sample(0, 0, 0),
      sample(1, 1, 2, [0, 0, 1, 0]),
    ] },
    { id: "run:b", step: 250000, color: "#ff8000", samples: [
      sample(0, 0, 10),
      sample(1, 1, 12),
      sample(2, 2, 14),
    ] },
  ]);
  assert.equal(status.loaded_ghosts, 2);
  assert.equal(status.visible_ghosts, 2);
  assert.equal(harness.cars[0].style.primaryHex, 0xff0000);
  assert.equal(harness.cars[1].style.primaryHex, 0xff8000);

  harness.bind(0.5);
  assert.equal(harness.cars[0].states.at(-1).position.x, 1);
  assert.equal(harness.cars[1].states.at(-1).position.x, 11);
  assert.ok(Math.abs(harness.cars[0].states.at(-1).quaternion.z - Math.SQRT1_2) < 1e-6);
  assert.ok(Math.abs(harness.cars[0].states.at(-1).quaternion.w - Math.SQRT1_2) < 1e-6);
});

test("supports different episode endpoints, seek, pause, resume, restart, and clear disposal", () => {
  const harness = makeHarness();
  loadSwarm(harness, [
    { id: "run:short", step: 0, color: "#ff0000", samples: [sample(0, 0, 0), sample(1, 1, 1)] },
    { id: "run:long", step: 100, color: "#00ff00", samples: [sample(0, 0, 10), sample(1, 1, 11), sample(2, 2, 12)] },
  ]);
  harness.bind(1.5);
  assert.equal(harness.cars[0].opacity, 0.25);
  assert.equal(harness.cars[1].opacity, 0.5);
  harness.dispatch("pause");
  harness.bind(0.5);
  assert.equal(harness.cars[1].states.at(-1).position.x, 11.5);
  harness.dispatch("seek", { seconds: 0.5 });
  assert.equal(harness.cars[1].states.at(-1).position.x, 10.5);
  harness.dispatch("play");
  harness.bind(0.5);
  assert.equal(harness.cars[1].states.at(-1).position.x, 11);
  harness.dispatch("restart");
  assert.equal(harness.cars[1].states.at(-1).position.x, 10);
  harness.bind(1.5);
  harness.dispatch("end_behavior", { value: "freeze" });
  assert.equal(harness.cars[0].visible, true);
  assert.equal(harness.cars[0].states.at(-1).position.x, 1);
  harness.dispatch("end_behavior", { value: "disappear" });
  assert.equal(harness.cars[0].visible, false);
  const cleared = harness.dispatch("clear");
  assert.equal(cleared.loaded_ghosts, 0);
  assert.ok(harness.cars.every((car) => car.disposed));
});

test("rejects malformed, duplicate, incomplete, and oversized swarm payloads", () => {
  const harness = makeHarness();
  assert.throws(
    () => harness.dispatch("swarm_begin", {
      ghost_count: 501,
      total_sample_count: 1,
      payload_bytes: 1,
      speed: 1,
      opacity: 0.5,
      end_behavior: "fade",
      fade_duration_s: 1,
    }),
    /ghost_count/,
  );
  assert.throws(
    () => harness.dispatch("swarm_begin", {
      ghost_count: 1,
      total_sample_count: 250001,
      payload_bytes: 1,
      speed: 1,
      opacity: 0.5,
      end_behavior: "fade",
      fade_duration_s: 1,
    }),
    /total_sample_count/,
  );
  const begin = {
    ghost_count: 1,
    total_sample_count: 2,
    payload_bytes: 1024,
    speed: 1,
    opacity: 0.5,
    end_behavior: "fade",
    fade_duration_s: 1,
  };
  harness.dispatch("swarm_begin", begin);
  harness.dispatch("swarm_episode_begin", {
    episode_id: "run:one", training_step_start: 0, sample_count: 2, color: "#ff0000",
  });
  assert.throws(
    () => harness.dispatch("swarm_episode_begin", {
      episode_id: "run:one", training_step_start: 1, sample_count: 1, color: "#00ff00",
    }),
    /no swarm is accepting/,
  );
  assert.throws(
    () => harness.dispatch("swarm_chunk", {
      episode_id: "wrong", start: 1, samples: [sample(1, 1, 1)],
    }),
    /out-of-order/,
  );
  assert.throws(
    () => harness.dispatch("swarm_chunk", {
      episode_id: "run:one", start: 0, samples: [[0, 0, 0, 0, 0, 0, 0, 0, 0]],
    }),
    /quaternion cannot be zero/,
  );
  harness.dispatch("swarm_chunk", {
    episode_id: "run:one", start: 0, samples: [sample(0, 0, 0)],
  });
  assert.throws(() => harness.dispatch("swarm_commit", { autoplay: true }), /incomplete/);

  const bytes = makeHarness();
  bytes.dispatch("swarm_begin", {
    ghost_count: 1,
    total_sample_count: 1,
    payload_bytes: 1,
    speed: 1,
    opacity: 0.5,
    end_behavior: "fade",
    fade_duration_s: 1,
  });
  bytes.dispatch("swarm_episode_begin", {
    episode_id: "run:bytes", training_step_start: 0, sample_count: 1, color: "#ff0000",
  });
  assert.throws(
    () => bytes.dispatch("swarm_chunk", {
      episode_id: "run:bytes", start: 0, samples: [sample(0, 0, 0)],
    }),
    /payload_bytes/,
  );

  const duplicate = makeHarness();
  duplicate.dispatch("swarm_begin", {
    ghost_count: 1,
    total_sample_count: 1,
    payload_bytes: 128,
    speed: 1,
    opacity: 0.5,
    end_behavior: "fade",
    fade_duration_s: 1,
  });
  duplicate.dispatch("swarm_episode_begin", {
    episode_id: "run:duplicate", training_step_start: 0, sample_count: 1, color: "#ff0000",
  });
  duplicate.dispatch("swarm_chunk", {
    episode_id: "run:duplicate", start: 0, samples: [sample(0, 0, 0)],
  });
  duplicate.dispatch("swarm_episode_commit", { episode_id: "run:duplicate" });
  assert.throws(
    () => duplicate.dispatch("swarm_episode_begin", {
      episode_id: "run:duplicate", training_step_start: 0, sample_count: 1, color: "#00ff00",
    }),
    /unique/,
  );
});

test("legacy one-replay protocol remains accepted without new payload metadata", () => {
  const harness = makeHarness();
  const rows = [sample(0, 0, 1)];
  harness.dispatch("begin", {
    sample_count: 1,
    color: "#123456",
    speed: 1,
    opacity: 0.5,
    end_behavior: "fade",
    fade_duration_s: 0.75,
  });
  harness.dispatch("chunk", { start: 0, samples: rows });
  const result = harness.dispatch("commit", { autoplay: false });
  assert.equal(result.loaded_ghosts, 1);
  assert.equal(harness.cars[0].style.primaryHex, 0x123456);
  assert.equal(harness.cars[0].visible, true);
  harness.changeOwner();
  assert.equal(harness.cars[0].disposed, true);
  assert.equal(harness.dispatch("status").loaded_ghosts, 0);
});
