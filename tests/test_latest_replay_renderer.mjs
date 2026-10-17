// Exercise the manifest's current renderer as well as the historical 0.1.42
// suite. Native cars are mocked; this is not a GPU/PolyTrack FPS benchmark.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const manifest = JSON.parse(readFileSync(new URL("../pml-mod/manifest.json", import.meta.url)));
const source = readFileSync(new URL(`../pml-mod/${manifest.latest["0.6.3"]}/replay_renderer.mjs`, import.meta.url), "utf8")
  .replace("export function installPolyBotReplayRenderer", "function installPolyBotReplayRenderer");

function harness() {
  const cars = [];
  let camera = {};
  const originalCamera = camera;
  const context = { performance, TextEncoder, structuredClone, console };
  context.globalThis = context;
  vm.runInNewContext(`${source}; installPolyBotReplayRenderer()`, context);
  const game = {
    owner: {}, deltaSeconds: 0,
    player: { getCarState: () => ({}), setVisible() {},
      getCarStyle: () => ({ clone() { return { ...this }; } }) },
    getCamera: () => camera, setCamera: (value) => { camera = value; },
    createCar: (_base, withAudio) => {
      const car = { cameraOrbit: {}, states: [], withAudio, disposed: false, setCarStyle() {}, setOpacity() {},
        setVisible() {}, update() {}, updateCameras() {}, dispose() { this.disposed = true; },
        setCarState(state) { this.states.push(structuredClone(state)); } };
      cars.push(car);
      return car;
    },
  };
  function bind(seconds) {
    game.deltaSeconds = seconds;
    context.__polybotBindVisualReplayRenderer(game);
  }
  function dispatch(op, params = {}) {
    let response;
    context.__polybotVisualReplayDispatch({ op, params }, (message) => { response = message; });
    assert.ok(response?.ok, response?.error);
    return response.result;
  }
  bind(0);
  return { cars, bind, dispatch, camera: () => camera, originalCamera };
}

function load(h, count) {
  const samples = [[0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0.5, 10, 0, 0, 0, 0, 0, 1],
    [2, 1, 20, 0, 0, 0, 0, 0, 1]];
  const wheel_states = samples.map((_, sample) => Array.from({length: 42}, (_, column) =>
    column < 40 ? [1, sample * 10, 0, 0, 0, 1, 0, sample * .1, sample * .2, sample * .4][column % 10] : 0));
  const chunk = {start: 0, samples, wheel_states};
  h.dispatch("swarm_begin", {ghost_count: count, total_sample_count: count * samples.length,
    payload_bytes: count * Buffer.byteLength(JSON.stringify(chunk)), speed: 1, opacity: 1,
    end_behavior: "fade", fade_duration_s: .75});
  for (let i = 0; i < count; i++) {
    h.dispatch("swarm_episode_begin", {episode_id: `e${i}`, training_step_start: i * 1000,
      sample_count: samples.length, lap_time_s: 1 + i * .01, color: "#ff8000"});
    h.dispatch("swarm_chunk", {episode_id: `e${i}`, ...structuredClone(chunk)});
    h.dispatch("swarm_episode_commit", {episode_id: `e${i}`});
  }
  return h.dispatch("swarm_commit", {autoplay: true});
}

test("latest wheel telemetry, seek and fixed-camera audio preserve recorded values", () => {
  const h = harness();
  load(h, 2);
  h.bind(.25);
  const leader = h.cars.find((car) => car.withAudio && !car.disposed);
  const state = leader.states.at(-1);
  assert.equal(state.position.x, 5);
  assert.equal(state.wheelContact[0].position.x, 5);
  assert.equal(state.wheelContact[0].normal.y, 1);
  assert.equal(state.wheelSkidInfo[0], .2);
  assert.equal(state.frames, 250);
  assert.equal(leader.audioVolume, 1);
  assert.ok(h.cars.filter((car) => !car.withAudio).every((car) => car.audioVolume === 0));
  assert.equal(h.camera(), leader.cameraOrbit);
  h.dispatch("seek", {seconds: .25});
  assert.deepEqual(leader.states.at(-1), state);
  h.bind(2);
  const status = h.dispatch("status");
  assert.equal(status.visible_ghosts, 0);
  assert.equal(status.leader_episode_id, "e0");
  assert.equal(h.camera(), leader.cameraOrbit);
  h.dispatch("clear");
  assert.equal(h.camera(), h.originalCamera);
});

test("latest renderer accepts the supported 500-car workload without physics APIs", () => {
  const h = harness();
  load(h, 500);
  h.bind(.25);
  const status = h.dispatch("status");
  assert.equal(status.visible_ghosts, 500);
  assert.equal(status.leader_episode_id, "e0");
  const active = h.cars.filter((car) => !car.disposed);
  assert.equal(active.length, 500);
  assert.equal(active.filter((car) => car.withAudio).length, 1);
  assert.ok(active.every((car) => car.states.at(-1).position.x === 5));
  h.dispatch("clear");
});
