// Exercise the manifest's current renderer as well as the historical 0.1.42
// suite. Native cars are mocked; this is not a GPU/PolyTrack FPS benchmark.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const manifest = JSON.parse(readFileSync(new URL("../pml-mod/manifest.json", import.meta.url)));
const source = readFileSync(new URL(`../pml-mod/${manifest.latest["0.6.3"]}/replay_renderer.mjs`, import.meta.url), "utf8")
  .replace("export function installPolyBotReplayRenderer", "function installPolyBotReplayRenderer");

function harness({ renderer = source, batching = false } = {}) {
  const cars = [];
  let camera = {};
  const originalCamera = camera;
  const context = { performance, TextEncoder, structuredClone, console };
  context.globalThis = context;
  vm.runInNewContext(`${renderer}; installPolyBotReplayRenderer()`, context);
  const batches = [];
  const material = name => ({name, opacity: 1, transparent: false,
    toJSON() { return {uuid: `${name}-per-car`, name, opacity: this.opacity, transparent: this.transparent}; },
    clone() { return {...this}; }, dispose() { this.disposed = true; }});
  class InstancedMesh {
    constructor(geometry, material, count) {
      Object.assign(this, {geometry, material, count, layers: {mask: 1},
        instanceMatrix: {array: new Float32Array(16 * count)}, dispose() { this.disposed = true; }});
      batches.push(this);
    }
  }
  const game = {
    owner: {}, deltaSeconds: 0,
    player: { getCarState: () => ({}), setVisible() {},
      getCarStyle: () => ({ clone() { return { ...this }; } }) },
    getCamera: () => camera, setCamera: (value) => { camera = value; },
    createCar: (_base, withAudio) => {
      const part = (id, mat) => ({isMesh: true, visible: true, layers: {mask: 1},
        castShadow: true, receiveShadow: true, geometry: {uuid: id}, material: mat,
        matrixWorld: {elements: [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]}});
      const parts = [part('metal', material('Metal')),
        part('wheel', [material('Tire'), material('Rim')]), part('body', material('Main'))];
      const root = {traverse(fn) { parts.forEach(fn); }, updateWorldMatrix() {
        for (const part of parts) part.matrixWorld.elements[12] = car.states.at(-1)?.position.x ?? 0;
      }};
      const car = { cameraOrbit: {}, states: [], withAudio, disposed: false, setCarStyle() {}, setOpacity() {},
        setVisible() {}, update() {}, updateCameras() {}, dispose() { this.disposed = true; },
        setCarState(state) { this.states.push(structuredClone(state)); }, parts,
        polybotReplayRoot: batching ? root : undefined };
      cars.push(car);
      return car;
    },
  };
  if (batching) {
    // The real namespace uses ZLX and the marker lives on instances, not prototypes.
    game.three = {ZLX: InstancedMesh};
    game.scene = {add() {}, remove() {}, traverse() { throw Error('per-frame scene traversal'); }};
  }
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
  return { cars, bind, dispatch, camera: () => camera, originalCamera, batches };
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

test("native instancing preserves every recorded car state and falls back for per-car fades", () => {
  const oldSource = readFileSync(new URL('../pml-mod/0.1.44/replay_renderer.mjs', import.meta.url), 'utf8')
    .replace('export function installPolyBotReplayRenderer', 'function installPolyBotReplayRenderer');
  const before = harness({renderer: oldSource, batching: true});
  const after = harness({batching: true});
  load(before, 3); load(after, 3);
  assert.equal(before.dispatch('status').instanced_batches, 0);
  assert.equal(after.dispatch('status').instanced_batches, 2);
  const buffers = after.batches.map(b => b.instanceMatrix.array);
  const active = h => h.cars.filter(c => !c.disposed);
  for (const seconds of [.125, .25, .5, .25]) {
    before.bind(seconds); after.bind(seconds);
    assert.deepEqual(active(after).map(c => c.states.at(-1)), active(before).map(c => c.states.at(-1)));
    assert.equal(after.camera(), active(after).find(c => c.withAudio).cameraOrbit);
  }
  assert.ok(after.batches.every(b => b.count === 0)); // Individually fading native draws.
  assert.ok(active(after).every(c => c.parts.every(p => p.visible)));
  before.dispatch('seek', {seconds: .5}); after.dispatch('seek', {seconds: .5});
  assert.deepEqual(active(after).map(c => c.states.at(-1)), active(before).map(c => c.states.at(-1)));
  for (let i = 0; i < buffers.length; i++) {
    assert.equal(after.batches[i].instanceMatrix.array, buffers[i]);
    assert.equal(after.batches[i].count, 3);
    assert.equal(buffers[i][12], 10);
  }
  after.dispatch('opacity', {value: .5});
  assert.ok(after.batches.every(b => b.count === 0));
  after.dispatch('opacity', {value: 1});
  assert.ok(after.batches.every(b => b.count === 3));
  after.dispatch('clear');
  assert.ok(after.batches.every(b => b.disposed));
});

test("new rendering release changes neither simulation worker nor physics integration hooks", () => {
  for (const file of ['worker_runtime.js', 'main.template.js']) {
    assert.deepEqual(readFileSync(new URL(`../pml-mod/0.1.45/${file}`, import.meta.url)),
      readFileSync(new URL(`../pml-mod/0.1.44/${file}`, import.meta.url)));
  }
});
