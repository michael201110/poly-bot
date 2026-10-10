import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../pml-mod/0.1.43/worker_runtime.js", import.meta.url), "utf8");
function runtime() {
  const context = vm.createContext({
    BridgeError: Error, maxReferenceTicks: 600000, referenceSampleTicks: 20,
    hiddenCarId: -1, messageTypes: { CreateCar: 1, StartCar: 2, DeleteCar: 3 },
    car: { frames: 0 },
  });
  vm.runInContext(
    source.slice(source.indexOf("function finite("), source.indexOf("function findCar(")) +
    source.slice(source.indexOf("function buildReferenceLookahead("), source.indexOf("function buildTelemetry(")) +
    source.slice(source.indexOf("function addReferencePoint("), source.indexOf("function buildReference(ghost")) + `
      function internalDispatch(message) { if (message.messageType === 1) car.frames = 0; }
      function findCar() { return car; }
      function advanceCar(car, controls) {
        return { frames: car.frames, position: [0, 0, car.frames],
          quaternion: [0, 0, 0, 1], nextCheckpointIndex: 0,
          hasFinished: car.frames === 200 };
      }
      function decodeState(state) { return state; }
    `, context);
  return context;
}

test("saved transforms retain arc length, curvature and lookahead geometry", () => {
  const context = runtime();
  const points = Array.from({ length: 25 }, (_, index) => ({
    tick: index * 20, position_m: [index * 0.3, 0, index * 2],
    quaternion_xyzw: [0, 0, 0, 1], checkpoint_index: 0,
  }));
  context.line = { schema: "polybot.racing-line.v1", points };
  vm.runInContext(`
    saved = buildReferenceFromSavedLine(line, {});
    nativePoints = [];
    for (const sample of line.points) {
      addReferencePoint(nativePoints, { frames: sample.tick,
        position: sample.position_m, quaternion: sample.quaternion_xyzw,
        nextCheckpointIndex: sample.checkpoint_index }, {}, true);
    }
    native = finishReference(nativePoints);
  `, context);
  assert.ok(context.saved.length > 40);
  assert.ok(context.saved.points.some(point => point.s >= 20));
  assert.deepEqual(
    JSON.parse(JSON.stringify(context.saved)),
    JSON.parse(JSON.stringify(context.native)),
  );
});

test("saved controls reconstruct dense reference independently of coarse transforms", () => {
  const context = runtime();
  context.line = {
    schema: "polybot.racing-line.v1",
    points: Array.from({ length: 10 }, (_, tick) => ({ tick, position_m: [0, 0, 0] })),
    prefix_actions: Array.from({ length: 201 }, () => [0, 1, 0]),
  };
  vm.runInContext("saved = buildReferenceFromSavedLine(line, {});", context);
  assert.equal(context.saved.length, 200);
  assert.equal(context.saved.points.length, 11);
  assert.equal(context.saved.points[5].frame, 100);
  assert.equal(context.saved.points[5].expertAction.throttle, 1);
});

test("an incomplete recorded run cannot silently become a valid reference", () => {
  const context = runtime();
  context.actions = Array.from({ length: 100 }, () => [0, 1, 0]);
  assert.throws(() => vm.runInContext("buildReferenceFromControls(actions, {});", context));
});

test("finish lookahead stays full and points forward beyond the recorded endpoint", () => {
  const context = runtime();
  context.line = {
    schema: "polybot.racing-line.v1",
    points: Array.from({ length: 25 }, (_, index) => ({
      tick: index * 20, position_m: [index * 2, 0, 0],
      quaternion_xyzw: [0, 0, 0, 1], checkpoint_index: 0,
    })),
  };
  vm.runInContext(`
    saved = buildReferenceFromSavedLine(line, {});
    before = JSON.stringify(saved);
    basis = { forward: [1, 0, 0], right: [0, 0, -1], up: [0, 1, 0] };
    approaching = buildReferenceLookahead(saved, 20, [40, 0, 0], basis, 12);
    atFinish = buildReferenceLookahead(saved, 24, [48, 0, 0], basis, 12);
    after = JSON.stringify(saved);
  `, context);
  const approaching = JSON.parse(JSON.stringify(context.approaching));
  const atFinish = JSON.parse(JSON.stringify(context.atFinish));
  assert.deepEqual(approaching.lookaheadMask, Array(12).fill(1));
  assert.deepEqual(atFinish.lookaheadMask, Array(12).fill(1));
  // Existing samples keep their original geometry; only missing ones extend.
  assert.equal(approaching.lookahead[0][0], 6);
  assert.deepEqual(approaching.lookahead.slice(1).map(p => p[0]),
    Array.from({ length: 11 }, (_, i) => (i + 2) * 5));
  assert.deepEqual(atFinish.lookahead.map(p => p[0]),
    Array.from({ length: 12 }, (_, i) => (i + 1) * 5));
  assert.ok(atFinish.lookahead.flat().every(Number.isFinite));
  assert.equal(context.before, context.after);
  assert.equal(context.saved.length, 48);
  assert.equal(context.saved.points.length, 25);
});
