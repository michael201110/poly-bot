import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import vm from "node:vm";

const rendererSource = readFileSync(new URL("../pml-mod/0.1.41/replay_renderer.mjs", import.meta.url), "utf8")
  .replace("export function installPolyBotReplayRenderer", "function installPolyBotReplayRenderer");
const sampleCount = 120;
const frameCount = 180;
const warmupFrames = 30;

function run(ghostCount) {
  if (globalThis.gc) globalThis.gc();
  const heapBefore = process.memoryUsage().heapUsed;
  let clock = 0;
  const vmContext = {
    performance: { now: () => (clock += 0.001) },
    TextEncoder,
    structuredClone,
    console,
  };
  vmContext.globalThis = vmContext;
  vm.runInNewContext(`${rendererSource}; installPolyBotReplayRenderer();`, vmContext);
  const game = {
    owner: {},
    deltaSeconds: 0,
    player: {
      getCarState: () => ({}),
      getCarStyle: () => ({ clone: () => ({}) }),
    },
    createCar: () => ({
      setCarStyle() {},
      setOpacity() {},
      setVisible() {},
      setCarState() {},
      update() {},
      dispose() {},
    }),
  };
  const bind = (deltaSeconds) => {
    game.deltaSeconds = deltaSeconds;
    vmContext.__polybotBindVisualReplayRenderer(game);
  };
  const dispatch = (op, params) => {
    let response;
    vmContext.__polybotVisualReplayDispatch({ op, params }, (message) => {
      response = message;
    });
    if (!response?.ok) throw new Error(response?.error ?? `no response for ${op}`);
    return response.result;
  };
  bind(0);

  const rows = Array.from({ length: sampleCount }, (_, index) => [
    index, index * 0.03, index * 0.1, 0, 0, 0, 0, 0, 1,
  ]);
  const rowBytes = Buffer.byteLength(JSON.stringify({samples: rows}));
  const totalSamples = ghostCount * sampleCount;
  dispatch("swarm_begin", {
    ghost_count: ghostCount,
    total_sample_count: totalSamples,
    payload_bytes: ghostCount * rowBytes,
    speed: 1,
    opacity: 0.5,
    end_behavior: "fade",
    fade_duration_s: 0.75,
  });
  for (let index = 0; index < ghostCount; index += 1) {
    dispatch("swarm_episode_begin", {
      episode_id: `bench:${index}`,
      training_step_start: index * 1000,
      sample_count: sampleCount,
      color: "#ff0000",
    });
    dispatch("swarm_chunk", {
      episode_id: `bench:${index}`,
      start: 0,
      samples: rows.map((row) => [...row]),
    });
    dispatch("swarm_episode_commit", { episode_id: `bench:${index}` });
  }
  dispatch("swarm_commit", { autoplay: true });
  if (globalThis.gc) globalThis.gc();
  const heapAfter = process.memoryUsage().heapUsed;
  const before = performance.now();
  for (let frame = 0; frame < frameCount; frame += 1) {
    bind(frame < warmupFrames ? 0 : 1 / 60);
  }
  const elapsedMs = performance.now() - before;
  const measuredFrames = frameCount - warmupFrames;
  return {
    ghosts: ghostCount,
    samples_per_ghost: sampleCount,
    total_samples: totalSamples,
    trajectory_json_mb: Number((ghostCount * rowBytes / 1024 / 1024).toFixed(2)),
    renderer_heap_delta_mb: Number(((heapAfter - heapBefore) / 1024 / 1024).toFixed(2)),
    mock_renderer_avg_ms_per_frame: Number((elapsedMs / frameCount).toFixed(3)),
    mock_renderer_avg_ms_per_measured_frame: Number(
      (elapsedMs / measuredFrames).toFixed(3),
    ),
  };
}

console.log(JSON.stringify({
  note: "Synthetic JS renderer logic with mocked native cars; no PolyTrack/GPU/FPS measurement.",
  cases: [50, 100, 250, 500].map(run),
}, null, 2));
