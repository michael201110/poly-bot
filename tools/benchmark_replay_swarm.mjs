import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import vm from "node:vm";

const manifest = JSON.parse(readFileSync(new URL("../pml-mod/manifest.json", import.meta.url), "utf8"));
const version = process.env.POLYBOT_BRIDGE_VERSION || manifest.latest["0.6.3"];
const rendererSource = readFileSync(process.env.POLYBOT_REPLAY_RENDERER ||
  new URL(`../pml-mod/${version}/replay_renderer.mjs`, import.meta.url), "utf8")
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
      getCarStyle: () => ({ clone() { return { ...this }; } }),
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
  const wheelRows = rows.map(() => Array.from({ length: 42 }, (_, column) =>
    column < 40 ? [1, 0.6, 0, 1.3, 0, 1, 0, 0.08, 0.01, 0.2][column % 10] : 0));
  const rowBytes = Buffer.byteLength(JSON.stringify({start: 0, samples: rows, wheel_states: wheelRows}));
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
      wheel_states: wheelRows.map((row) => [...row]),
    });
    dispatch("swarm_episode_commit", { episode_id: `bench:${index}` });
  }
  dispatch("swarm_commit", { autoplay: true });
  if (globalThis.gc) globalThis.gc();
  const heapAfter = process.memoryUsage().heapUsed;
  for (let frame = 0; frame < warmupFrames; frame += 1) bind(1 / 60);
  const timings = [];
  const cpuBefore = process.cpuUsage();
  const before = performance.now();
  for (let frame = warmupFrames; frame < frameCount; frame += 1) {
    const start = performance.now();
    bind(1 / 60);
    timings.push(performance.now() - start);
  }
  const elapsedMs = performance.now() - before;
  const measuredFrames = frameCount - warmupFrames;
  timings.sort((a, b) => a - b);
  const percentile = (fraction) => timings[Math.min(timings.length - 1, Math.floor(timings.length * fraction))];
  const cpu = process.cpuUsage(cpuBefore);
  return {
    ghosts: ghostCount,
    samples_per_ghost: sampleCount,
    total_samples: totalSamples,
    trajectory_json_mb: Number((ghostCount * rowBytes / 1024 / 1024).toFixed(2)),
    renderer_heap_delta_mb: Number(((heapAfter - heapBefore) / 1024 / 1024).toFixed(2)),
    median_ms: Number(percentile(0.5).toFixed(3)),
    p95_ms: Number(percentile(0.95).toFixed(3)),
    p99_ms: Number(percentile(0.99).toFixed(3)),
    process_cpu_ms: (cpu.user + cpu.system) / 1000,
    mock_renderer_avg_ms_per_measured_frame: Number(
      (elapsedMs / measuredFrames).toFixed(3),
    ),
  };
}

console.log(JSON.stringify({
  version,
  note: "Synthetic JS renderer logic with mocked native cars; no PolyTrack/GPU/FPS measurement.",
  cases: [50, 100, 250, 500].map(run),
}, null, 2));
