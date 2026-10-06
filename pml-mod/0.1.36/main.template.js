import {
  MixinType,
  PolyMod,
} from "https://cdn.polymodloader.com/pml/PolyModLoader/0.6.2/PolyTypes.js";

/* POLYBOT_WORKER_RUNTIME */

class PolyBotBridgeMod extends PolyMod {
  touchingPhysics = true;

  preInit = (pml) => {
    if (typeof pml?.registerSimWorkerMixin !== "function") {
      throw new Error(
        "PolyBot Bridge requires PolyModLoader 0.6.2 with simulation-worker mixins.",
      );
    }

    const requestedPort = Number(new URL(location.href).searchParams.get("polybotPort") ?? 8765);
    const bridgePort = Number.isInteger(requestedPort) && requestedPort > 0 && requestedPort < 65536
      ? requestedPort : 8765;
    // Offline training workers are independent per port. Keep the native
    // one-instance guard within each port instead of blocking every simulator.
    pml.registerGlobalMixin({
      type: MixinType.REPLACEBETWEEN,
      tokenStart: '"polytrack-single-instance"',
      tokenEnd: '"polytrack-single-instance"',
      func: JSON.stringify("polytrack-single-instance-polybot-" + bridgePort),
    });
    globalThis.__polybotWrapSimulationWorker = (worker) => {
      const originalPostMessage = worker.postMessage.bind(worker);
      worker.postMessage = (message, ...options) => {
        const tagged = message && typeof message === "object" &&
          !ArrayBuffer.isView(message) && !(message instanceof ArrayBuffer)
          ? { ...message, polybotBridgePort: bridgePort } : message;
        return originalPostMessage(tagged, ...options);
      };
      let restartScheduled = false;
      const pressBackspace = () => {
        const eventOptions = {
          key: "Backspace", code: "Backspace", keyCode: 8, which: 8,
          bubbles: true, cancelable: true,
        };
        const keyboardTarget = document.body ?? document;
        keyboardTarget.dispatchEvent(new KeyboardEvent("keydown", eventOptions));
        keyboardTarget.dispatchEvent(new KeyboardEvent("keyup", eventOptions));
      };
      worker.addEventListener("message", (event) => {
        const command = event.data?.polybotVisualReplayCommand;
        if (command) {
          const reply = (message) => originalPostMessage({
            polybotVisualReplayReply: { id: command.id, ...message },
          });
          try {
            globalThis.__polybotVisualReplayDispatch(command, reply);
          } catch (error) {
            reply({ ok: false, error: error instanceof Error ? error.message : String(error) });
          }
        }
        const playerFinished = event.data?.polybotPlayerFinished === true;
        if (
          (event.data?.polybotAbortRestart !== true && !playerFinished) || restartScheduled
        ) return;
        restartScheduled = true;
        const restartDelayMs = playerFinished ? 500 : 0;
        setTimeout(() => { pressBackspace(); restartScheduled = false; }, restartDelayMs);
      });
      return worker;
    };
    const replayState = {
      context: null,
      car: null,
      samples: null,
      loading: null,
      playbackSeconds: 0,
      playing: false,
      speed: 1,
      opacity: 0.5,
      lastRenderedOpacity: null,
      color: "#ffffff",
      endBehavior: "freeze",
      fadeDuration: 0.75,
    };
    const replayPendingCommands = [];
    const replayFinite = (value, label) => {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`${label} must be finite`);
      }
      return value;
    };
    const replayColor = (value) => {
      if (typeof value !== "string" || !/^#[0-9a-fA-F]{6}$/.test(value)) {
        throw new Error("color must be #RRGGBB");
      }
      return value.toLowerCase();
    };
    const replaySetColor = () => {
      if (!replayState.car || !replayState.context) return;
      const style = replayState.context.player.getCarStyle().clone();
      style.primaryHex = Number.parseInt(replayState.color.slice(1), 16);
      replayState.car.setCarStyle(style);
    };
    const replayClear = () => {
      replayState.playing = false;
      replayState.loading = null;
      replayState.samples = null;
      replayState.playbackSeconds = 0;
      replayState.lastRenderedOpacity = null;
      if (replayState.car) replayState.car.dispose();
      replayState.car = null;
    };
    const replayValidateSample = (sample, previous) => {
      if (!Array.isArray(sample) || sample.length !== 9) {
        throw new Error("each replay sample must contain tick, elapsed_s, position, and quaternion");
      }
      if (!sample.every((value) => typeof value === "number" && Number.isFinite(value))) {
        throw new Error("replay sample values must be finite numbers");
      }
      if (!Number.isSafeInteger(sample[0]) || sample[0] < 0 || sample[1] < 0) {
        throw new Error("replay sample tick and elapsed_s must be non-negative");
      }
      const quaternionNorm = Math.hypot(sample[5], sample[6], sample[7], sample[8]);
      if (quaternionNorm < 1e-6) throw new Error("replay quaternion cannot be zero");
      for (let index = 5; index < 9; index += 1) sample[index] /= quaternionNorm;
      if (previous && (sample[0] < previous[0] || sample[1] < previous[1])) {
        throw new Error("replay samples must be ordered by tick and elapsed_s");
      }
      return sample;
    };
    const replaySlerp = (left, right, fraction) => {
      let a = left.map((value) => value);
      let b = right.map((value) => value);
      const normA = Math.hypot(...a);
      const normB = Math.hypot(...b);
      a = a.map((value) => value / normA);
      b = b.map((value) => value / normB);
      let dot = a.reduce((sum, value, index) => sum + value * b[index], 0);
      if (dot < 0) {
        b = b.map((value) => -value);
        dot = -dot;
      }
      if (dot > 0.9995) {
        const value = a.map((component, index) => component + fraction * (b[index] - component));
        const norm = Math.hypot(...value);
        return value.map((component) => component / norm);
      }
      const angle = Math.acos(Math.min(1, Math.max(-1, dot)));
      const denominator = Math.sin(angle);
      return a.map((component, index) =>
        (Math.sin((1 - fraction) * angle) * component +
          Math.sin(fraction * angle) * b[index]) / denominator);
    };
    const replayTransform = () => {
      const samples = replayState.samples;
      const first = samples[0];
      const last = samples[samples.length - 1];
      const target = first[1] + replayState.playbackSeconds;
      if (target >= last[1]) {
        if (target > last[1] && replayState.endBehavior === "disappear") return null;
        let opacity = replayState.opacity;
        if (target > last[1] && replayState.endBehavior === "fade") {
          opacity *= Math.max(0, 1 - (target - last[1]) / replayState.fadeDuration);
          if (opacity <= 0) return null;
        }
        return { sample: last, opacity };
      }
      let rightIndex = 1;
      while (rightIndex < samples.length && samples[rightIndex][1] < target) rightIndex += 1;
      const left = samples[rightIndex - 1];
      const right = samples[rightIndex];
      const span = right[1] - left[1];
      const fraction = span > 0 ? (target - left[1]) / span : 1;
      const position = [2, 3, 4].map((index) => left[index] + fraction * (right[index] - left[index]));
      const quaternion = replaySlerp(left.slice(5, 9), right.slice(5, 9), fraction);
      return {
        sample: [left[0], target, ...position, ...quaternion],
        opacity: replayState.opacity,
      };
    };
    const replayRender = (deltaSeconds) => {
      if (!replayState.car || !replayState.samples) return;
      if (replayState.playing && Number.isFinite(deltaSeconds) && deltaSeconds > 0) {
        replayState.playbackSeconds += deltaSeconds * replayState.speed;
      }
      const transform = replayTransform();
      if (!transform) {
        replayState.car.setVisible(false);
        return;
      }
      const base = replayState.context.player.getCarState();
      const sample = transform.sample;
      replayState.car.setVisible(true);
      if (replayState.lastRenderedOpacity !== transform.opacity) {
        replayState.car.setOpacity(transform.opacity);
        replayState.lastRenderedOpacity = transform.opacity;
      }
      replayState.car.setCarState({
        ...base,
        frames: Math.round(sample[1] * 1000),
        hasStarted: true,
        finishFrames: null,
        position: { x: sample[2], y: sample[3], z: sample[4] },
        quaternion: { x: sample[5], y: sample[6], z: sample[7], w: sample[8] },
        controls: { up: false, right: false, down: false, left: false, reset: false },
      }, false);
      replayState.car.isPaused = true;
      replayState.car.audioVolume = 0;
      replayState.car.update(deltaSeconds);
    };
    const replayDispatch = (command) => {
      if (!command || typeof command.op !== "string" || !command.params ||
          typeof command.params !== "object" || Array.isArray(command.params)) {
        throw new Error("malformed visual replay command");
      }
      const params = command.params;
      switch (command.op) {
        case "begin": {
          if (!Number.isSafeInteger(params.sample_count) || params.sample_count < 1 ||
              params.sample_count > 500000) throw new Error("sample_count must be from 1 to 500000");
          if (params.end_behavior !== "disappear" && params.end_behavior !== "freeze" &&
              params.end_behavior !== "fade") throw new Error("invalid end_behavior");
          const color = replayColor(params.color);
          const speed = replayFinite(params.speed, "speed");
          const opacity = replayFinite(params.opacity, "opacity");
          const fadeDuration = replayFinite(params.fade_duration_s, "fade_duration_s");
          if (speed < 0.1 || speed > 8) throw new Error("speed must be from 0.1 to 8");
          if (opacity < 0 || opacity > 1) throw new Error("opacity must be from 0 to 1");
          if (fadeDuration < 0 || fadeDuration > 10) {
            throw new Error("fade_duration_s must be from 0 to 10");
          }
          replayClear();
          replayState.color = color;
          replayState.speed = speed;
          replayState.opacity = opacity;
          replayState.fadeDuration = fadeDuration;
          replayState.endBehavior = params.end_behavior;
          replayState.loading = { expected: params.sample_count, samples: [] };
          return { accepted: true, sample_count: params.sample_count };
        }
        case "chunk": {
          if (!replayState.loading || !Array.isArray(params.samples) ||
              params.samples.length < 1 || params.samples.length > 256 ||
              params.start !== replayState.loading.samples.length) {
            throw new Error("invalid or out-of-order replay chunk");
          }
          if (replayState.loading.samples.length + params.samples.length > replayState.loading.expected) {
            throw new Error("replay chunk exceeds declared sample_count");
          }
          for (const sample of params.samples) {
            const previous = replayState.loading.samples.at(-1);
            replayState.loading.samples.push(replayValidateSample(sample, previous));
          }
          return { received: replayState.loading.samples.length };
        }
        case "commit": {
          if (!replayState.loading ||
              replayState.loading.samples.length !== replayState.loading.expected) {
            throw new Error("replay payload is incomplete");
          }
          replayState.samples = replayState.loading.samples;
          replayState.loading = null;
          replayState.playbackSeconds = 0;
          replayState.car = replayState.context.createCar(replayState.context.player.getCarState());
          replaySetColor();
          replayState.car.setOpacity(replayState.opacity);
          replayState.car.audioVolume = 0;
          replayState.car.isPaused = true;
          replayRender(0);
          replayState.playing = params.autoplay === true;
          return { loaded: true, playing: replayState.playing };
        }
        case "play":
          if (!replayState.samples || !replayState.car) throw new Error("no visual replay is loaded");
          replayState.playing = true;
          return { playing: true };
        case "pause":
          replayState.playing = false;
          return { playing: false };
        case "restart":
          if (!replayState.samples || !replayState.car) throw new Error("no visual replay is loaded");
          replayState.playbackSeconds = 0;
          replayState.playing = true;
          return { playing: true, playback_seconds: 0 };
        case "seek": {
          if (!replayState.samples || !replayState.car) throw new Error("no visual replay is loaded");
          const seconds = replayFinite(params.seconds, "seconds");
          const duration = replayState.samples.at(-1)[1] - replayState.samples[0][1];
          if (seconds < 0 || seconds > duration) throw new Error(`seek seconds must be from 0 to ${duration}`);
          replayState.playbackSeconds = seconds;
          replayRender(0);
          return { playback_seconds: seconds };
        }
        case "speed": {
          const speed = replayFinite(params.value, "speed");
          if (speed < 0.1 || speed > 8) throw new Error("speed must be from 0.1 to 8");
          replayState.speed = speed;
          return { speed };
        }
        case "opacity": {
          const opacity = replayFinite(params.value, "opacity");
          if (opacity < 0 || opacity > 1) throw new Error("opacity must be from 0 to 1");
          replayState.opacity = opacity;
          if (replayState.car) replayState.car.setOpacity(opacity);
          replayState.lastRenderedOpacity = opacity;
          return { opacity };
        }
        case "color":
          replayState.color = replayColor(params.value);
          replaySetColor();
          return { color: replayState.color };
        case "end_behavior":
          if (!["disappear", "freeze", "fade"].includes(params.value)) {
            throw new Error("invalid end_behavior");
          }
          replayState.endBehavior = params.value;
          return { end_behavior: params.value };
        case "fade_duration": {
          const duration = replayFinite(params.value, "fade_duration_s");
          if (duration < 0 || duration > 10) throw new Error("fade_duration_s must be from 0 to 10");
          replayState.fadeDuration = duration;
          return { fade_duration_s: duration };
        }
        case "clear":
          replayClear();
          return { cleared: true };
        default:
          throw new Error(`unsupported visual replay command: ${command.op}`);
      }
    };
    globalThis.__polybotVisualReplayDispatch = (command, reply) => {
      if (!replayState.context && command.op !== "begin" && command.op !== "chunk") {
        replayPendingCommands.push({ command, reply });
        return;
      }
      try {
        reply({ ok: true, result: replayDispatch(command) });
      } catch (error) {
        reply({ ok: false, error: error instanceof Error ? error.message : String(error) });
      }
    };
    globalThis.__polybotBindVisualReplayRenderer = (context) => {
      if (replayState.context && replayState.context.owner !== context.owner) replayClear();
      replayState.context = context;
      while (replayPendingCommands.length) {
        const { command, reply } = replayPendingCommands.shift();
        try {
          reply({ ok: true, result: replayDispatch(command) });
        } catch (error) {
          reply({ ok: false, error: error instanceof Error ? error.message : String(error) });
        }
      }
      replayRender(context.deltaSeconds);
    };

    pml.registerSimWorkerMixin({
      type: MixinType.INSERT,
      token: "const r = i.data;",
      func: polybotWorkerInjection,
    });
    pml.registerGlobalMixin({
      type: MixinType.REPLACEBETWEEN,
      tokenStart: "new Worker(ActivePolyModLoader.getSimURL())",
      tokenEnd: "new Worker(ActivePolyModLoader.getSimURL())",
      func: "globalThis.__polybotWrapSimulationWorker(new Worker(ActivePolyModLoader.getSimURL()))",
    });
    pml.registerGlobalMixin({
      type: MixinType.REPLACEBETWEEN,
      tokenStart: "this.setCarState(e, !1);",
      tokenEnd: "this.setCarState(e, !1);",
      // The worker recreates cars on reset. Clear the local player's recorder
      // and prime its previous state before recording the first new tick.
      // The native recorder uses the PREVIOUS state's frame and finish flag.
      func: `{
        const previous = this.getCarState();
        const rewind = e.frames <= previous.frames;
        if (rewind && (0, l.gn)(this, ie, "f")) {
          (0, l.GG)(this, re, new st.A(), "f");
          (0, l.GG)(this, te, {
            ...previous,
            frames: Math.max(0, e.frames - 1),
            finishFrames: null,
            nextCheckpointIndex: 0,
            controls: { up: false, right: false, down: false, left: false, reset: false },
          }, "f");
        }
        this.setCarState(e, rewind);
      }`,
    });
    pml.registerGlobalMixin({
      type: MixinType.INSERT,
      token: 'update(e) {\n              const t = (0, R.gn)(this, ta, "m", Ps).call(this);',
      func: `globalThis.__polybotBindVisualReplayRenderer({
        owner: this,
        deltaSeconds: e,
        player: (0, R.gn)(this, Xa, "f"),
        createCar: (state) => new z.A(
          null, state, null, null,
          (0, R.gn)(this, la, "f"),
          (0, R.gn)(this, ca, "f"),
          (0, R.gn)(this, aa, "f"),
          (0, R.gn)(this, ra, "f"),
          (0, R.gn)(this, Ta, "f"),
          (0, R.gn)(this, ua, "f"),
          null,
        ),
      });`,
    });
    for (const token of [
      "submitLeaderboard(e, t, n, i, r, a, s, o) {",
      "submitUserProfile(e, t, n, i) {",
      "verifyRecordings(e, t, n, i, r) {",
      "getIceServers() {",
    ]) {
      pml.registerGlobalMixin({
        type: MixinType.INSERT, token,
        func: 'return Promise.reject(new Error("PolyBot offline mode"));',
      });
    }
    for (const token of [
      "createMultiplayerHostWebSocket() {", "createMultiplayerJoinWebSocket() {",
    ]) {
      pml.registerGlobalMixin({
        type: MixinType.INSERT, token,
        func: 'throw new Error("PolyBot offline mode");',
      });
    }
  };
}

export const polyMod = new PolyBotBridgeMod();
