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
      ghosts: [],
      loading: null,
      playbackSeconds: 0,
      playing: false,
      speed: 1,
      opacity: 0.5,
      endBehavior: "fade",
      fadeDuration: 0.75,
      renderFrames: 0,
      renderTimeMs: 0,
    };
    const replayMaxGhosts = 500;
    const replayMaxSamplesPerGhost = 500000;
    const replayMaxTotalSamples = 250000;
    const replayMaxPayloadBytes = 33554432;
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
    const replayClear = () => {
      replayState.playing = false;
      const loading = replayState.loading;
      replayState.loading = null;
      for (const ghost of replayState.ghosts) {
        if (ghost.car) ghost.car.dispose();
        ghost.car = null;
        ghost.samples = null;
      }
      if (loading) {
        for (const episode of loading.episodes.values()) {
          if (episode.car) episode.car.dispose();
          episode.car = null;
          episode.samples = null;
        }
      }
      replayState.ghosts.length = 0;
      replayState.playbackSeconds = 0;
      replayState.renderFrames = 0;
      replayState.renderTimeMs = 0;
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
    const replayResetIndex = (ghost, target) => {
      const samples = ghost.samples;
      let low = 1;
      let high = samples.length - 1;
      while (low < high) {
        const middle = (low + high) >>> 1;
        if (samples[middle][1] < target) low = middle + 1;
        else high = middle;
      }
      ghost.sampleIndex = low;
    };
    const replaySetGhostColor = (ghost) => {
      const style = replayState.context.player.getCarStyle().clone();
      style.primaryHex = Number.parseInt(ghost.color.slice(1), 16);
      ghost.car.setCarStyle(style);
    };
    const replayCreateGhostCar = (ghost) => {
      const base = replayState.context.player.getCarState();
      ghost.car = replayState.context.createCar(base);
      ghost.renderState = {
        ...base,
        frames: 0,
        hasStarted: true,
        finishFrames: null,
        position: ghost.position,
        quaternion: ghost.orientation,
        controls: { up: false, right: false, down: false, left: false, reset: false },
      };
      ghost.car.audioVolume = 0;
      ghost.car.isPaused = true;
      replaySetGhostColor(ghost);
      ghost.car.setOpacity(replayState.opacity);
      ghost.car.setVisible(false);
    };
    const replayBeginSwarm = (params) => {
      if (!Number.isSafeInteger(params.ghost_count) || params.ghost_count < 1 ||
          params.ghost_count > replayMaxGhosts) {
        throw new Error(`ghost_count must be from 1 to ${replayMaxGhosts}`);
      }
      if (!Number.isSafeInteger(params.total_sample_count) || params.total_sample_count < 1 ||
          params.total_sample_count > replayMaxTotalSamples) {
        throw new Error(`total_sample_count must be from 1 to ${replayMaxTotalSamples}`);
      }
      if (!Number.isSafeInteger(params.payload_bytes) || params.payload_bytes < 1 ||
          params.payload_bytes > replayMaxPayloadBytes) {
        throw new Error(`payload_bytes must be from 1 to ${replayMaxPayloadBytes}`);
      }
      if (!["disappear", "freeze", "fade"].includes(params.end_behavior)) {
        throw new Error("invalid end_behavior");
      }
      const speed = replayFinite(params.speed, "speed");
      const opacity = replayFinite(params.opacity, "opacity");
      const fadeDuration = replayFinite(params.fade_duration_s, "fade_duration_s");
      if (speed < 0.1 || speed > 8) throw new Error("speed must be from 0.1 to 8");
      if (opacity < 0 || opacity > 1) throw new Error("opacity must be from 0 to 1");
      if (fadeDuration < 0 || fadeDuration > 10) {
        throw new Error("fade_duration_s must be from 0 to 10");
      }
      replayClear();
      replayState.speed = speed;
      replayState.opacity = opacity;
      replayState.endBehavior = params.end_behavior;
      replayState.fadeDuration = fadeDuration;
      replayState.loading = {
        expectedGhosts: params.ghost_count,
        expectedSamples: params.total_sample_count,
        expectedBytes: params.payload_bytes,
        declaredSamples: 0,
        receivedSamples: 0,
        receivedBytes: 0,
        episodes: new Map(),
        openEpisode: null,
      };
      return { accepted: true, ghost_count: params.ghost_count };
    };
    const replayBeginEpisode = (params) => {
      const loading = replayState.loading;
      if (!loading || loading.openEpisode) throw new Error("no swarm is accepting an episode");
      if (typeof params.episode_id !== "string" || !params.episode_id ||
          params.episode_id.length > 128 || loading.episodes.has(params.episode_id)) {
        throw new Error("episode_id must be unique non-empty text of at most 128 characters");
      }
      if (!Number.isSafeInteger(params.training_step_start) || params.training_step_start < 0) {
        throw new Error("training_step_start must be a non-negative safe integer");
      }
      if (!Number.isSafeInteger(params.sample_count) || params.sample_count < 1 ||
          params.sample_count > replayMaxSamplesPerGhost) {
        throw new Error(`sample_count must be from 1 to ${replayMaxSamplesPerGhost}`);
      }
      if (loading.declaredSamples + params.sample_count > loading.expectedSamples) {
        throw new Error("episode declarations exceed total_sample_count");
      }
      const episode = {
        episodeId: params.episode_id,
        trainingStepStart: params.training_step_start,
        color: replayColor(params.color),
        expected: params.sample_count,
        samples: [],
        sampleIndex: 1,
        duration: 0,
        car: null,
        position: { x: 0, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
        lastOpacity: null,
        visible: false,
      };
      loading.episodes.set(episode.episodeId, episode);
      loading.openEpisode = episode;
      loading.declaredSamples += episode.expected;
      return { accepted: true, episode_id: episode.episodeId };
    };
    const replayAddChunk = (params) => {
      const loading = replayState.loading;
      const episode = loading && loading.openEpisode;
      if (!episode || (params.episode_id !== undefined && params.episode_id !== episode.episodeId) ||
          !Array.isArray(params.samples) || params.samples.length < 1 ||
          params.samples.length > 256 || params.start !== episode.samples.length) {
        throw new Error("invalid or out-of-order swarm replay chunk");
      }
      if (episode.samples.length + params.samples.length > episode.expected) {
        throw new Error("replay chunk exceeds declared sample_count");
      }
      const bytes = new TextEncoder().encode(JSON.stringify(params.samples)).length;
      if (loading.receivedBytes + bytes > loading.expectedBytes ||
          loading.receivedBytes + bytes > replayMaxPayloadBytes) {
        throw new Error("swarm chunks exceed declared payload_bytes");
      }
      const checked = [];
      let previous = episode.samples.at(-1);
      for (const sample of params.samples) {
        checked.push(replayValidateSample(sample, previous));
        previous = sample;
      }
      loading.receivedBytes += bytes;
      episode.samples.push(...checked);
      loading.receivedSamples += checked.length;
      return { episode_id: episode.episodeId, received: episode.samples.length };
    };
    const replayCommitEpisode = (params = {}) => {
      const loading = replayState.loading;
      const episode = loading && loading.openEpisode;
      if (!episode || (params.episode_id !== undefined && params.episode_id !== episode.episodeId) ||
          episode.samples.length !== episode.expected) {
        throw new Error("episode payload is incomplete");
      }
      episode.duration = episode.samples.at(-1)[1] - episode.samples[0][1];
      if (replayState.context) replayCreateGhostCar(episode);
      loading.openEpisode = null;
      return { episode_id: episode.episodeId, committed: true };
    };
    const replayCommitSwarm = (params) => {
      const loading = replayState.loading;
      if (!loading || loading.openEpisode || loading.episodes.size !== loading.expectedGhosts ||
          loading.declaredSamples !== loading.expectedSamples ||
          loading.receivedSamples !== loading.expectedSamples ||
          loading.receivedBytes > loading.expectedBytes) {
        throw new Error("swarm payload is incomplete");
      }
      try {
        for (const episode of loading.episodes.values()) {
          if (!episode.car) replayCreateGhostCar(episode);
          replayState.ghosts.push(episode);
        }
      } catch (error) {
        replayClear();
        throw error;
      }
      replayState.loading = null;
      replayState.playbackSeconds = 0;
      replayState.playing = params.autoplay === true;
      replayRender(0);
      return replayStatus();
    };
    const replayStatus = () => {
      let visible = 0;
      let minimumStep = null;
      let maximumStep = null;
      let duration = 0;
      for (const ghost of replayState.ghosts) {
        if (ghost.visible) visible += 1;
        minimumStep = minimumStep === null
          ? ghost.trainingStepStart : Math.min(minimumStep, ghost.trainingStepStart);
        maximumStep = maximumStep === null
          ? ghost.trainingStepStart : Math.max(maximumStep, ghost.trainingStepStart);
        duration = Math.max(duration, ghost.duration);
      }
      return {
        loaded_ghosts: replayState.ghosts.length,
        visible_ghosts: visible,
        playing: replayState.playing,
        playback_seconds: replayState.playbackSeconds,
        duration_seconds: duration,
        training_step_range: minimumStep === null ? null : [minimumStep, maximumStep],
        average_render_ms: replayState.renderFrames
          ? replayState.renderTimeMs / replayState.renderFrames : 0,
      };
    };
    const replaySlerpInto = (left, right, fraction, result) => {
      let ax = left[5], ay = left[6], az = left[7], aw = left[8];
      let bx = right[5], by = right[6], bz = right[7], bw = right[8];
      let dot = ax * bx + ay * by + az * bz + aw * bw;
      if (dot < 0) {
        dot = -dot;
        bx = -bx; by = -by; bz = -bz; bw = -bw;
      }
      let leftWeight;
      let rightWeight;
      if (dot > 0.9995) {
        leftWeight = 1 - fraction;
        rightWeight = fraction;
      } else {
        const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
        const inverseSin = 1 / Math.sin(angle);
        leftWeight = Math.sin((1 - fraction) * angle) * inverseSin;
        rightWeight = Math.sin(fraction * angle) * inverseSin;
      }
      let x = leftWeight * ax + rightWeight * bx;
      let y = leftWeight * ay + rightWeight * by;
      let z = leftWeight * az + rightWeight * bz;
      let w = leftWeight * aw + rightWeight * bw;
      const norm = 1 / Math.hypot(x, y, z, w);
      result.x = x * norm; result.y = y * norm; result.z = z * norm; result.w = w * norm;
    };
    const replayInterpolateInto = (ghost, elapsed, output) => {
      const samples = ghost.samples;
      const lastIndex = samples.length - 1;
      const first = samples[0];
      if (lastIndex === 0) {
        output.source = first;
        output.fraction = 0;
        output.left = first;
        output.right = first;
        return output;
      }
      const target = first[1] + elapsed;
      while (ghost.sampleIndex < lastIndex && samples[ghost.sampleIndex][1] < target) {
        ghost.sampleIndex += 1;
      }
      if (target <= first[1]) {
        output.source = first;
        output.fraction = 0;
        output.left = first;
        output.right = first;
        return output;
      }
      const right = samples[ghost.sampleIndex];
      const left = samples[ghost.sampleIndex - 1];
      const span = right[1] - left[1];
      output.left = left;
      output.right = right;
      output.fraction = span > 0 ? Math.max(0, Math.min(1, (target - left[1]) / span)) : 1;
      output.source = null;
      return output;
    };
    const replayRender = (deltaSeconds) => {
      if (!replayState.ghosts.length) return;
      const frameStart = performance.now();
      if (replayState.playing && Number.isFinite(deltaSeconds) && deltaSeconds > 0) {
        replayState.playbackSeconds += deltaSeconds * replayState.speed;
      }
      const interval = { source: null, left: null, right: null, fraction: 0 };
      for (const ghost of replayState.ghosts) {
        const elapsed = replayState.playbackSeconds;
        const ended = elapsed > ghost.duration;
        if (ended && replayState.endBehavior === "disappear") {
          if (ghost.visible) ghost.car.setVisible(false);
          ghost.visible = false;
          continue;
        }
        let opacity = replayState.opacity;
        if (ended && replayState.endBehavior === "fade") {
          opacity *= replayState.fadeDuration > 0
            ? Math.max(0, 1 - (elapsed - ghost.duration) / replayState.fadeDuration)
            : 0;
          if (opacity <= 0) {
            if (ghost.visible) ghost.car.setVisible(false);
            ghost.visible = false;
            continue;
          }
        }
        const localElapsed = ended ? ghost.duration : elapsed;
        replayInterpolateInto(ghost, localElapsed, interval);
        const left = interval.left;
        const right = interval.right;
        const fraction = interval.fraction;
        const source = interval.source;
        const orientation = ghost.orientation;
        const positionX = source ? source[2] : left[2] + fraction * (right[2] - left[2]);
        const positionY = source ? source[3] : left[3] + fraction * (right[3] - left[3]);
        const positionZ = source ? source[4] : left[4] + fraction * (right[4] - left[4]);
        if (source) {
          orientation.x = source[5]; orientation.y = source[6];
          orientation.z = source[7]; orientation.w = source[8];
        } else {
          replaySlerpInto(left, right, fraction, orientation);
        }
        if (!ghost.visible) ghost.car.setVisible(true);
        ghost.visible = true;
        if (ghost.lastOpacity !== opacity) {
          ghost.car.setOpacity(opacity);
          ghost.lastOpacity = opacity;
        }
        const elapsedAtTransform = source ? source[1] : left[1] + fraction * (right[1] - left[1]);
        ghost.position.x = positionX;
        ghost.position.y = positionY;
        ghost.position.z = positionZ;
        const state = ghost.renderState;
        state.frames = Math.round(elapsedAtTransform * 1000);
        state.position = ghost.position;
        state.quaternion = orientation;
        ghost.car.setCarState(state, false);
        ghost.car.isPaused = true;
        ghost.car.audioVolume = 0;
        ghost.car.update(deltaSeconds);
      }
      replayState.renderFrames += 1;
      replayState.renderTimeMs += performance.now() - frameStart;
    };
    const replayDispatch = (command) => {
      if (!command || typeof command.op !== "string" || !command.params ||
          typeof command.params !== "object" || Array.isArray(command.params)) {
        throw new Error("malformed visual replay command");
      }
      const params = command.params;
      switch (command.op) {
        case "begin": {
          const color = replayColor(params.color);
          replayBeginSwarm({
            ghost_count: 1,
            total_sample_count: params.sample_count,
            payload_bytes: params.payload_bytes ?? replayMaxPayloadBytes,
            speed: params.speed,
            opacity: params.opacity,
            end_behavior: params.end_behavior,
            fade_duration_s: params.fade_duration_s,
          });
          replayBeginEpisode({
            episode_id: "legacy",
            training_step_start: 0,
            sample_count: params.sample_count,
            color,
          });
          return { accepted: true, sample_count: params.sample_count };
        }
        case "chunk": {
          return replayAddChunk(params);
        }
        case "commit": {
          replayCommitEpisode();
          return replayCommitSwarm(params);
        }
        case "swarm_begin":
          return replayBeginSwarm(params);
        case "swarm_episode_begin":
          return replayBeginEpisode(params);
        case "swarm_chunk":
          return replayAddChunk(params);
        case "swarm_episode_commit":
          return replayCommitEpisode(params);
        case "swarm_commit":
          return replayCommitSwarm(params);
        case "play":
          if (!replayState.ghosts.length) throw new Error("no visual replay swarm is loaded");
          replayState.playing = true;
          return replayStatus();
        case "pause":
          replayState.playing = false;
          return replayStatus();
        case "restart":
          if (!replayState.ghosts.length) throw new Error("no visual replay swarm is loaded");
          replayState.playbackSeconds = 0;
          for (const ghost of replayState.ghosts) replayResetIndex(ghost, ghost.samples[0][1]);
          replayState.playing = true;
          replayRender(0);
          return replayStatus();
        case "seek": {
          if (!replayState.ghosts.length) throw new Error("no visual replay swarm is loaded");
          const seconds = replayFinite(params.seconds, "seconds");
          const duration = Math.max(...replayState.ghosts.map((ghost) => ghost.duration));
          if (seconds < 0 || seconds > duration) throw new Error(`seek seconds must be from 0 to ${duration}`);
          replayState.playbackSeconds = seconds;
          for (const ghost of replayState.ghosts) {
            replayResetIndex(ghost, ghost.samples[0][1] + Math.min(seconds, ghost.duration));
          }
          replayRender(0);
          return replayStatus();
        }
        case "speed": {
          const speed = replayFinite(params.value, "speed");
          if (speed < 0.1 || speed > 8) throw new Error("speed must be from 0.1 to 8");
          replayState.speed = speed;
          return replayStatus();
        }
        case "opacity": {
          const opacity = replayFinite(params.value, "opacity");
          if (opacity < 0 || opacity > 1) throw new Error("opacity must be from 0 to 1");
          replayState.opacity = opacity;
          for (const ghost of replayState.ghosts) {
            ghost.car.setOpacity(opacity);
            ghost.lastOpacity = opacity;
          }
          replayRender(0);
          return replayStatus();
        }
        case "color": {
          const color = replayColor(params.value);
          for (const ghost of replayState.ghosts) {
            ghost.color = color;
            replaySetGhostColor(ghost);
          }
          return replayStatus();
        }
        case "end_behavior":
          if (!["disappear", "freeze", "fade"].includes(params.value)) {
            throw new Error("invalid end_behavior");
          }
          replayState.endBehavior = params.value;
          replayRender(0);
          return replayStatus();
        case "fade_duration": {
          const duration = replayFinite(params.value, "fade_duration_s");
          if (duration < 0 || duration > 10) throw new Error("fade_duration_s must be from 0 to 10");
          replayState.fadeDuration = duration;
          replayRender(0);
          return replayStatus();
        }
        case "clear":
          replayClear();
          return replayStatus();
        case "status":
          return replayStatus();
        default:
          throw new Error(`unsupported visual replay command: ${command.op}`);
      }
    };
    globalThis.__polybotVisualReplayDispatch = (command, reply) => {
      if (!replayState.context && ![
        "begin", "chunk", "swarm_begin", "swarm_episode_begin", "swarm_chunk",
        "swarm_episode_commit",
      ].includes(command.op)) {
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
      token: 'update(e) {\n              const t = (0, R.gn)(this, jr, "m", ys).call(this);',
      func: `globalThis.__polybotBindVisualReplayRenderer({
        owner: this,
        deltaSeconds: e,
        player: (0, R.gn)(this, Fa, "f"),
        createCar: (state) => new U.A(
          null, state, null, null,
          (0, R.gn)(this, Zr, "f"),
          (0, R.gn)(this, $r, "f"),
          (0, R.gn)(this, Jr, "f"),
          (0, R.gn)(this, Qr, "f"),
          (0, R.gn)(this, ga, "f"),
          (0, R.gn)(this, na, "f"),
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
