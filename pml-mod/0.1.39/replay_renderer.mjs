export function installPolyBotReplayRenderer() {
    const replayState = {
      context: null,
      ghosts: [],
      loading: null,
      playbackSeconds: 0,
      playing: false,
      speed: 1,
      opacity: 1,
      endBehavior: "fade",
      fadeDuration: 0.75,
      renderFrames: 0,
      renderTimeMs: 0,
      camera: null,
      previousCamera: null,
      leader: null,
      groundSampler: null,
      carStyle: null,
      lastHudFrame: null,
      lastPlaybackSeconds: null,
      dirty: true,
      endTime: 0,
      durationSeconds: 0,
      batches: [],
      batchedParts: 0,
    };
    const replayInterval = { source: null, left: null, right: null, fraction: 0 };
    const frameProfile = [];
    let previousFrameAt = null;
    const replayMaxGhosts = 500;
    const replayMaxSamplesPerGhost = 500000;
    const replayMaxTotalSamples = 250000;
    const replayMaxPayloadBytes = 67108864;
    const replayPendingCommands = [];
    const replayTextureEntries = new WeakMap();
    globalThis.__polybotReplaySharedTexture = {
      lookup(renderer, pattern) {
        const entry = replayTextureEntries.get(renderer)?.get(pattern);
        if (!entry) return null;
        entry.references += 1;
        return entry.texture;
      },
      register(renderer, pattern, texture) {
        let entries = replayTextureEntries.get(renderer);
        if (!entries) replayTextureEntries.set(renderer, entries = new Map());
        let entry = entries.get(pattern);
        if (entry) { entry.references += 1; texture.dispose(); return entry.texture; }
        entry = { texture, references: 1 };
        entries.set(pattern, entry);
        const dispose = texture.dispose.bind(texture);
        texture.dispose = () => {
          if (entry.references <= 0) return;
          if (--entry.references === 0) { entries.delete(pattern); dispose(); }
        };
        return texture;
      },
    };
    globalThis.__polybotReplayCaptureRoot = (_car, root) => {
      if (globalThis.__polybotReplayTextureSharing) globalThis.__polybotReplayPendingRoot = root;
    };
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
    const replayClear = ({ restoreCamera = true } = {}) => {
      for (const batch of replayState.batches) {
        replayState.context?.scene?.remove(batch.mesh);
        replayState.context?.removeMaterial?.(batch.mesh.material);
        batch.mesh.material.dispose();
        batch.mesh.dispose();
      }
      replayState.batches.length = 0;
      replayState.batchedParts = 0;
      globalThis.__polybotHudUpdate?.clear?.("replay");
      replayState.playing = false;
      if (restoreCamera && replayState.camera && replayState.context) {
        replayState.context.setCamera(replayState.previousCamera);
        replayState.context.player.setVisible(true);
      }
      replayState.camera = null;
      replayState.previousCamera = null;
      replayState.leader = null;
      replayState.lastHudFrame = null;
      replayState.lastPlaybackSeconds = null;
      replayState.dirty = true;
      replayState.endTime = 0;
      replayState.durationSeconds = 0;
      replayState.groundSampler = null;
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
      ghost.cameraNeedsReset = true;
    };
    const replaySetGhostColor = (ghost) => {
      // Use one stable, clean stock design across the whole replay group.
      // Training-age colour remains on the body; both paint channels match.
      if (!replayState.carStyle) {
        const style = replayState.context.player.getCarStyle().clone();
        style.pattern = 0;
        style.rims = 0;
        style.exhaust = 0;
        style.frameHex = 0x252b35;
        style.rimsHex = 0xf4f6fa;
        replayState.carStyle = style;
      }
      const style = replayState.carStyle.clone();
      style.primaryHex = Number.parseInt(ghost.color.slice(1), 16);
      style.secondaryHex = style.primaryHex;
      globalThis.__polybotReplayTextureSharing = true;
      try { ghost.car.setCarStyle(style); }
      finally { globalThis.__polybotReplayTextureSharing = false; }
    };
    const replayCreateGhostCar = (ghost, withAudio = false) => {
      // Native cars retain their state. Give every replay its own full copy,
      // including wheel arrays and controls, rather than sharing player state.
      const base = structuredClone(replayState.context.player.getCarState());
      ghost.car = replayState.context.createCar(base, withAudio);
      ghost.car.getCarState?.();
      ghost.renderState = {
        ...base,
        frames: 0,
        hasStarted: true,
        finishFrames: null,
        position: ghost.position,
        quaternion: ghost.orientation,
        wheelContact: [null, null, null, null],
        wheelSuspensionLength: [...(base.wheelSuspensionLength || [0.0781, 0.0781, 0.0781, 0.0781])],
        wheelSuspensionVelocity: [0, 0, 0, 0],
        wheelDeltaRotation: [0, 0, 0, 0],
        wheelSkidInfo: [1, 1, 1, 1],
        controls: { up: false, right: false, down: false, left: false, reset: false },
        collisionImpulses: [],
      };
      ghost.car.audioVolume = 0;
      ghost.lastAudioVolume = 0;
      ghost.car.isPaused = true;
      replaySetGhostColor(ghost);
      ghost.car.setOpacity(replayState.opacity);
      ghost.car.setVisible(false);
    };
    const replayApplyWheels = (state, left, right, fraction) => {
      for (let index = 0; index < 4; index += 1) {
        const offset = index * 10;
        const current = fraction < 0.5 ? left : right;
        const interpolate = (column) => left[offset + column]
          + fraction * (right[offset + column] - left[offset + column]);
        if (current[offset]) {
          const smooth = left[offset] && right[offset];
          const value = (column) => smooth ? interpolate(column) : current[offset + column];
          const length = Math.hypot(value(4), value(5), value(6)) || 1;
          state.wheelContact[index] = {
            position: { x: value(1), y: value(2), z: value(3) },
            normal: { x: value(4) / length, y: value(5) / length, z: value(6) / length },
          };
        } else state.wheelContact[index] = null;
        state.wheelSuspensionLength[index] = interpolate(7);
        state.wheelDeltaRotation[index] = interpolate(8);
        state.wheelSkidInfo[index] = interpolate(9);
      }
      state.steering = left[40] + fraction * (right[40] - left[40]);
      state.brakeLightEnabled = (fraction < 0.5 ? left[41] : right[41]) === 1;
    };
    const replayGroundSampler = () => {
      const { three: T, scene } = replayState.context;
      if (!T || !scene) return null;
      const ray = new T.tBo();
      ray.near = 0; ray.far = 1.4;
      const grid = new Map();
      const wide = [];
      const cellSize = 20;
      const isCar = (object) => {
        for (let node = object; node; node = node.parent) if (node.userData?.polybotCar) return true;
        return false;
      };
      // Index static road geometry once. Per-sample rays only visit nearby meshes.
      scene.updateMatrixWorld(true);
      const addMesh = (object) => {
        const box = object.geometry.boundingBox.clone().applyMatrix4(object.matrixWorld);
        const minX = Math.floor(box.min.x / cellSize), maxX = Math.floor(box.max.x / cellSize);
        const minZ = Math.floor(box.min.z / cellSize), maxZ = Math.floor(box.max.z / cellSize);
        if ((maxX - minX + 1) * (maxZ - minZ + 1) > 256) { wide.push(object); return; }
        for (let x = minX; x <= maxX; x += 1) for (let z = minZ; z <= maxZ; z += 1) {
          const key = `${x},${z}`;
          if (!grid.has(key)) grid.set(key, []);
          grid.get(key).push(object);
        }
      };
      scene.traverse((object) => {
        if (!object.isMesh || isCar(object) || !object.geometry ||
            (Array.isArray(object.material) ? object.material.every((m) => m.opacity === 0) : object.material?.opacity === 0)) return;
        object.geometry.computeBoundingBox();
        if (!object.geometry.boundingBox || object.geometry.boundingBox.isEmpty()) return;
        // Track tiles use instancing: geometry bounds alone sit at the origin.
        // Index each static instance at its world transform, sharing resources.
        if (object.isInstancedMesh) {
          const matrix = new T.kn4();
          for (let index = 0; index < object.count; index += 1) {
            object.getMatrixAt(index, matrix);
            const proxy = new T.eaF(object.geometry, object.material);
            proxy.matrixAutoUpdate = false;
            proxy.matrixWorld.multiplyMatrices(object.matrixWorld, matrix);
            proxy.layers.mask = object.layers.mask;
            addMesh(proxy);
          }
        } else addMesh(object);
      });
      return (sample) => {
        const q = new T.PTz(sample[5], sample[6], sample[7], sample[8]);
        const up = new T.Pq0(0, 1, 0).applyQuaternion(q);
        ray.ray.origin.set(sample[2], sample[3], sample[4]).addScaledVector(up, 0.25);
        ray.ray.direction.copy(up).negate();
        const nearby = grid.get(`${Math.floor(sample[2] / cellSize)},${Math.floor(sample[4] / cellSize)}`) || [];
        const hits = ray.intersectObjects([...nearby, ...wide], false);
        const hit = hits.find((item) => item.face && !isCar(item.object));
        if (!hit) return null;
        const normal = hit.face.normal.clone().transformDirection(hit.object.matrixWorld);
        if (normal.dot(up) < 0.5) return null;
        return { point: hit.point.clone(), normal, q, T };
      };
    };
    const replayLegacyWheels = (sample, previous, next) => {
      if (sample.legacyWheels !== undefined) return sample.legacyWheels;
      if (!replayState.groundSampler) replayState.groundSampler = replayGroundSampler();
      const surface = replayState.groundSampler?.(sample);
      const wheels = Array(42).fill(0);
      if (!surface) { sample.legacyWheels = wheels; return wheels; }
      const { T, point, normal, q } = surface;
      const dt = next[1] - previous[1];
      const velocity = new T.Pq0();
      if (dt > 0) velocity.set(next[2] - previous[2], next[3] - previous[3], next[4] - previous[4]).divideScalar(dt);
      const forward = new T.Pq0(0, 0, 1).applyQuaternion(q);
      const right = new T.Pq0(1, 0, 0).applyQuaternion(q);
      const forwardSpeed = velocity.dot(forward);
      const lateralSpeed = Math.abs(velocity.dot(right));
      // Older files contain transforms only. Estimate visual slip; never feed it into physics.
      const slipping = lateralSpeed > Math.max(0.65, Math.abs(forwardSpeed) * 0.02);
      const mounts = [[0.627909, 0.27, 1.3478], [-0.627909, 0.27, 1.3478],
        [0.720832, 0.27, -1.52686], [-0.720832, 0.27, -1.52686]];
      for (let index = 0; index < 4; index += 1) {
        const contact = new T.Pq0(...mounts[index]).applyQuaternion(q)
          .add(new T.Pq0(sample[2], sample[3], sample[4]));
        contact.addScaledVector(normal, -contact.clone().sub(point).dot(normal));
        wheels.splice(index * 10, 10, 1, contact.x, contact.y, contact.z,
          normal.x, normal.y, normal.z, 0.0781, forwardSpeed * 0.001 / 0.35, slipping ? 0 : 1);
      }
      sample.legacyWheels = wheels;
      return wheels;
    };
    const replayFollowLeader = (deltaSeconds) => {
      const context = replayState.context;
      if (!context.getCamera || !context.setCamera) return;
      const leader = replayState.leader;
      if (!leader) return;
      if (!replayState.camera) replayState.previousCamera = context.getCamera();
      // Use the game's own chase camera and smoothing, just as normal driving.
      leader.car.updateCameras(deltaSeconds);
      replayState.camera = leader.car.cameraOrbit;
      replayState.leader = leader;
      // Re-selecting the same camera rebuilds native shadow-camera bounds.
      if (context.getCamera() !== replayState.camera) context.setCamera(replayState.camera);
      context.player.setVisible(false);
      if (!leader.hudFrame) {
        replayState.lastHudFrame = null;
        globalThis.__polybotHudUpdate?.clear?.("replay");
      }
      if (leader.hudFrame && leader.hudFrame !== replayState.lastHudFrame &&
          typeof globalThis.__polybotHudUpdate === "function") {
        replayState.lastHudFrame = leader.hudFrame;
        globalThis.__polybotHudUpdate({
          ...leader.hudFrame,
          mode: replayState.ghosts.length > 1 ? "replay_swarm" : "replay",
        });
      }
    };
    const replayBuildBatches = () => {
      const context = replayState.context;
      const T = context?.three;
      const InstancedMesh = T?.InstancedMesh || Object.values(T || {}).find((value) =>
        typeof value === "function" && value.prototype?.isInstancedMesh === true);
      if (!InstancedMesh || !context?.scene) return;
      const groups = new Map();
      for (const ghost of replayState.ghosts) {
        const root = ghost.car?.polybotReplayRoot;
        if (!root) continue;
        root.traverse((object) => {
          const material = object.material;
          if (!object.isMesh || !object.geometry || !material || Array.isArray(material) ||
              material.name === "Main" || material.name === "BrakeLight") return;
          const key = [object.geometry.uuid, material.name, material.color?.getHex?.() ?? 0,
            material.map?.uuid ?? "", material.transparent, material.side].join(":");
          let group = groups.get(key);
          if (!group) groups.set(key, group = { geometry: object.geometry, material,
            items: [], castShadow: object.castShadow, receiveShadow: object.receiveShadow });
          group.items.push({ ghost, object });
        });
      }
      for (const group of groups.values()) {
        if (group.items.length < 2) continue;
        const material = group.material.clone();
        material.opacity = replayState.opacity;
        material.transparent = replayState.opacity < 1;
        const mesh = new InstancedMesh(group.geometry, material, group.items.length);
        mesh.frustumCulled = false;
        mesh.castShadow = group.castShadow;
        mesh.receiveShadow = group.receiveShadow;
        context.addMaterial?.(material);
        for (const { object } of group.items) object.visible = false;
        context.scene.add(mesh);
        replayState.batches.push({ mesh, items: group.items });
        replayState.batchedParts += group.items.length;
      }
      replayUpdateBatches();
    };
    const replayUpdateBatches = () => {
      const context = replayState.context;
      const Matrix4 = context?.three?.kn4;
      if (!replayState.batches.length || !context?.scene || !Matrix4) return;
      const matrix = new Matrix4();
      const hidden = new Matrix4().makeScale(0, 0, 0);
      for (const batch of replayState.batches) {
        for (let index = 0; index < batch.items.length; index += 1) {
          const { ghost, object } = batch.items[index];
          ghost.car.polybotReplayRoot?.updateWorldMatrix(true, true);
          batch.mesh.setMatrixAt(index, ghost.visible ? object.matrixWorld : hidden);
        }
        batch.mesh.instanceMatrix.needsUpdate = true;
      }
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
        finalProgressM: Number.isFinite(params.final_progress_m) ? params.final_progress_m : 0,
        lapTime: Number.isFinite(params.lap_time_s) && params.lap_time_s > 0 ? params.lap_time_s : Infinity,
        cameraNeedsReset: true,
        progressM: null,
        distanceM: 0,
        hudFrame: null,
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
      const chunkPayload = { start: params.start, samples: params.samples };
      if (params.wheel_states !== undefined) {
        if (!Array.isArray(params.wheel_states) || params.wheel_states.length !== params.samples.length ||
            params.wheel_states.some((values) => values !== null && (!Array.isArray(values) ||
              values.length !== 42 || values.some((value) => typeof value !== "number" || !Number.isFinite(value))))) {
          throw new Error("replay wheel states must match the transform samples and contain 42 finite numbers");
        }
        chunkPayload.wheel_states = params.wheel_states;
      }
      if (params.hud_frames !== undefined) {
        if (!Array.isArray(params.hud_frames) || params.hud_frames.length !== params.samples.length ||
            params.hud_frames.some((frame) => frame !== null && (
              !frame || typeof frame !== "object" || Array.isArray(frame) ||
              frame.schema !== "polybot.ai-overlay-frame.v1"
            ))) {
          throw new Error("replay HUD frames must match the transform samples");
        }
        chunkPayload.hud_frames = params.hud_frames;
      }
      const bytes = new TextEncoder().encode(JSON.stringify(chunkPayload)).length;
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
      if (params.hud_frames) {
        for (let index = 0; index < checked.length; index += 1) {
          checked[index].hudFrame = params.hud_frames[index];
        }
      }
      if (params.wheel_states) checked.forEach((sample, index) => { sample.wheelState = params.wheel_states[index]; });
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
      let distance = 0;
      episode.samples.forEach((sample, index) => {
        if (index) {
          const previous = episode.samples[index - 1];
          distance += Math.hypot(sample[2] - previous[2], sample[3] - previous[3], sample[4] - previous[4]);
        }
        sample.distanceM = distance;
      });
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
      // Pick once for the entire group; overtakes never change the camera car.
      replayState.leader = replayState.ghosts.reduce((best, ghost) => {
        if (!best || ghost.lapTime < best.lapTime ||
            (ghost.lapTime === best.lapTime && ghost.finalProgressM > best.finalProgressM)) return ghost;
        return best;
      }, null);
      // Native cars allocate looping engine/tire sources even at volume zero.
      // Only the fixed camera car needs an audio manager.
      replayState.leader.car.dispose();
      replayCreateGhostCar(replayState.leader, true);
      replayState.durationSeconds = replayState.ghosts.reduce((longest,ghost)=>Math.max(longest,ghost.duration),0);
      replayBuildBatches();
      replayState.dirty = true;
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
        instanced_batches: replayState.batches.length,
        batched_parts: replayState.batchedParts,
        training_step_range: minimumStep === null ? null : [minimumStep, maximumStep],
        average_render_ms: replayState.renderFrames
          ? replayState.renderTimeMs / replayState.renderFrames : 0,
        performance: {
          samples: frameProfile.length,
          frame_ms: frameProfile.length ? frameProfile.reduce((sum, row) => sum + row.frame, 0) / frameProfile.length : 0,
          replay_update_ms: frameProfile.length ? frameProfile.reduce((sum, row) => sum + row.replay, 0) / frameProfile.length : 0,
          renderer: frameProfile.length ? {
            calls: frameProfile.reduce((sum, row) => sum + (row.renderer?.calls ?? 0), 0) / frameProfile.length,
            triangles: frameProfile.reduce((sum, row) => sum + (row.renderer?.triangles ?? 0), 0) / frameProfile.length,
            render_cpu_ms: frameProfile.reduce((sum, row) => sum + (row.renderer?.render_cpu_ms ?? 0), 0) / frameProfile.length,
            width: frameProfile.at(-1).renderer?.width,
            height: frameProfile.at(-1).renderer?.height,
          } : null,
        },
        camera_mode: replayState.camera ? "best_run_chase" : "player",
        leader_episode_id: replayState.leader?.episodeId ?? null,
        rendered_episodes: replayState.ghosts.map((ghost) => ({
          episode_id: ghost.episodeId, visible: ghost.visible,
          position: { ...ghost.position }, color: ghost.color,
          effects: ghost.car?.polybotVisualEffects?.(),
          wheel_contacts: ghost.renderState?.wheelContact.filter(Boolean).length ?? 0,
          skidding_wheels: ghost.renderState?.wheelSkidInfo.filter((skid, index) =>
            ghost.renderState.wheelContact[index] && skid <
              Math.pow(Math.min(1, Math.abs(ghost.renderState.wheelDeltaRotation[index]) / 0.08), 3) * 0.5
          ).length ?? 0,
        })),
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
      if (replayState.dirty) {
        replayState.endTime = replayState.durationSeconds
          + (replayState.endBehavior === "freeze" ? 0 : Math.max(0.75,replayState.fadeDuration));
      }
      if (replayState.playing && Number.isFinite(deltaSeconds) && deltaSeconds > 0) {
        replayState.playbackSeconds += deltaSeconds * replayState.speed;
      }
      if (!replayState.dirty && replayState.lastPlaybackSeconds === replayState.playbackSeconds) {
        replayFollowLeader(deltaSeconds);
        return;
      }
      replayState.dirty = false;
      replayState.lastPlaybackSeconds = replayState.playbackSeconds;
      const interval = replayInterval;
      for (let ghostIndex = 0; ghostIndex < replayState.ghosts.length; ghostIndex += 1) {
        const ghost = replayState.ghosts[ghostIndex];
        const elapsed = replayState.playbackSeconds;
        const ended = elapsed > ghost.duration;
        const coastDuration = replayState.endBehavior === "fade"
          ? Math.max(0.75, replayState.fadeDuration) : 0.75;
        const coastTime = ended && replayState.endBehavior !== "freeze"
          ? Math.min(elapsed - ghost.duration, coastDuration) : 0;
        if (ended && replayState.endBehavior !== "freeze" && elapsed - ghost.duration >= coastDuration) {
          if (ghost.lastAudioVolume !== 0) ghost.car.audioVolume = 0;
          ghost.lastAudioVolume = 0;
          if (ghost.visible) ghost.car.setVisible(false);
          ghost.visible = false;
          continue;
        }
        let opacity = replayState.opacity;
        if (ended && replayState.endBehavior === "fade") {
          opacity *= Math.max(0, 1 - coastTime / coastDuration);
        }
        const localElapsed = ended ? ghost.duration : elapsed;
        replayInterpolateInto(ghost, localElapsed, interval);
        const left = interval.left;
        const right = interval.right;
        const fraction = interval.fraction;
        const source = interval.source;
        ghost.hudFrame = (source || (fraction >= 1 ? right : left)).hudFrame;
        const leftProgress = left.hudFrame?.progress_m;
        const rightProgress = right.hudFrame?.progress_m;
        ghost.progressM = Number.isFinite(leftProgress)
          ? leftProgress + fraction * ((Number.isFinite(rightProgress) ? rightProgress : leftProgress) - leftProgress)
          : null;
        ghost.distanceM = left.distanceM + fraction * (right.distanceM - left.distanceM);
        const orientation = ghost.orientation;
        let positionX = source ? source[2] : left[2] + fraction * (right[2] - left[2]);
        let positionY = source ? source[3] : left[3] + fraction * (right[3] - left[3]);
        let positionZ = source ? source[4] : left[4] + fraction * (right[4] - left[4]);
        // Visual-only coast: extend the final measured velocity briefly while
        // fading. Recorded transforms and lap results remain unchanged.
        const tail = ghost.samples.at(-1);
        const prior = ghost.samples.length > 1 ? ghost.samples.at(-2) : tail;
        const tailSpan = tail[1] - prior[1];
        if (coastTime > 0 && tailSpan > 0) {
          const travelTime = coastTime - 0.25 * coastTime * coastTime / coastDuration;
          positionX += (tail[2] - prior[2]) / tailSpan * travelTime;
          positionY += (tail[3] - prior[3]) / tailSpan * travelTime;
          positionZ += (tail[4] - prior[4]) / tailSpan * travelTime;
        }
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
        state.frames = Math.round((elapsedAtTransform + coastTime) * 1000);
        state.position = ghost.position;
        state.quaternion = orientation;
        const leftIndex = Math.max(0, ghost.sampleIndex - 1);
        const leftWheels = left.wheelState || replayLegacyWheels(left,
          ghost.samples[Math.max(0, leftIndex - 1)], right);
        const rightWheels = right.wheelState || replayLegacyWheels(right, left,
          ghost.samples[Math.min(ghost.samples.length - 1, ghost.sampleIndex + 1)]);
        replayApplyWheels(state, leftWheels, rightWheels, fraction);
        if (ended) state.wheelContact.fill(null);
        const span = right[1] - left[1];
        state.speedKmh = span > 0
          ? Math.hypot(right[2] - left[2], right[3] - left[3], right[4] - left[4]) / span * 3.6 * (ended ? 1 - 0.5 * coastTime / coastDuration : 1) : 0;
        const resetEffects = ghost.cameraNeedsReset;
        const applied = ghost.hudFrame?.controls?.applied;
        state.controls.up = Number(applied?.throttle) > 0;
        state.controls.down = Number(applied?.brake) > 0;
        ghost.car.setCarState(state, resetEffects);
        ghost.cameraNeedsReset = false;
        ghost.car.isPaused = true;
        const audioVolume = ghost === replayState.leader && replayState.playing &&
          (!ended || coastTime > 0) ? (ended ? Math.max(0,1-coastTime/coastDuration) : 1) : 0;
        if (ghost.lastAudioVolume !== audioVolume) {
          ghost.car.audioVolume = audioVolume;
          ghost.lastAudioVolume = audioVolume;
        }
        ghost.car.update(replayState.playing && !resetEffects && !ended ? deltaSeconds * replayState.speed : 0);
      }
      replayFollowLeader(deltaSeconds);
      replayUpdateBatches();
      const endTime = replayState.endTime;
      if (replayState.playbackSeconds >= endTime) {
        replayState.playbackSeconds = endTime;
        replayState.playing = false;
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
      if (command.op !== "status") {
        replayState.dirty = true;
      }
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
            lap_time_s: params.lap_time_s,
            training_step_start: params.training_step_start ?? 0,
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
          replayRender(0);
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
          for (const batch of replayState.batches) {
            batch.mesh.material.opacity = opacity;
            batch.mesh.material.transparent = opacity < 1;
            batch.mesh.material.needsUpdate = true;
          }
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
        case "color_scale": {
          if (!Array.isArray(params.stops) || params.stops.length < 2) throw new Error("at least two colour stops are required");
          const stops = params.stops.map((stop) => ({
            step: replayFinite(stop.step, "colour step"),
            rgb: replayColor(stop.color).slice(1).match(/../g).map((channel) => Number.parseInt(channel, 16)),
          }));
          if (stops.some((stop, index) => !Number.isSafeInteger(stop.step) || stop.step < 0 ||
              (index && stop.step <= stops[index - 1].step))) throw new Error("colour steps must be ordered and distinct");
          for (const ghost of replayState.ghosts) {
            const step = Math.max(stops[0].step, Math.min(stops.at(-1).step, ghost.trainingStepStart));
            const rightIndex = stops.findIndex((stop, index) => index && step <= stop.step);
            const left = stops[rightIndex - 1], right = stops[rightIndex];
            const fraction = (step - left.step) / (right.step - left.step);
            ghost.color = "#" + left.rgb.map((channel, index) =>
              Math.round(channel + fraction * (right.rgb[index] - channel)).toString(16).padStart(2, "0")).join("");
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
      const frameAt = performance.now();
      if (replayState.context && replayState.context.owner !== context.owner) replayClear({ restoreCamera: false });
      replayState.context = context;
      while (replayPendingCommands.length) {
        const { command, reply } = replayPendingCommands.shift();
        try {
          reply({ ok: true, result: replayDispatch(command) });
        } catch (error) {
          reply({ ok: false, error: error instanceof Error ? error.message : String(error) });
        }
      }
      const updateAt = performance.now();
      replayRender(context.deltaSeconds);
      if (previousFrameAt !== null) {
        frameProfile.push({ frame: frameAt - previousFrameAt, replay: performance.now() - updateAt,
          renderer: context.getRenderStats?.() });
        if (frameProfile.length > 60) frameProfile.shift();
      }
      previousFrameAt = frameAt;
    };
    globalThis.__polybotStopVisualReplay = () => replayClear();
    globalThis.__polybotReplayAudioVolume = (car, volume) => {
      if (!replayState.ghosts.length) return volume;
      // Also silence the hidden player and any native ghosts in the scene.
      return car === replayState.leader?.car ? volume : 0;
    };
    globalThis.__polybotVisualReplayTime = (player) => {
      const leader = replayState.leader;
      if (!leader || !replayState.ghosts.length || player !== replayState.context?.player) return null;
      // Sample time is race elapsed time; playback time starts at the first sample.
      // Keep the fixed camera run's finish time while its car coasts away.
      const start = leader.samples[0][1];
      const end = Number.isFinite(leader.lapTime) ? leader.lapTime : start + leader.duration;
      const seconds = Math.max(0,Math.min(start + replayState.playbackSeconds,end));
      const NativeTime = player.getTime().constructor;
      return new NativeTime(Math.round(seconds * 1000));
    };


}
