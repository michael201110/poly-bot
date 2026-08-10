/*
 * This function is serialized by PolyModLoader and inserted into the exact
 * PolyTrack 0.6.2 simulation-worker message handler. Keep every dependency in
 * the function body: imported bindings are not available in the worker blob.
 */
export function polybotWorkerInjection() {
  {
    const runtimeKey = "__polybotWorkerRuntime_v1";

    if (!self[runtimeKey]) {
      self[runtimeKey] = (() => {
        const cars = e;
        const advanceCar = n;
        const messageTypes = Ki;
        const protocol = "polybot.sim";
        const protocolVersion = 1;
        const bridgeUrl = "ws://127.0.0.1:8765";
        const fixedDtSeconds = 0.001;
        const maxTicksPerStep = 250;
        const maxLookaheadCount = 64;
        const referenceSampleTicks = 20;
        const maxReferenceTicks = 600000;
        const hiddenCarId = 2147483000;
        const reconnectDelayMs = 1000;
        const readyWaitMs = 8000;
        const relayName = "polybot-ghost-relay-v1";

        let socket = null;
        let reconnectTimer = null;
        let internalDispatchDepth = 0;
        let busy = false;
        let manualMode = false;
        let playerCarId = null;
        let lookaheadCount = null;
        let episodeCounter = 0;
        let session = null;
        let reference = null;
        let referenceRecording = null;
        let referenceTrackData = null;
        let gameVersion = "0.6.2";

        const createMessages = new Map();
        const startMessages = new Map();
        const localGhostMessages = new Map();
        const remoteGhostMessages = new Map();
        const workerRelayId = `${Date.now()}-${Math.random()}`;
        let ghostRelay = null;

        class BridgeError extends Error {
          constructor(code, message) {
            super(message);
            this.name = "BridgeError";
            this.code = code;
          }
        }

        function relayGhost(message) {
          try {
            ghostRelay?.postMessage({
              protocol: "polybot.ghost-relay",
              v: 1,
              source: workerRelayId,
              op: "ghost",
              message: {
                trackData: message.trackData,
                carRecording: message.carRecording,
              },
            });
          } catch (error) {
            console.warn("[PolyBot] Could not relay a ghost", error);
          }
        }

        if (typeof BroadcastChannel === "function") {
          try {
            ghostRelay = new BroadcastChannel(relayName);
            ghostRelay.addEventListener("message", (event) => {
              const data = event.data;
              if (
                !data ||
                data.protocol !== "polybot.ghost-relay" ||
                data.v !== 1 ||
                data.source === workerRelayId
              ) {
                return;
              }
              if (data.op === "request") {
                for (const message of localGhostMessages.values()) {
                  relayGhost(message);
                }
                return;
              }
              const message = data.message;
              if (
                data.op === "ghost" &&
                message &&
                typeof message.trackData === "string" &&
                message.carRecording != null
              ) {
                remoteGhostMessages.set(message.trackData, message);
              }
            });
            ghostRelay.postMessage({
              protocol: "polybot.ghost-relay",
              v: 1,
              source: workerRelayId,
              op: "request",
            });
          } catch (error) {
            ghostRelay = null;
            console.warn("[PolyBot] Ghost relay is unavailable", error);
          }
        }

        function finite(value, fallback = 0) {
          return Number.isFinite(value) ? value : fallback;
        }

        function clamp(value, minimum, maximum) {
          return Math.max(minimum, Math.min(maximum, value));
        }

        function add(a, b) {
          return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        }

        function subtract(a, b) {
          return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        }

        function scale(vector, amount) {
          return [vector[0] * amount, vector[1] * amount, vector[2] * amount];
        }

        function dot(a, b) {
          return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }

        function cross(a, b) {
          return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
          ];
        }

        function length(vector) {
          return Math.sqrt(Math.max(0, dot(vector, vector)));
        }

        function distance(a, b) {
          return length(subtract(a, b));
        }

        function normalize(vector, fallback = [0, 0, 1]) {
          const magnitude = length(vector);
          if (!Number.isFinite(magnitude) || magnitude < 1e-8) {
            return [...fallback];
          }
          return scale(vector, 1 / magnitude);
        }

        function normalizeQuaternion(quaternion) {
          const magnitude = Math.sqrt(
            quaternion[0] * quaternion[0] +
              quaternion[1] * quaternion[1] +
              quaternion[2] * quaternion[2] +
              quaternion[3] * quaternion[3],
          );
          if (!Number.isFinite(magnitude) || magnitude < 1e-8) {
            return [0, 0, 0, 1];
          }
          return quaternion.map((value) => value / magnitude);
        }

        function quaternionMultiply(a, b) {
          return [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
          ];
        }

        function rotateVector(quaternion, vector) {
          const q = normalizeQuaternion(quaternion);
          const pure = [vector[0], vector[1], vector[2], 0];
          const conjugate = [-q[0], -q[1], -q[2], q[3]];
          const rotated = quaternionMultiply(quaternionMultiply(q, pure), conjugate);
          return [rotated[0], rotated[1], rotated[2]];
        }

        function requireObject(value, name) {
          if (value === null || typeof value !== "object" || Array.isArray(value)) {
            throw new BridgeError("invalid_request", `${name} must be an object`);
          }
          return value;
        }

        function findCar(carId) {
          return cars.find((car) => car.id === carId) ?? null;
        }

        function internalDispatch(message) {
          internalDispatchDepth += 1;
          try {
            self.onmessage({ data: message });
          } finally {
            internalDispatchDepth -= 1;
          }
        }

        function deleteAllCars() {
          const ids = cars.map((car) => car.id);
          for (const carId of ids) {
            internalDispatch({ messageType: messageTypes.DeleteCar, carId });
          }
        }

        function restoreCachedCars({ paused }) {
          deleteAllCars();
          for (const message of createMessages.values()) {
            internalDispatch(message);
          }
          for (const message of startMessages.values()) {
            if (createMessages.has(message.carId)) {
              internalDispatch(message);
            }
          }
          for (const car of cars) {
            car.isPaused = paused;
          }
        }

        function pauseManualCars() {
          if (!manualMode) {
            return;
          }
          for (const car of cars) {
            car.isPaused = true;
          }
        }

        function leaveManualMode() {
          if (!manualMode) {
            return;
          }
          manualMode = false;
          session = null;
          try {
            // Recreate at the start before returning control. This prevents a
            // native finished state, hidden from the UI, surfacing on the next
            // vanilla realtime tick.
            restoreCachedCars({ paused: false });
          } catch (error) {
            console.error("[PolyBot] Could not restore vanilla simulation", error);
            for (const car of cars) {
              car.isPaused = false;
            }
          }
        }

        function decodeState(buffer) {
          const bytes = new Uint8Array(buffer);
          if (bytes.byteLength < 84) {
            throw new BridgeError("invalid_state", "PolyTrack returned a short car-state packet");
          }
          const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
          let offset = 4;

          const frames = bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
          offset += 3;
          const speedKmh = view.getFloat32(offset, true);
          offset += 4;
          const flags = bytes[offset];
          offset += 1;
          const hasStarted = Boolean(flags & 1);
          const hasFinished = Boolean(flags & 2);
          const hasCheckpointToRespawnAt = Boolean(flags & 4);
          const contactFlags = [8, 16, 32, 64].map((mask) => Boolean(flags & mask));

          let finishFrames = null;
          if (hasFinished) {
            finishFrames =
              bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
            offset += 3;
          }

          const nextCheckpointIndex = view.getUint16(offset, true);
          offset += 2;
          const position = [
            view.getFloat32(offset, true),
            view.getFloat32(offset + 4, true),
            view.getFloat32(offset + 8, true),
          ];
          offset += 12;
          const quaternion = normalizeQuaternion([
            view.getFloat32(offset, true),
            view.getFloat32(offset + 4, true),
            view.getFloat32(offset + 8, true),
            view.getFloat32(offset + 12, true),
          ]);
          offset += 16;

          const impulseCount = view.getUint8(offset);
          offset += 1;
          if (impulseCount > 4) {
            throw new BridgeError("invalid_state", "PolyTrack returned too many collision impulses");
          }
          const collisionImpulses = [];
          for (let index = 0; index < impulseCount; index += 1) {
            collisionImpulses.push(view.getFloat32(offset, true));
            offset += 4;
          }

          for (const hasContact of contactFlags) {
            if (hasContact) {
              offset += 24;
            }
          }
          // Suspension length, suspension velocity, wheel rotation, and skid.
          offset += 4 * 4 * 4;
          const steering = view.getFloat32(offset, true);
          offset += 4;
          const controlFlags = view.getUint8(offset);

          if (offset >= bytes.byteLength) {
            throw new BridgeError("invalid_state", "PolyTrack car-state packet is truncated");
          }

          return {
            frames,
            speedKmh: finite(speedKmh),
            hasStarted,
            hasFinished,
            finishFrames,
            hasCheckpointToRespawnAt,
            nextCheckpointIndex,
            position: position.map((value) => finite(value)),
            quaternion,
            collisionImpulses: collisionImpulses.map((value) => finite(value)),
            wheelContacts: contactFlags,
            steering: finite(steering),
            controls: {
              up: Boolean(controlFlags & 1),
              right: Boolean(controlFlags & 2),
              down: Boolean(controlFlags & 4),
              left: Boolean(controlFlags & 8),
              reset: Boolean(controlFlags & 16),
            },
          };
        }

        function addReferencePoint(points, decoded, force = false) {
          const point = {
            position: decoded.position,
            quaternion: decoded.quaternion,
            nextCheckpointIndex: decoded.nextCheckpointIndex,
            s: 0,
            tangent: [0, 0, 1],
            up: [0, 1, 0],
            right: [1, 0, 0],
            curvature: 0,
          };
          const previous = points.at(-1);
          if (!previous) {
            points.push(point);
            return;
          }
          const separation = distance(previous.position, point.position);
          if (force || separation >= 0.15) {
            point.s = previous.s + separation;
            points.push(point);
          }
        }

        function finishReference(points) {
          if (points.length < 10) {
            throw new BridgeError(
              "missing_reference",
              "The loaded ghost did not produce a usable trajectory.",
            );
          }

          let forwardAgreement = 0;
          for (let index = 0; index < points.length; index += 1) {
            const previous = points[Math.max(0, index - 1)];
            const next = points[Math.min(points.length - 1, index + 1)];
            const tangent = normalize(subtract(next.position, previous.position));
            const up = normalize(rotateVector(points[index].quaternion, [0, 1, 0]), [0, 1, 0]);
            points[index].tangent = tangent;
            points[index].up = up;
            points[index].right = normalize(
              rotateVector(points[index].quaternion, [1, 0, 0]),
              [1, 0, 0],
            );
            forwardAgreement += dot(
              rotateVector(points[index].quaternion, [0, 0, 1]),
              tangent,
            );
          }

          const forwardSign = forwardAgreement >= 0 ? 1 : -1;

          for (let index = 1; index < points.length - 1; index += 1) {
            const before = points[index - 1];
            const after = points[index + 1];
            const angle = Math.atan2(
              dot(cross(before.tangent, after.tangent), points[index].up),
              clamp(dot(before.tangent, after.tangent), -1, 1),
            );
            const arc = Math.max(0.1, after.s - before.s);
            // Multiplying by forwardSign keeps positive curvature aligned with
            // the game's physical "right" control for either +Z- or
            // -Z-forward car assets.
            points[index].curvature = finite((angle * forwardSign) / arc);
          }
          points[0].curvature = points[1].curvature;
          points.at(-1).curvature = points.at(-2).curvature;

          return {
            points,
            length: points.at(-1).s,
            forwardSign,
          };
        }

        function buildReference(ghostCreateMessage) {
          internalDispatch({
            ...ghostCreateMessage,
            messageType: messageTypes.CreateCar,
            carId: hiddenCarId,
          });
          internalDispatch({
            messageType: messageTypes.StartCar,
            carId: hiddenCarId,
            targetSimulationTimeFrames: null,
          });

          const hiddenCar = findCar(hiddenCarId);
          if (!hiddenCar || hiddenCar.userControls !== null) {
            throw new BridgeError("missing_reference", "Could not initialize the loaded ghost lap.");
          }
          hiddenCar.isPaused = true;

          const points = [];
          let finished = false;
          try {
            for (let frame = 0; frame < maxReferenceTicks; frame += 1) {
              const controls = hiddenCar.controls.getControls(hiddenCar.frames);
              const buffer = advanceCar(hiddenCar, controls);
              hiddenCar.frames += 1;
              const decoded = decodeState(buffer);
              if (frame % referenceSampleTicks === 0 || decoded.hasFinished) {
                addReferencePoint(points, decoded, decoded.hasFinished);
              }
              if (decoded.hasFinished) {
                finished = true;
                break;
              }
            }
          } finally {
            internalDispatch({ messageType: messageTypes.DeleteCar, carId: hiddenCarId });
          }

          if (!finished) {
            throw new BridgeError(
              "missing_reference",
              "The loaded ghost did not reach the finish within ten simulated minutes.",
            );
          }
          return finishReference(points);
        }

        function choosePlayerMessage() {
          if (playerCarId === null) {
            return null;
          }
          return createMessages.get(playerCarId) ?? null;
        }

        function chooseGhostMessage(playerMessage) {
          if (!playerMessage) {
            return null;
          }
          for (const message of createMessages.values()) {
            if (
              message.carRecording != null &&
              message.trackData === playerMessage.trackData
            ) {
              return message;
            }
          }
          return remoteGhostMessages.get(playerMessage.trackData) ?? null;
        }

        function readyForEpisode() {
          const playerMessage = choosePlayerMessage();
          return Boolean(
            playerMessage &&
              startMessages.has(playerMessage.carId) &&
              chooseGhostMessage(playerMessage),
          );
        }

        async function waitUntilReady() {
          const deadline = performance.now() + readyWaitMs;
          while (!readyForEpisode() && performance.now() < deadline) {
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
          if (!choosePlayerMessage()) {
            throw new BridgeError(
              "game_not_ready",
              "Enter a PolyTrack time-trial race before starting PolyBot.",
            );
          }
          if (!startMessages.has(playerCarId)) {
            throw new BridgeError(
              "game_not_ready",
              "The player car has not started. Enter the race, then try again.",
            );
          }
          if (!chooseGhostMessage(choosePlayerMessage())) {
            throw new BridgeError(
              "missing_reference",
              "Load a ghost lap for the current track before starting PolyBot.",
            );
          }
        }

        function carBasis(decoded) {
          const sign = reference?.forwardSign ?? 1;
          const up = normalize(rotateVector(decoded.quaternion, [0, 1, 0]), [0, 1, 0]);
          const forward = normalize(
            rotateVector(decoded.quaternion, [0, 0, sign]),
            [0, 0, 1],
          );
          const right = normalize(
            rotateVector(decoded.quaternion, [1, 0, 0]),
            [1, 0, 0],
          );
          return { right, up, forward };
        }

        function findReferenceIndex(decoded) {
          const points = reference.points;
          const previousIndex = session?.referenceIndex ?? 0;
          const previousProgress = session?.progressM ?? 0;
          const displacement = session?.previousDecoded
            ? distance(decoded.position, session.previousDecoded.position)
            : 0;
          const firstProgress = Math.max(0, previousProgress - 2);
          const lastProgress = previousProgress + Math.max(12, displacement * 2 + 2);
          let first = previousIndex;
          while (first > 0 && points[first - 1].s >= firstProgress) {
            first -= 1;
          }
          let last = previousIndex;
          while (last + 1 < points.length && points[last + 1].s <= lastProgress) {
            last += 1;
          }
          let bestIndex = first;
          let bestDistanceSquared = Number.POSITIVE_INFINITY;
          for (let index = first; index <= last; index += 1) {
            if (points[index].nextCheckpointIndex !== decoded.nextCheckpointIndex) {
              continue;
            }
            const delta = subtract(decoded.position, points[index].position);
            const squared = dot(delta, delta);
            if (squared < bestDistanceSquared) {
              bestDistanceSquared = squared;
              bestIndex = index;
            }
          }
          // A checkpoint transition can briefly precede the next sampled ghost
          // point. Stay at the last guide point until that sample is in range.
          if (!Number.isFinite(bestDistanceSquared)) {
            bestIndex = previousIndex;
          }
          if (session) {
            session.referenceIndex = bestIndex;
            session.progressM = Math.max(session.progressM, points[bestIndex].s);
          }
          return bestIndex;
        }

        function angularVelocity(previousQuaternion, quaternion, dt, basis) {
          if (!previousQuaternion || dt <= 0) {
            return [0, 0, 0];
          }
          const previous = normalizeQuaternion(previousQuaternion);
          let delta = quaternionMultiply(quaternion, [
            -previous[0],
            -previous[1],
            -previous[2],
            previous[3],
          ]);
          if (delta[3] < 0) {
            delta = delta.map((value) => -value);
          }
          delta = normalizeQuaternion(delta);
          const halfSin = Math.sqrt(
            delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2],
          );
          if (halfSin < 1e-8) {
            return [0, 0, 0];
          }
          const angle = 2 * Math.atan2(halfSin, clamp(delta[3], -1, 1));
          const world = scale([delta[0], delta[1], delta[2]], angle / (halfSin * dt));
          return [dot(world, basis.right), dot(world, basis.up), dot(world, basis.forward)].map(
            (value) => finite(value),
          );
        }

        function buildTelemetry(decoded, action, dtSeconds) {
          const basis = carBasis(decoded);
          const referenceIndex = findReferenceIndex(decoded);
          const guide = reference.points[referenceIndex];
          const offset = subtract(decoded.position, guide.position);
          const lateralOffset = dot(offset, guide.right);
          const headingError =
            (reference.forwardSign ?? 1) *
            Math.atan2(
              dot(cross(guide.tangent, basis.forward), basis.up),
              clamp(dot(guide.tangent, basis.forward), -1, 1),
            );

          let localVelocity = [0, 0, 0];
          if (session?.previousDecoded && dtSeconds > 0) {
            const worldVelocity = scale(
              subtract(decoded.position, session.previousDecoded.position),
              1 / dtSeconds,
            );
            localVelocity = [
              dot(worldVelocity, basis.right),
              dot(worldVelocity, basis.up),
              dot(worldVelocity, basis.forward),
            ].map((value) => finite(value));
          }
          const localAngularVelocity = angularVelocity(
            session?.previousDecoded?.quaternion ?? null,
            decoded.quaternion,
            dtSeconds,
            basis,
          );

          const lookahead = [];
          const lookaheadMask = [];
          let cursor = referenceIndex;
          for (let index = 0; index < lookaheadCount; index += 1) {
            const targetDistance = guide.s + (index + 1) * 5;
            while (cursor < reference.points.length && reference.points[cursor].s < targetDistance) {
              cursor += 1;
            }
            if (cursor >= reference.points.length) {
              lookahead.push([0, 0, 0, 0]);
              lookaheadMask.push(0);
              continue;
            }
            const target = reference.points[cursor];
            const delta = subtract(target.position, decoded.position);
            lookahead.push([
              finite(dot(delta, basis.forward)),
              finite(dot(delta, basis.right)),
              finite(dot(delta, basis.up)),
              finite(target.curvature),
            ]);
            lookaheadMask.push(1);
          }

          const worldUp = [0, 1, 0];
          const pitch = Math.asin(clamp(basis.forward[1], -1, 1));
          const roll = Math.atan2(dot(basis.right, worldUp), dot(basis.up, worldUp));

          return {
            position_m: decoded.position.map((value) => finite(value)),
            quaternion_xyzw: decoded.quaternion.map((value) => finite(value)),
            local_velocity_mps: localVelocity,
            angular_velocity_radps: localAngularVelocity,
            up_vector: basis.up.map((value) => finite(value)),
            pitch_rad: finite(pitch),
            roll_rad: finite(roll),
            wheel_contacts: decoded.wheelContacts.map((contact) => (contact ? 1 : 0)),
            checkpoint_index: decoded.nextCheckpointIndex,
            elapsed_s: session.tick * fixedDtSeconds,
            previous_action: action,
            track: {
              progress_m: finite(session?.progressM ?? guide.s),
              length_m: Math.max(0.1, finite(reference.length, 0.1)),
              half_width_m: 5,
              lateral_offset_m: finite(lateralOffset),
              heading_error_rad: finite(headingError),
              lookahead,
              lookahead_mask: lookaheadMask,
            },
          };
        }

        function publishStates(buffers) {
          const visibleBuffers = buffers.map((buffer) => {
            const visibleBuffer = buffer.slice(0);
            // Finish flags are deliberately hidden from the game UI. Python
            // still receives the player result, but PolyTrack cannot submit an
            // automated finish.
            const visibleBytes = new Uint8Array(visibleBuffer);
            if (visibleBytes[11] & 2) {
              // A finished packet has a three-byte finishFrames field directly
              // after flags. Removing only the flag would misalign every later
              // field in the game's decoder, so remove that field as well.
              visibleBytes.copyWithin(12, 15);
              visibleBytes.fill(0, visibleBytes.length - 3);
              visibleBytes[11] &= ~2;
            }
            return visibleBuffer;
          });
          postMessage(
            {
              messageType: messageTypes.UpdateResult,
              carStateBuffers: visibleBuffers,
            },
            { transfer: visibleBuffers },
          );
        }

        function transition(decoded, action, ticksAdvanced, events, seed = null) {
          const state = buildTelemetry(decoded, action, ticksAdvanced * fixedDtSeconds);
          const guide = reference.points[session.referenceIndex];
          const offTrack = Math.abs(state.track.lateral_offset_m) > state.track.half_width_m * 1.5;
          if (offTrack && !events.includes("off_track")) {
            events.push("off_track");
          }
          const result = {
            episode_id: session.episodeId,
            tick: session.tick,
            ticks_advanced: ticksAdvanced,
            state,
            events,
            info: {
              track_id: "current",
              guide: "loaded_ghost",
              reference_points: reference.points.length,
              guide_progress_m: finite(session.progressM),
              native_frame: decoded.frames,
              speed_kmh: decoded.speedKmh,
              collision_impulses: decoded.collisionImpulses,
              off_track: offTrack,
              leaderboard_finish_masked: true,
            },
          };
          if (seed !== null) {
            result.info.seed = seed;
            result.info.seed_ignored = true;
          }
          session.previousDecoded = decoded;
          session.previousAction = action;
          return result;
        }

        async function resetEpisode(params) {
          if (lookaheadCount === null) {
            throw new BridgeError("invalid_state", "hello must be called before reset");
          }
          if (!Number.isSafeInteger(params.seed) || params.seed < 0) {
            throw new BridgeError("invalid_request", "seed must be a non-negative integer");
          }
          if (params.track_id !== "current") {
            throw new BridgeError(
              "track_mismatch",
              "The PolyTrack worker controls the loaded track; use --track current.",
            );
          }

          await waitUntilReady();
          const playerMessage = choosePlayerMessage();
          const ghostCandidate = chooseGhostMessage(playerMessage);
          // Only the serialized controls cross the worker boundary. Terrain
          // and track initialization come from this player worker's own
          // CreateCar payload.
          const ghostMessage = {
            ...playerMessage,
            carRecording: ghostCandidate.carRecording,
          };
          manualMode = true;
          try {
            deleteAllCars();
            try {
              if (
                !reference ||
                referenceRecording !== ghostMessage.carRecording ||
                referenceTrackData !== ghostMessage.trackData
              ) {
                reference = buildReference(ghostMessage);
                referenceRecording = ghostMessage.carRecording;
                referenceTrackData = ghostMessage.trackData;
              }
            } finally {
              restoreCachedCars({ paused: true });
            }

            const player = findCar(playerCarId);
            if (!player) {
              throw new BridgeError("game_not_ready", "Could not recreate the player car");
            }
            player.isPaused = true;
            const neutralControls = {
              up: false,
              right: false,
              down: false,
              left: false,
              reset: false,
            };
            const initialBuffers = [];
            let initialBuffer = null;
            for (const car of cars) {
              if (!car.hasStarted) {
                continue;
              }
              const controls =
                car.id === playerCarId
                  ? neutralControls
                  : car.userControls === null
                    ? car.controls.getControls(car.frames)
                    : neutralControls;
              const buffer = advanceCar(car, controls);
              car.frames += 1;
              initialBuffers.push(buffer);
              if (car.id === playerCarId) {
                initialBuffer = buffer;
              }
            }
            if (!initialBuffer) {
              throw new BridgeError("game_not_ready", "The player car has not started");
            }
            const decoded = decodeState(initialBuffer);
            publishStates(initialBuffers);

            episodeCounter += 1;
            session = {
              episodeId: `polytrack-${episodeCounter}`,
              tick: 0,
              previousDecoded: decoded,
              previousAction: { steer: 0, throttle: 0, brake: 0 },
              previousCheckpoint: decoded.nextCheckpointIndex,
              referenceIndex: 0,
              progressM: 0,
            };
            return transition(
              decoded,
              { steer: 0, throttle: 0, brake: 0 },
              0,
              [],
              params.seed,
            );
          } catch (error) {
            leaveManualMode();
            throw error;
          }
        }

        function validateAction(value) {
          const action = requireObject(value, "action");
          if (![-1, 0, 1].includes(action.steer)) {
            throw new BridgeError("invalid_action", "steer must be -1, 0, or 1");
          }
          if (![0, 1].includes(action.throttle) || ![0, 1].includes(action.brake)) {
            throw new BridgeError("invalid_action", "throttle and brake must be 0 or 1");
          }
          return {
            steer: action.steer,
            throttle: action.throttle,
            brake: action.brake,
          };
        }

        function stepEpisode(params) {
          if (!session || params.episode_id !== session.episodeId) {
            throw new BridgeError("stale_episode", "episode_id is stale or unknown");
          }
          if (
            !Number.isSafeInteger(params.ticks) ||
            params.ticks < 1 ||
            params.ticks > maxTicksPerStep
          ) {
            throw new BridgeError(
              "invalid_action",
              `ticks must be an integer from 1 to ${maxTicksPerStep}`,
            );
          }
          const action = validateAction(params.action);
          const player = findCar(playerCarId);
          if (!player) {
            throw new BridgeError("game_not_ready", "The player car no longer exists");
          }
          player.isPaused = true;
          const controls = {
            up: Boolean(action.throttle),
            right: action.steer === 1,
            down: Boolean(action.brake),
            left: action.steer === -1,
            reset: false,
          };

          let decoded = session.previousDecoded;
          const finalBuffers = new Map();
          let ticksAdvanced = 0;
          for (let tick = 0; tick < params.ticks; tick += 1) {
            for (const car of cars) {
              if (!car.hasStarted) {
                continue;
              }
              const carControls =
                car.id === playerCarId
                  ? controls
                  : car.userControls === null
                    ? car.controls.getControls(car.frames)
                    : { up: false, right: false, down: false, left: false, reset: false };
              const buffer = advanceCar(car, carControls);
              car.frames += 1;
              finalBuffers.set(car.id, buffer);
              if (car.id === playerCarId) {
                decoded = decodeState(buffer);
              }
            }
            ticksAdvanced += 1;
            if (decoded.hasFinished) {
              break;
            }
          }
          session.tick += ticksAdvanced;

          if (finalBuffers.size > 0) {
            publishStates([...finalBuffers.values()]);
          }
          const events = [];
          if (decoded.nextCheckpointIndex > session.previousCheckpoint) {
            for (
              let index = session.previousCheckpoint;
              index < decoded.nextCheckpointIndex;
              index += 1
            ) {
              events.push("checkpoint");
            }
          }
          session.previousCheckpoint = decoded.nextCheckpointIndex;
          if (decoded.hasFinished) {
            events.push("finish");
          }
          return transition(decoded, action, ticksAdvanced, events);
        }

        async function dispatchRequest(request) {
          if (request.protocol !== protocol) {
            throw new BridgeError("unsupported_protocol", "unsupported protocol name");
          }
          if (request.v !== protocolVersion) {
            throw new BridgeError("unsupported_version", "unsupported protocol version");
          }
          if (!Number.isSafeInteger(request.id) || request.id < 0) {
            throw new BridgeError("invalid_request", "id must be a non-negative integer");
          }
          const params = requireObject(request.params, "params");
          switch (request.op) {
            case "hello":
              if (params.protocol !== protocol || params.protocol_version !== protocolVersion) {
                throw new BridgeError("unsupported_version", "trainer protocol is unsupported");
              }
              if (
                !Number.isSafeInteger(params.lookahead_count) ||
                params.lookahead_count < 1 ||
                params.lookahead_count > maxLookaheadCount
              ) {
                throw new BridgeError(
                  "invalid_request",
                  `lookahead_count must be from 1 to ${maxLookaheadCount}`,
                );
              }
              lookaheadCount = params.lookahead_count;
              return {
                protocol,
                protocol_version: protocolVersion,
                simulator: "polytrack-pml-worker",
                game_version: gameVersion,
                fixed_dt_s: fixedDtSeconds,
                max_ticks_per_step: maxTicksPerStep,
                lookahead_count: lookaheadCount,
                features: [
                  "offline_finish_mask",
                  "fixed_step",
                  "native_physics",
                  "ghost_reference",
                  "cross_worker_reference",
                  "visible_car",
                ],
              };
            case "reset":
              return resetEpisode(params);
            case "step":
              return stepEpisode(params);
            case "close":
              leaveManualMode();
              return { closed: true };
            default:
              throw new BridgeError(
                "unknown_operation",
                `unsupported operation: ${String(request.op)}`,
              );
          }
        }

        function send(message) {
          if (socket?.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(message));
          }
        }

        async function onSocketMessage(raw) {
          let request = null;
          try {
            request = JSON.parse(raw);
            requireObject(request, "request");
          } catch (error) {
            send({
              protocol,
              v: protocolVersion,
              id: null,
              ok: false,
              error: { code: "malformed_json", message: error.message || String(error) },
            });
            return;
          }

          if (busy) {
            send({
              protocol,
              v: protocolVersion,
              id: request.id ?? null,
              ok: false,
              error: { code: "busy", message: "another simulator request is in progress" },
            });
            return;
          }

          busy = true;
          try {
            const result = await dispatchRequest(request);
            send({ protocol, v: protocolVersion, id: request.id, ok: true, result });
            if (request.op === "close") {
              setTimeout(() => socket?.close(1000, "PolyBot bridge closed"), 0);
            }
          } catch (error) {
            const code = error instanceof BridgeError ? error.code : "adapter_error";
            console.error("[PolyBot] Request failed", error);
            send({
              protocol,
              v: protocolVersion,
              id: request.id ?? null,
              ok: false,
              error: { code, message: error.message || String(error) },
            });
          } finally {
            busy = false;
          }
        }

        function scheduleReconnect() {
          if (reconnectTimer !== null || playerCarId === null) {
            return;
          }
          reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connectSocket();
          }, reconnectDelayMs);
        }

        function connectSocket() {
          if (playerCarId === null || socket?.readyState === WebSocket.OPEN || socket?.readyState === WebSocket.CONNECTING) {
            return;
          }
          let nextSocket;
          try {
            nextSocket = new WebSocket(bridgeUrl);
          } catch (error) {
            console.warn(`[PolyBot] Could not connect to ${bridgeUrl}`, error);
            scheduleReconnect();
            return;
          }
          socket = nextSocket;
          nextSocket.addEventListener("open", () => {
            console.info(`[PolyBot] Connected to ${bridgeUrl}`);
          });
          nextSocket.addEventListener("message", (event) => {
            void onSocketMessage(event.data);
          });
          nextSocket.addEventListener("close", () => {
            if (socket === nextSocket) {
              socket = null;
              leaveManualMode();
              scheduleReconnect();
            }
          });
          nextSocket.addEventListener("error", () => {
            // The Python server is normally started after the race. The close
            // event schedules a quiet retry without interrupting the game.
          });
        }

        function observe(message) {
          if (internalDispatchDepth > 0) {
            return;
          }
          switch (message.messageType) {
            case messageTypes.Init:
              gameVersion = String(message.version ?? gameVersion);
              break;
            case messageTypes.CreateCar:
              createMessages.set(message.carId, message);
              startMessages.delete(message.carId);
              if (message.carRecording == null) {
                playerCarId = message.carId;
                try {
                  ghostRelay?.postMessage({
                    protocol: "polybot.ghost-relay",
                    v: 1,
                    source: workerRelayId,
                    op: "request",
                  });
                } catch (error) {
                  console.warn("[PolyBot] Could not request a reference ghost", error);
                }
              } else if (typeof message.trackData === "string") {
                localGhostMessages.set(message.trackData, message);
                relayGhost(message);
              }
              if (manualMode) {
                setTimeout(pauseManualCars, 0);
              }
              break;
            case messageTypes.DeleteCar: {
              const deletingPlayer = message.carId === playerCarId;
              createMessages.delete(message.carId);
              startMessages.delete(message.carId);
              if (deletingPlayer) {
                playerCarId = null;
                leaveManualMode();
                socket?.close(1000, "player left race");
              }
              break;
            }
            case messageTypes.StartCar:
              startMessages.set(message.carId, message);
              if (message.carId === playerCarId) {
                connectSocket();
              }
              if (manualMode) {
                setTimeout(pauseManualCars, 0);
              }
              break;
            default:
              break;
          }
        }

        return { observe };
      })();
    }

    self[runtimeKey].observe(r);
  }
}
