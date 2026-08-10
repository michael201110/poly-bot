/*
 * Transport-only bridge between PolyTrackTrainingGameApi and the Python trainer.
 * This file intentionally contains no minified/version-specific game hooks.
 */
(function installPolyTrackTrainingBridge(global) {
  "use strict";

  const PROTOCOL = "polybot.sim";
  const VERSION = 1;

  class BridgeError extends Error {
    constructor(code, message) {
      super(message);
      this.name = "BridgeError";
      this.code = code;
    }
  }

  function requireObject(value, name) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      throw new BridgeError("invalid_request", `${name} must be an object`);
    }
    return value;
  }

  function assertFiniteJson(value, path = "result") {
    if (typeof value === "number" && !Number.isFinite(value)) {
      throw new BridgeError("invalid_telemetry", `${path} contains a non-finite number`);
    }
    if (Array.isArray(value)) {
      value.forEach((item, index) => assertFiniteJson(item, `${path}[${index}]`));
    } else if (value !== null && typeof value === "object") {
      Object.entries(value).forEach(([key, item]) => assertFiniteJson(item, `${path}.${key}`));
    }
  }

  class PolyTrackTrainingBridge {
    constructor({ gameApi, url = "ws://127.0.0.1:8765", logger = console } = {}) {
      if (!gameApi) {
        throw new TypeError("gameApi is required");
      }
      this.gameApi = gameApi;
      this.url = url;
      this.logger = logger;
      this.socket = null;
      this.episodeId = null;
      this.episodeCounter = 0;
      this.lookaheadCount = null;
      this.queue = Promise.resolve();
    }

    async start() {
      if (this.socket) {
        throw new Error("bridge is already started");
      }
      if (typeof this.gameApi.ensureOffline !== "function") {
        throw new Error("gameApi.ensureOffline() is required");
      }
      const offline = await this.gameApi.ensureOffline();
      if (offline !== true) {
        throw new Error("game adapter did not confirm offline mode");
      }

      const socket = new WebSocket(this.url);
      this.socket = socket;
      socket.addEventListener("message", (event) => {
        this.queue = this.queue
          .then(() => this.#onMessage(event.data))
          .catch((error) => this.logger.error("PolyBot bridge command failed", error));
      });
      socket.addEventListener("close", () => {
        if (this.socket === socket) {
          this.socket = null;
        }
      });
      socket.addEventListener("error", () => {
        this.logger.error(`PolyBot bridge could not connect to ${this.url}`);
      });
      await new Promise((resolve, reject) => {
        socket.addEventListener("open", resolve, { once: true });
        socket.addEventListener("error", () => reject(new Error("WebSocket connection failed")), {
          once: true,
        });
      });
    }

    async stop() {
      const socket = this.socket;
      this.socket = null;
      this.episodeId = null;
      if (typeof this.gameApi.close === "function") {
        await this.gameApi.close();
      }
      if (socket && socket.readyState < WebSocket.CLOSING) {
        socket.close(1000, "bridge stopped");
      }
    }

    async #onMessage(raw) {
      let request;
      try {
        request = JSON.parse(raw);
        requireObject(request, "request");
      } catch (error) {
        this.#sendError(null, "malformed_json", error.message);
        return;
      }

      try {
        if (request.protocol !== PROTOCOL) {
          throw new BridgeError("unsupported_protocol", "unsupported protocol name");
        }
        if (request.v !== VERSION) {
          throw new BridgeError("unsupported_version", "unsupported protocol version");
        }
        if (!Number.isSafeInteger(request.id) || request.id < 0) {
          throw new BridgeError("invalid_request", "id must be a non-negative integer");
        }
        const params = requireObject(request.params, "params");
        const result = await this.#dispatch(request.op, params);
        assertFiniteJson(result);
        this.#send({
          protocol: PROTOCOL,
          v: VERSION,
          id: request.id,
          ok: true,
          result,
        });
        if (request.op === "close") {
          await this.stop();
        }
      } catch (error) {
        const code = error instanceof BridgeError ? error.code : "adapter_error";
        this.#sendError(request.id ?? null, code, error.message || String(error));
      }
    }

    async #dispatch(op, params) {
      switch (op) {
        case "hello":
          return this.#hello(params);
        case "reset":
          return this.#reset(params);
        case "step":
          return this.#step(params);
        case "close":
          return { closed: true };
        default:
          throw new BridgeError("unknown_operation", `unsupported operation: ${String(op)}`);
      }
    }

    async #hello(params) {
      if (params.protocol !== PROTOCOL || params.protocol_version !== VERSION) {
        throw new BridgeError("unsupported_version", "trainer protocol is unsupported");
      }
      if (!Number.isSafeInteger(params.lookahead_count) || params.lookahead_count < 1) {
        throw new BridgeError("invalid_request", "lookahead_count must be a positive integer");
      }
      const description = requireObject(await this.gameApi.describe(), "gameApi.describe result");
      if (
        !Number.isFinite(description.fixed_dt_s) ||
        description.fixed_dt_s <= 0 ||
        !Number.isSafeInteger(description.max_ticks_per_step)
      ) {
        throw new BridgeError("invalid_adapter", "game API returned invalid stepping capabilities");
      }
      this.lookaheadCount = params.lookahead_count;
      return {
        protocol: PROTOCOL,
        protocol_version: VERSION,
        simulator: description.simulator || "polytrack-local",
        game_version: description.game_version || "unknown",
        fixed_dt_s: description.fixed_dt_s,
        max_ticks_per_step: description.max_ticks_per_step,
        lookahead_count: this.lookaheadCount,
        features: ["offline", "fixed_step", ...(description.features || [])],
      };
    }

    async #reset(params) {
      if (this.lookaheadCount === null) {
        throw new BridgeError("invalid_state", "hello must be called before reset");
      }
      if (!Number.isSafeInteger(params.seed) || params.seed < 0) {
        throw new BridgeError("invalid_request", "seed must be a non-negative integer");
      }
      if (typeof params.track_id !== "string" || !params.track_id) {
        throw new BridgeError("invalid_request", "track_id must be a non-empty string");
      }
      const result = requireObject(
        await this.gameApi.reset({
          seed: params.seed,
          trackId: params.track_id,
          lookaheadCount: this.lookaheadCount,
        }),
        "gameApi.reset result",
      );
      this.episodeCounter += 1;
      this.episodeId = `polytrack-${this.episodeCounter}`;
      return {
        episode_id: this.episodeId,
        tick: result.tick ?? 0,
        ticks_advanced: 0,
        state: requireObject(result.state, "gameApi.reset result.state"),
        events: result.events || [],
        info: result.info || {},
      };
    }

    async #step(params) {
      if (!this.episodeId || params.episode_id !== this.episodeId) {
        throw new BridgeError("stale_episode", "episode_id is stale or unknown");
      }
      if (!Number.isSafeInteger(params.ticks) || params.ticks < 1) {
        throw new BridgeError("invalid_action", "ticks must be a positive integer");
      }
      const action = requireObject(params.action, "action");
      if (![-1, 0, 1].includes(action.steer)) {
        throw new BridgeError("invalid_action", "steer must be -1, 0, or 1");
      }
      if (![0, 1].includes(action.throttle) || ![0, 1].includes(action.brake)) {
        throw new BridgeError("invalid_action", "throttle and brake must be 0 or 1");
      }
      const result = requireObject(
        await this.gameApi.step({ action, ticks: params.ticks }),
        "gameApi.step result",
      );
      if (!Number.isSafeInteger(result.tick) || !Number.isSafeInteger(result.ticks_advanced)) {
        throw new BridgeError("invalid_telemetry", "step tick fields must be integers");
      }
      if (result.ticks_advanced < 0 || result.ticks_advanced > params.ticks) {
        throw new BridgeError("invalid_telemetry", "ticks_advanced is outside the requested range");
      }
      return {
        episode_id: this.episodeId,
        tick: result.tick,
        ticks_advanced: result.ticks_advanced,
        state: requireObject(result.state, "gameApi.step result.state"),
        events: result.events || [],
        info: result.info || {},
      };
    }

    #sendError(id, code, message) {
      this.#send({
        protocol: PROTOCOL,
        v: VERSION,
        id,
        ok: false,
        error: { code, message },
      });
    }

    #send(message) {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
        throw new Error("WebSocket is not open");
      }
      this.socket.send(JSON.stringify(message));
    }
  }

  global.PolyTrackTrainingBridge = PolyTrackTrainingBridge;
})(globalThis);

