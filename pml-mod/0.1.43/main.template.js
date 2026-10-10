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
        if (event.data?.polybotHudInactive) {
          globalThis.__polybotHudUpdate?.clear?.("live");
        }
        const hudFrame = event.data?.polybotHudFrame;
        if (hudFrame) {
          try {
            if (["training", "evaluation", "manual_model_drive"].includes(hudFrame.mode)) {
              globalThis.__polybotStopVisualReplay?.();
            }
            globalThis.__polybotHudUpdate(hudFrame);
          } catch (error) {
            console.warn("[PolyBot] Could not update AI HUD", error);
          }
        }
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
    /* POLYBOT_HUD_RENDERER */
    globalThis.__polybotHudUpdate = installPolyBotHudOverlay();
    globalThis.__polybotHudUpdate();
    /* POLYBOT_REPLAY_RENDERER */
    installPolyBotReplayRenderer();
    pml.registerGlobalMixin({
      type: MixinType.INSERT,
      token: "(Oe = function (e, t) {",
      func: "if (globalThis.__polybotReplayTextureSharing) { const shared = globalThis.__polybotReplaySharedTexture?.lookup(e, t); if (shared) return shared; }",
    });
    pml.registerGlobalMixin({
      type: MixinType.INSERT,
      token: '((0, l.gn)(this, ge, "f").scene.add((0, l.gn)(this, me, "f")),',
      func: 'globalThis.__polybotReplayCaptureRoot?.(this, (0, l.gn)(this, me, "f")),',
    });
    pml.registerGlobalMixin({
      type: MixinType.REPLACEBETWEEN,
      tokenStart: "return i;\n              }),\n              (Ve = function (e) {",
      tokenEnd: "return i;\n              }),\n              (Ve = function (e) {",
      func: "if (globalThis.__polybotReplayTextureSharing) globalThis.__polybotReplaySharedTexture?.register(e, t, i);\n                return i;\n              }),\n              (Ve = function (e) {",
    });
    pml.registerGlobalMixin({
      type: MixinType.INSERT,
      token: "setAnimationLoop(e) {",
      func: `if (!this.polybotRenderStats) {
        const renderer = (0, i.gn)(this, S, "f");
        const render = renderer.render;
        let calls = 0, triangles = 0, cpuMs = 0;
        renderer.render = function (...args) {
          const start = performance.now();
          const result = render.apply(this, args);
          cpuMs += performance.now() - start;
          calls += this.info.render.calls;
          triangles += this.info.render.triangles;
          return result;
        };
        this.polybotRenderStats = () => {
          const result = { calls, triangles, render_cpu_ms: cpuMs,
            width: renderer.domElement.width, height: renderer.domElement.height };
          calls = 0; triangles = 0; cpuMs = 0;
          return result;
        };
      }`,
    });
    pml.registerGlobalMixin({
      type: MixinType.INSERT,
      token: "set audioVolume(e) {",
      func: "e = globalThis.__polybotReplayAudioVolume?.(this, e) ?? e;",
    });
    pml.registerGlobalMixin({
      type: MixinType.REPLACEBETWEEN,
      tokenStart: "t.car.setOpacity(i);",
      tokenEnd: "t.car.setOpacity(i);",
      func: "t.car.setOpacity(!t.car.polybotVisualReplay && globalThis.__polybotLoadedGhostOpacity?.active && globalThis.__polybotLoadedGhostOpacity?.enabled ? globalThis.__polybotLoadedGhostOpacity.value : i);",
    });
    pml.registerGlobalMixin({
      type: MixinType.REPLACEBETWEEN,
      tokenStart: "n.car.setOpacity(r)",
      tokenEnd: "n.car.setOpacity(r)",
      func: "n.car.setOpacity(!n.car.polybotVisualReplay && globalThis.__polybotLoadedGhostOpacity?.active && globalThis.__polybotLoadedGhostOpacity?.enabled ? globalThis.__polybotLoadedGhostOpacity.value : r)",
    });
    pml.registerGlobalMixin({
      type: MixinType.REPLACEBETWEEN,
      tokenStart: "const t = e.getFinishTime() ?? e.getTime();",
      tokenEnd: "const t = e.getFinishTime() ?? e.getTime();",
      // Override only the displayed time; replay playback never advances the player physics.
      func: "const t = globalThis.__polybotVisualReplayTime?.(e) ?? e.getFinishTime() ?? e.getTime();",
    });
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
      token: "getCarState() {",
      // Surface sampling for legacy visual replays must ignore all cars.
      func: `(0, l.gn)(this, me, "f").userData.polybotCar = true;
        this.polybotVisualEffects = () => ({
          skidmarks_enabled: (0, l.gn)(this, Re, "f")?.getSettingBoolean(12),
          skid_trails: (0, l.gn)(this, Pe, "f").length,
          skid_cooldowns: [...(0, l.gn)(this, Le, "f")],
        });`,
    });
    pml.registerGlobalMixin({
      type: MixinType.INSERT,
      token: 'update(e) {\n              const t = (0, R.gn)(this, ta, "m", Ps).call(this);',
      func: `globalThis.__polybotBindVisualReplayRenderer({
        owner: this,
        deltaSeconds: e,
        player: (0, R.gn)(this, Xa, "f"),
        three: M,
        scene: (0, R.gn)(this, la, "f").scene,
        addMaterial: (material) => (0, R.gn)(this, la, "f").addMaterial(material),
        removeMaterial: (material) => (0, R.gn)(this, la, "f").removeMaterial(material),
        getRenderStats: () => (0, R.gn)(this, la, "f").polybotRenderStats?.(),
        getCamera: () => (0, R.gn)(this, la, "f").camera,
        setCamera: (camera) => (0, R.gn)(this, la, "f").setCamera(camera),
        createCar: (state, withAudio = false) => {
          globalThis.__polybotReplayPendingRoot = null;
          globalThis.__polybotReplayTextureSharing = true;
          try {
            const car = new z.A(
              null, state, null, null,
              (0, R.gn)(this, la, "f"),
              withAudio ? (0, R.gn)(this, ca, "f") : null,
              (0, R.gn)(this, aa, "f"),
              (0, R.gn)(this, ra, "f"),
              (0, R.gn)(this, Ta, "f"),
              (0, R.gn)(this, ua, "f"),
              null,
            );
            car.polybotVisualReplay = true;
            car.polybotReplayRoot = globalThis.__polybotReplayPendingRoot;
            return car;
          } finally {
            globalThis.__polybotReplayTextureSharing = false;
            globalThis.__polybotReplayPendingRoot = null;
          }
        },
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
