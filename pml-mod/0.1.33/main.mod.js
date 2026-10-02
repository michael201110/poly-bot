import {
  MixinType,
  PolyMod,
} from "https://cdn.polymodloader.com/pml/PolyModLoader/0.6.2/PolyTypes.js";

import { polybotWorkerInjection } from "https://cdn.polymodloader.com/gh/michael201110/poly-bot/4eeebdce2bea6b80cb961953bad4da1a17614657/pml-mod/0.1.32/worker_runtime.js";

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
      // The native recorder requires strictly increasing frame IDs. Treat an
      // equal frame as a rewind/reset too, so it does not append a duplicate.
      func: "this.setCarState(e, e.frames <= this.getCarState().frames);",
    });
    pml.registerGlobalMixin({
      type: MixinType.INSERT,
      token: '(0, l.GG)(this, Ue, null, "f"),',
      func: '(0, l.GG)(this, re, new st.A(), "f"),',
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
