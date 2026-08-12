import {
  MixinType,
  PolyMod,
} from "https://cdn.polymodloader.com/pml/PolyModLoader/0.6.2/PolyTypes.js";

import { polybotWorkerInjection } from "https://cdn.polymodloader.com/gh/michael201110/poly-bot/v0.1.15/pml-mod/0.1.0/worker_runtime.js";

class PolyBotBridgeMod extends PolyMod {
  touchingPhysics = true;

  preInit = (pml) => {
    if (typeof pml?.registerSimWorkerMixin !== "function") {
      throw new Error(
        "PolyBot Bridge requires PolyModLoader 0.6.2 with simulation-worker mixins.",
      );
    }

    globalThis.__polybotWrapSimulationWorker = (worker) => {
      let restartScheduled = false;
      const pressBackspace = () => {
        const eventOptions = {
          key: "Backspace",
          code: "Backspace",
          keyCode: 8,
          which: 8,
          bubbles: true,
          cancelable: true,
        };
        const keyboardTarget = document.body ?? document;
        keyboardTarget.dispatchEvent(new KeyboardEvent("keydown", eventOptions));
        keyboardTarget.dispatchEvent(new KeyboardEvent("keyup", eventOptions));
      };
      worker.addEventListener("message", (event) => {
        if (
          (event.data?.polybotAbortRestart !== true &&
            event.data?.polybotPlayerFinished !== true) ||
          restartScheduled
        ) {
          return;
        }
        restartScheduled = true;
        setTimeout(() => {
          pressBackspace();
          restartScheduled = false;
        }, 500);
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
      func: "this.setCarState(e, e.frames < this.getCarState().frames);",
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
        type: MixinType.INSERT,
        token,
        func: 'return Promise.reject(new Error("PolyBot offline mode"));',
      });
    }

    for (const token of [
      "createMultiplayerHostWebSocket() {",
      "createMultiplayerJoinWebSocket() {",
    ]) {
      pml.registerGlobalMixin({
        type: MixinType.INSERT,
        token,
        func: 'throw new Error("PolyBot offline mode");',
      });
    }
  };
}

export const polyMod = new PolyBotBridgeMod();
