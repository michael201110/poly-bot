import {
  MixinType,
  PolyMod,
} from "https://cdn.polymodloader.com/pml/PolyModLoader/0.6.2/PolyTypes.js";

import { polybotWorkerInjection } from "https://cdn.polymodloader.com/gh/michael201110/poly-bot/v0.1.4/pml-mod/0.1.0/worker_runtime.js";

class PolyBotBridgeMod extends PolyMod {
  touchingPhysics = true;

  preInit = (pml) => {
    if (typeof pml?.registerSimWorkerMixin !== "function") {
      throw new Error(
        "PolyBot Bridge requires PolyModLoader 0.6.2 with simulation-worker mixins.",
      );
    }

    pml.registerSimWorkerMixin({
      type: MixinType.INSERT,
      token: "const r = i.data;",
      func: polybotWorkerInjection,
    });

    pml.registerGlobalMixin({
      type: MixinType.REPLACEBETWEEN,
      tokenStart: "this.setCarState(e, !1);",
      tokenEnd: "this.setCarState(e, !1);",
      func: "this.setCarState(e, e.frames < this.getCarState().frames);",
    });

    // PolyBot resets native frame numbers. The vanilla control recorder
    // requires globally increasing frames and is only needed for run uploads,
    // which this mod deliberately disables.
    pml.registerGlobalMixin({
      type: MixinType.INSERT,
      token: "recordFrame(e, t) {",
      func: "return;",
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
