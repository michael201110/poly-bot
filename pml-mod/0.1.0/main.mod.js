import {
  MixinType,
  PolyMod,
} from "https://cdn.polymodloader.com/pml/PolyModLoader/0.6.2/PolyTypes.js";

// Keep this import absolute: PML's mod cache reloads main.mod.js from a Blob
// URL, where a relative module specifier cannot be resolved.
import { polybotWorkerInjection } from "https://cdn.polymodloader.com/gh/michael201110/poly-bot/v0.1.0/pml-mod/0.1.0/worker_runtime.js";

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

    // Native frame numbers return to the beginning after PolyBot recreates a
    // car. Tell the renderer to treat that regression as a true reset so its
    // interpolation cache and cameras do not retain the previous episode.
    pml.registerGlobalMixin({
      type: MixinType.REPLACEBETWEEN,
      tokenStart: "this.setCarState(e, !1);",
      tokenEnd: "this.setCarState(e, !1);",
      func: "this.setCarState(e, e.frames < this.getCarState().frames);",
    });

    // Read-only leaderboard calls remain available so the user can select a
    // reference ghost. Every write endpoint and multiplayer entry point is
    // disabled while the mod is loaded. The worker also masks its finish bit
    // before publishing state to the game UI.
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
