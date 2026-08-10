/*
 * Version-specific integration template.
 *
 * Copy this file to `polytrack_game_api.js` and replace each method with hooks
 * for the local PolyTrack desktop build. Do not weaken ensureOffline().
 */
(function installPolyTrackGameApiStub(global) {
  "use strict";

  class PolyTrackGameApi {
    async ensureOffline() {
      // Required: disable leaderboard, multiplayer, analytics, and all game API
      // requests before returning true. Prefer intercepting the game's own API
      // wrapper instead of globally breaking local WebSocket traffic.
      throw new Error("TODO: install network guards for this PolyTrack version");
    }

    async describe() {
      // Return the actual fixed physics timestep and the maximum safe number of
      // ticks that can be advanced while one policy action is held.
      throw new Error("TODO: expose simulation capabilities");
    }

    async reset({ seed, trackId, lookaheadCount }) {
      void seed;
      void trackId;
      void lookaheadCount;
      // Load/reset locally and return { tick: 0, state, events: [], info }.
      throw new Error("TODO: reset the local simulation and collect telemetry");
    }

    async step({ action, ticks }) {
      void action;
      void ticks;
      // Hold the digital input for exactly `ticks` fixed simulation updates.
      // Return { tick, ticks_advanced, state, events, info }.
      throw new Error("TODO: synchronously step physics and collect telemetry");
    }

    async close() {
      // Release held controls and restore any reversible local hooks.
    }
  }

  global.PolyTrackTrainingGameApi = PolyTrackGameApi;
})(globalThis);

