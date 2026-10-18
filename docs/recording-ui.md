# Recording UI

## Compatible releases

| Plugin | Release |
| --- | --- |
| PolyBot Bridge | 0.1.50 |
| PolyCam | 0.1.12 |
| PolyShade | 0.5.3 |

Each plugin works independently. When installed together, the panels use matching liquid glass styling and reserve space for PolyBot's HUD selector. PolyCam and PolyShade dock alongside one another on desktop and stack on narrow screens. Controls remain scrollable; no settings were removed.

## Recording

- **F2** hides or restores all three plugins' windows and HUD controls. The selected AI HUD remains visible, without its dropdown, toggle, move grip or resize handle.
- Choose **None** in PolyBot's AI HUD dropdown to hide the HUD itself. None plus F2 gives a clean view without these plugins' overlays.
- The **Hide windows for recording (F2)** checkboxes in PolyCam/PolyShade and action in PolyBot's dropdown invoke the same shared state. Press F2 to restore.
- Restore preserves each window's own hidden/collapsed state and the chosen HUD mode. **Show AI HUD** restores the last visible HUD mode after None.
- Recording visibility is session-only and does not pause replay playback, camera movement, rendering or training. Native game UI is unaffected.
- PolyCam's F4 window toggle and F9 Clean Capture remain available. PolyShade's End panel shortcut and Home enable shortcut are unchanged.

## Optional integration contract

`globalThis.polyCaptureUI` exposes `apiVersion: 1`, `hidden`, `setHidden(boolean)` and `subscribe(listener)` (returns an unsubscribe function). Subscribers share one F2 listener and the root attribute `data-poly-capture-ui="hidden"`. The last subscriber's disposal removes the listener and attribute. No plugin depends on another being installed; the first participant creates the service.

This release changes UI only: physics, replay interpolation/timing, learner semantics, camera rigs and rendering quality are unchanged.

## Release checks

All three release bundles built successfully. Live Edge/PML review confirmed matching glass panels, readable controls, no horizontal panel overflow, F2 hide/restore while retaining the Controls HUD, the dropdown recording action, None hiding the HUD after its fade, and Show AI HUD restoring the previous mode. No training was started. No automated test suite was run for this UI-only release.
