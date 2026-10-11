# Desktop workspace

Launch with `polybot-gui` (or `.venv\Scripts\python.exe -m polybot.gui.main`).

Choose the **track and algorithm in the header**. These choose the model workspace
used by training, evaluation, and live driving. Replay has its own algorithm
filter so you can browse recordings from other algorithms on the same track.

## Everyday tasks

- **Overview** shows the champion's evaluated median and completion count beside
  the latest checkpoint. Latest is the most recently saved learner and can be
  slower or unevaluated.
- **Continue best model** restores the best compatible checkpoint's architecture,
  curriculum, and rewards. Set the next session's budget in **Run setup** first.
- **Play champion** drives the selected algorithm's champion in the live game.
  Open PolyTrack with the PolyBot bridge; follow output in **Activity & logs**.
- **Evaluate champion** runs a deterministic evaluation and records its attempts.
- For a new model, use **Guided setup**, or edit **Run setup**, **Learning settings**,
  **Rewards**, **Curriculum**, and **Evaluation**, then start a new run.
- **Replay** lets you watch a highlighted attempt or check several attempts to
  compare them as a swarm. Its advanced section contains filters and appearance.
- **In-game display** configures observations, controls, and rewards in the HUD.

Existing models use their saved racing line. Load an initialization ghost when
initializing a new model. Starting the GUI does not start training or playback.

## Pure Pace rewards

Choose **Pure Pace** in **Rewards** for a model that already finishes laps.
Set **Target lap time** to your next milestone (the profile defaults to 32 seconds).
Use **Continue best with selected rewards** to apply it to a saved champion with
fresh replay; ordinary continuation restores the checkpoint's saved rewards.

The profile costs 10 points per simulated second. A valid finish earns 2,000
points plus `5000 * exp(-0.15 * max(lap_seconds - target_seconds, 0))`.
This makes the raw score of completed laps strictly better for faster times,
including times below the target. Progress, speed, ghost matching, checkpoints,
braking, smoothness, and nonterminal contact have no separate reward.
Crash, stall, and off-track failures cost 1,000 points; unfinished timeouts lose
all finish credit. Existing termination rules and champion evaluation still apply.
This profile is a new experiment, not a verified improvement over Pace.

## Finding controls

**Rewards → Barrier speed-loss penalty (per m/s)** adds a cost for net planar
speed lost over a control decision containing a detected barrier impact.
Pace uses -50 points per m/s beyond a 1 m/s deadzone: a 10 m/s loss costs an
extra 450 points. Pace + failure x3 uses -150. The deadzone is adjustable in
advanced Rewards. Braking without contact and impacts during landing grace
do not incur this cost. This estimates loss during contact from consecutive
observations; it cannot separate simultaneous braking from the impact or recover
speed losses hidden by acceleration within the decision.

Existing checkpoints retain their saved rewards. To enable this on a trained
model, set the coefficient and use **Continue best with selected rewards** or
**Continue latest with selected rewards**. Changed reward settings require fresh
replay; GRTQC holds actor updates through the configured recovery cooldown while
the critics adapt. Ordinary continuation with the coefficient disabled can reuse
replay from checkpoints saved before this option existed.

GRTQC recovery keeps the saved critics and replay while restoring a verified actor.
The actor stays frozen for the full configured recovery cooldown, even when the
initial critic warmup is shorter. Detailed JSONL progress includes
`critic_recovery_updates_remaining`; this counter survives checkpoint continuation.
Episode reward totals accumulate across training chunks and checkpoint saves, and
reset when an attempt ends or evaluation resets the simulator.

Press **Ctrl+K** and search for a setting or action, such as `actor learning rate`,
`frame skip`, or `port`. Press Enter to open the selected result, or use Down and
the arrow keys to choose another result. Search opens the correct page, reveals
advanced settings when needed, and scrolls to the control. It never runs an action
merely because you selected a search result. Escape clears the search field.

**Advanced settings** reveals every secondary training field and the research
workflows in Models. Each workflow expands independently. **Show archived
checkpoints** includes saved and rejected snapshots in the model inventory;
file locations remain under **Checkpoint details and file locations**.

Load and save exact configurations from the sidebar. Shortcuts are **Ctrl+O**
and **Ctrl+Shift+S**. Loading a configuration also selects its registered track.

## While a task is running

The activity bar stays visible on every page. It identifies the active task and
shows session progress during training. **View activity** opens metrics and logs;
the detailed metric list can expand separately from the event log.

**Stop cleanly** asks training and searches to save and stop safely. It cancels a
live drive or evaluation. Commands without a safe stop interface finish normally;
the stop button explains when it is unavailable. Simulator tasks cannot compete
for the bridge. Track and algorithm selection are locked during an active task.

Errors appear in a selectable, dismissible message above the workspace, with
details also in Activity & logs. Closing during an active task waits for safe
completion. Choose **Keep open** to cancel the pending window close.
