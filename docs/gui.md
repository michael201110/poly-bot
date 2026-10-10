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

## Finding controls

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
