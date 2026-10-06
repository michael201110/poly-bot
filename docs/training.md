# Training PolyBot

The GUI, CLI, and saved metadata use the same typed `TrainingConfig`. GRTQC is the active Summer 1 path; TQC is the immutable reference and legacy experiment path, while PPO remains for historical experiments. The real simulator connects to Python at `ws://127.0.0.1:8765` after training starts. The local mock track supports fast installation checks.

Tracks use a persistent registry and separate model, replay, and log workspaces.
See [Track workspaces](tracks.md) for creating and selecting tracks, legacy
discovery, and the directory layout.

## Summer 1 GRTQC path

1. Keep `models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion/` unchanged. Its verified five-lap median is 24.263 seconds. Its policy hash is recorded in the GRTQC transfer report.
2. For a new experiment, run `python tools/initialize_grtqc.py --config profiles/training/summer-1-grtqc-precise-values-30.json --destination models/experiments/grtqc-precise-values-20261002/summer-1/grtqc/initialization` once. The tool copies compatible actor weights and saved overlays, initializes actor gates to exact pass-through, and checks 2,048 saved replay observations. Critics start fresh; source replay supplies the transfer check but is not copied into the new learner.
3. Use `polybot-train --config profiles/training/summer-1-grtqc-precise-values-30.json` or the GUI. Starting from initialization requires five paired live laps against the source before learning. Resuming a learner uses its saved raw replay and verifies a reference lap from the best reliable GRTQC driver.
4. The actor stays frozen during critic initialization. The current profile requires five complete episodes whose recorded actions match that policy, then fits their actual discounted returns for 4,000 critic-only updates. Both successful and failed complete episodes qualify; noisy/mixed-policy episodes, timeouts and interrupted attempts do not. Normal one-step TQC updates follow. Actor unlock requires at least 5,000 total critic updates, stable recent quantile loss and disagreement, and recorded-driving-return calibration error at most 20%. This calibration excludes entropy and quantile truncation and is not held-out lap performance.
5. All 64,132 actor parameters then learn with actor rate `3e-6`, critic rate `5e-5`, and correlated rollout exploration. The Gaussian used by critic targets and actor sampling is bounded to `1e-6`, with target entropy -28. These values follow a live sampling audit; they are not general-purpose defaults. Each actor step is backtracked if its deterministic change on the sampled replay batch exceeds `5e-5`. There is no cumulative teacher-action lock or prescribed replacement maneuver.
6. Frozen-policy checks use five full laps every 5,000 decisions. Learning-actor checks are due every 512 decisions; actor updates pause until the current attempt ends, preserving complete rewards in replay. A failed one-lap screen skips confirmation. A finishing policy needs five deterministic finishes before promotion. Five weaker checks restore the verified actor, preserve critic/replay progress, and repeat a 1,000-update critic cooldown with reference calibration. Only a reliable median faster than the source and any existing GRTQC champion promotes a champion; below 22.000 seconds remains the target.

The current cleaner candidate completed 5/5 laps at 24.485 seconds with one contact per lap, improving the previous 24.634-second cleaner driver. It remains slower than the source, and other updated snapshots failed. The current resumable learner retains 93,744 real transitions. See [the experiment record](grtqc-experiment.md) for the controlled comparisons and their limits.

The profile preserves 30 physics ticks per decision, source overlays and action execution. Actor and critics receive the four real PWM accumulators/directions in addition to the inherited 105 inputs. The inherited actor matrix is retained with a zero-initialized controller contribution for exact initial transfer. The reward charges 600 raw points for verified nonterminal barrier impacts, with impulse threshold 250 and landing grace two seconds; guidance reward is disabled. Changes to reward semantics require fresh replay. Old 105-input replay cannot be used with these 109-input models.

Resume with `polybot-train --config profiles/training/summer-1-grtqc-precise-values-30.json --resume latest`. The GUI's **Continue best model** chooses a verified GRTQC champion if present, otherwise latest. `latest/` retains the learner and replay; `champion/` retains the fastest verified policy; `contact-candidate/` retains a reliable cleaner stepping stone within the configured pace tolerance; rejected evaluation policies remain under `checkpoints/`. The source is never overwritten. Complete-return reference tensors are derived from replay; initialization counters and critic optimizers persist across resume.

The earlier 20-tick experiment remains separate. Its unchanged transferred source finished 5/5 laps at 24.616 seconds and learned snapshots failed. It is not evidence of faster control. Earlier 64-step return and first-chicane curriculum profiles are also retained for diagnosis.

## Algorithm details

GRTQC extends the installed SB3 Contrib TQC implementation. Its critic still predicts per-critic return quantiles, sorts the mixture and discards the highest configured quantiles for bootstrapped targets. Actor and critic hidden layers have a learned sigmoid gate applied after ReLU. Actor gates use `2 × sigmoid` with zero parameters for exact transfer identity; fresh critic gates use ordinary sigmoid initialized at 0.5. The new loss adds `λ × mean_batch,quantile variance_critics(Q_i)` to TQC's quantile Huber loss. This implements the gated feature modulation and variance-based ensemble consistency described in the [GRTQC paper](https://doi.org/10.1016/j.eswa.2026.132517). The actor gate scale, variance reduction axes, readiness rule and PolyBot hyperparameters are implementation choices because the publicly accessible paper text does not specify a runnable reference implementation. They are recorded here and tested rather than presented as an exact reproduction of unpublished source code.

The critic log includes quantile Huber loss, disagreement, weighted penalty, current quantile mean/standard deviation, target mean/standard deviation, actor unlock state, and update count. Compare these with deterministic lap results; a lower critic loss alone does not mean better driving.

## Evaluation and saved models

An environment step is one policy decision; `frame_skip` specifies fixed physics ticks per decision. TPS is decisions per wall-clock second, including simulator communication and optimizer updates. The continuous action adapter translates steering and signed longitudinal demand into PWM controls. GRTQC uses the same observation, action, collision, landing, reward and overlay semantics as the source TQC policy.

Each model slot contains `policy.zip` and `metadata.json`; resumable off-policy slots also contain `replay.pkl`. Metadata records schemas, track, architecture, complete configuration, counters, device, overlays, Git commit, and the most recent evaluation only if that exact policy was evaluated. Changing reward coefficients while resuming requires fresh replay. Evaluations are deterministic full-track laps and rank completion before time. A stochastic training finish cannot promote a model.

The GUI exposes all configuration fields under **Advanced settings**. `polybot-train --parameter-help` prints the same descriptions. `polybot-doctor --smoke grtqc` tests model construction and replay persistence; `polybot-eval --algorithm grtqc --track-name "Summer 1" --slot champion --backend websocket --episodes 5` tests a promoted policy. Legacy TQC and PPO controls remain available as separate experiment paths; they do not participate in GRTQC updates.

The GUI's **AI HUD** tab configures the live, non-physical in-game overlay for
WebSocket runs. It displays the actual policy observation and outputs,
controller-applied controls, reward breakdown, and episode events. The HUD is
separate from training and does not alter actions or rewards. See
[Live AI HUD](ai-overlay.md) for bridge-version requirements and timing limits.
