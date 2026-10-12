# Pace rewards and connection recovery

## Reward audit (2026-10-12)

Saved Simple Track replay HUDs expose accumulated reward components. Two
completed laps with the same progress reward (4,052.65) and contact cost (-100)
illustrated an inversion: 34.224 seconds earned 12,728.85, while the slower
34.261-second lap earned 12,747.35. The speed and distance bonuses outweighed
the faster finish. This is evidence of reward misalignment, not proof that it
alone caused the training plateau.

The existing **Pace** preset now uses:

- Route progress: +2 per newly credited metre (existing progress accounting).
- Time: -50 per simulated second, throughout the lap and below the target time.
- Raw speed, distance-at-speed, corridor-speed and control-change rewards: zero.
- Finish: +1,200 plus the existing +5,000 exponential target-time bonus.
- Failure: -1,000 for crash, stall or off-track termination.
- Barrier contact: -100; barrier speed loss: -50 per m/s beyond the 1 m/s deadzone.

The preparation configuration keeps Simple Track's 32-second target, 16-step
return, frame skip 30, learning rates and five-lap promotion gates. It sets a
one-million-decision budget. Existing checkpoint rewards are not edited.
Use **Continue best with selected rewards** to restore the 34.199-second
champion's weights with fresh replay and the existing critic adaptation gate.
The previous Latest snapshot is archived before applying the new configuration.

Progress feedback is the existing credited-route-distance reward; this change
does not implement formal potential-based shaping or claim policy invariance
under discounting. Reward balance remains an experiment requiring verified laps.
Episode JSONL reward terms and groups now contain whole-episode totals, matching
the accumulated episode return instead of reporting only the terminal step.

## GUI training watchdog

Live GUI training automatically retries connection errors, timeouts and
`stale_episode` failures up to five times, waiting three seconds between attempts.
After learning begins, recovery requires a successfully saved checkpoint from
the failed attempt. A failed checkpoint write stops recovery rather than loading
stale state. Each retry retains the remaining session budget and curriculum,
replay, optimizer state, current actor and champion protection. An interrupted
episode or evaluation restarts; an uncertain physics action is never resent.
Transferred-model fidelity checks still repeat when required. TQC pace-polish
sessions retain their existing failure handling instead of automatic recovery.

Stop requests share one event across retries. CUDA errors, arbitrary protocol
errors and model errors are not retried. The WebSocket sends a protocol ping
every ten seconds with a sixty-second pong allowance; clean disconnects wake
pending requests immediately. A silent physics request still uses its existing
request deadline. Recovery requires the game's bridge to reconnect; this cannot
prevent browser crashes, sleep or tab unloading. No bridge plugin update is needed.

Validation for this change: Python syntax parsing and diff whitespace review.
No automated tests or deliberate simulator-disconnect test were run.
