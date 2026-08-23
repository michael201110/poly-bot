# PolyBot simulator protocol v1

The protocol is a synchronous request/response API transported as JSON over a local WebSocket.
Exactly one request may be in flight. A future transport may reuse the same message schema.

All messages contain:

```json
{
  "protocol": "polybot.sim",
  "v": 1,
  "id": 7
}
```

Requests add `op` and `params`:

```json
{
  "protocol": "polybot.sim",
  "v": 1,
  "id": 7,
  "op": "step",
  "params": {
    "episode_id": "polytrack-1",
    "action": { "steer": -1, "throttle": 1, "brake": 0 },
    "ticks": 4
  }
}
```

Successful and failed responses have one of these forms:

```json
{"protocol":"polybot.sim","v":1,"id":7,"ok":true,"result":{}}
```

```json
{
  "protocol": "polybot.sim",
  "v": 1,
  "id": 7,
  "ok": false,
  "error": { "code": "stale_episode", "message": "episode_id is stale or unknown" }
}
```

Every numeric value must be finite. Binary frames, `NaN`, and infinity are invalid.

## Operations

### `hello`

Negotiates a fixed lookahead size and reports stepping capabilities.

```json
{
  "op": "hello",
  "params": {
    "protocol": "polybot.sim",
    "protocol_version": 1,
    "lookahead_count": 12
  }
}
```

Result:

```json
{
  "protocol": "polybot.sim",
  "protocol_version": 1,
  "simulator": "polytrack-local",
  "game_version": "0.6.2",
  "fixed_dt_s": 0.0166666667,
  "max_ticks_per_step": 16,
  "lookahead_count": 12,
  "features": ["offline", "fixed_step", "ordered_progress"]
}
```

### `reset`

```json
{"op":"reset","params":{"seed":123,"track_id":"official/01"}}
```

The adapter must fully reset physics without advancing a tick. It creates a new `episode_id`, which
is required by every subsequent step. A stale command from a previous run is rejected.

### `step`

Adapters advertising the `action_sequence` feature also accept an `actions` array containing
exactly one legal digital action per requested tick. This batches PWM scheduling into one transport
round trip while preserving the game boundary: every entry still uses steering `-1`, `0`, or `+1`
and digital throttle/brake. Trainers fall back to constant-action requests for older adapters.

The adapter holds one digital action for exactly `ticks` physics updates. It may stop early if the
car finishes or crashes, in which case `ticks_advanced` is less than requested.

`steer` is one of `-1`, `0`, `1`; `throttle` and `brake` are each `0` or `1`. Restart is deliberately
not a policy action.

### `close`

Releases held controls and closes the bridge.

## Reset and step result

Both operations return the same shape. Reset returns `ticks_advanced: 0`.

```json
{
  "episode_id": "polytrack-1",
  "tick": 240,
  "ticks_advanced": 4,
  "state": {
    "position_m": [1.2, 0.4, 18.3],
    "quaternion_xyzw": [0.0, 0.1, 0.0, 0.995],
    "local_velocity_mps": [0.2, 0.0, 24.5],
    "angular_velocity_radps": [0.0, 0.3, 0.0],
    "up_vector": [0.0, 1.0, 0.0],
    "pitch_rad": 0.0,
    "roll_rad": 0.05,
    "wheel_contacts": [1, 1, 1, 1],
    "checkpoint_index": 1,
    "elapsed_s": 4.0,
    "previous_action": { "steer": 1, "throttle": 1, "brake": 0 },
    "track": {
      "progress_m": 83.1,
      "length_m": 320.0,
      "half_width_m": 5.0,
      "lateral_offset_m": -0.4,
      "heading_error_rad": 0.08,
      "lookahead": [
        [5.0, 0.1, 0.0, 0.002],
        [10.0, 0.4, 0.1, 0.004]
      ],
      "lookahead_mask": [1, 1]
    }
  },
  "events": ["checkpoint"],
  "info": { "track_id": "official/01", "off_track": false }
}
```

The real array lengths must equal the count negotiated in `hello`; the short example above is only
for readability. Invalid tail entries are zero-padded and have mask value `0`.

## Coordinates and units

- Distances are metres, time is seconds, and angles are radians.
- Quaternions are ordered XYZW.
- Car-local coordinates are right-handed: +X right, +Y up, +Z forward.
- Each lookahead row is `[forward_m, right_m, up_m, curvature_inverse_m]` in the car frame.
- `route_progress_m` must be monotonic along ordered track segments/checkpoints. Nearest-point-only
  progress is not sufficient on crossings or loops.
- `tick` is simulator time. Wall-clock time and `requestAnimationFrame` do not define training steps.

## Events

Core event strings are `checkpoint`, `off_track`, `finish`, `crash`, and `time_limit`. Repeated
`checkpoint` entries represent multiple checkpoints crossed within one action. The simulator emits
facts only; Python owns shaped rewards and the wrapper episode limit.

