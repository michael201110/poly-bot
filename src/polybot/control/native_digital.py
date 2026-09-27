"""Nine stable keyboard-style controls for DQN, held for every physics tick."""

from __future__ import annotations

from gymnasium import spaces

from polybot.control.actions import AppliedAction, ControlDemand
from polybot.protocol import Action


class NativeDigitalActionAdapter:
    """Map a Discrete(9) policy action directly to native digital controls."""

    schema = "digital-discrete-9-v2"
    sequence = False

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(9)

    def reset(self) -> None:
        # No pulse phase or other control state exists.
        pass

    def apply(self, action: int, ticks: int) -> AppliedAction:
        if not self.action_space.contains(action):
            raise ValueError(f"digital action must be an integer from 0 to 8, got {action!r}")
        index = int(action)
        if ticks < 1:
            raise ValueError("digital action must be held for at least one tick")
        steer = (0, -1, 1)[index // 3]
        pedal = index % 3
        digital = Action(steer, pedal == 1, pedal == 2)
        return AppliedAction(ControlDemand.from_action(digital), [digital] * ticks)
