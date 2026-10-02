"""Policy actions, physical control demand, and digital tick schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from gymnasium import spaces

from polybot.control.pwm import ContinuousPwmControls
from polybot.protocol import Action


@dataclass(frozen=True, slots=True)
class ControlDemand:
    steer: float
    throttle: float
    brake: float

    def __post_init__(self) -> None:
        if not -1 <= self.steer <= 1 or not 0 <= self.throttle <= 1 or not 0 <= self.brake <= 1:
            raise ValueError("control demand is outside normalized bounds")
        if self.throttle and self.brake:
            raise ValueError("throttle and brake cannot overlap")

    @classmethod
    def from_action(cls, action: Action) -> ControlDemand:
        # External telemetry can report both keys pressed at a curriculum spawn.
        # Match DigitalActionAdapter's established brake-priority behavior.
        brake = float(action.brake)
        throttle = 0.0 if brake else float(action.throttle)
        return cls(float(action.steer), throttle, brake)

    @classmethod
    def from_continuous(cls, steer: float, longitudinal: float) -> ControlDemand:
        return cls(steer, max(0.0, longitudinal), max(0.0, -longitudinal))


@dataclass(slots=True)
class AppliedAction:
    demand: ControlDemand
    ticks: list[Action]


class ActionAdapter(Protocol):
    action_space: spaces.Space
    schema: str
    sequence: bool

    def reset(self) -> None: ...
    def apply(self, action: np.ndarray, ticks: int) -> AppliedAction: ...


class DigitalActionAdapter:
    schema = "digital-v2"
    sequence = False

    def __init__(self) -> None:
        self.action_space = spaces.MultiDiscrete(np.asarray([3, 2, 2]))

    def reset(self) -> None:
        pass

    def apply(self, action: np.ndarray, ticks: int) -> AppliedAction:
        digital = Action.from_policy(action)
        if digital.throttle and digital.brake:
            digital = Action(digital.steer, False, True)
        return AppliedAction(ControlDemand.from_action(digital), [digital] * ticks)


class ContinuousActionAdapter:
    schema = "continuous-pwm-v2"
    sequence = True

    def __init__(self, *, expose_controller_state: bool = False) -> None:
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._controls = ContinuousPwmControls()
        self.expose_controller_state = expose_controller_state

    def observation_state(self) -> tuple[float, ...]:
        return self._controls.state() if self.expose_controller_state else ()

    def reset(self) -> None:
        self._controls.reset()

    def apply(self, action: np.ndarray, ticks: int) -> AppliedAction:
        steer, longitudinal = (float(value) for value in action)
        return AppliedAction(
            ControlDemand.from_continuous(steer, longitudinal),
            [Action(*values) for values in self._controls.generate(steer, longitudinal, ticks)],
        )
