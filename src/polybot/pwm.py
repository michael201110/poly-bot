"""Deterministic pulse-density steering scheduler."""

from __future__ import annotations


class PwmSteering:
    def __init__(self) -> None:
        self._error = 0.0
        self._direction = 0

    def reset(self) -> None:
        self._error = 0.0
        self._direction = 0

    def generate(self, steering: float, ticks: int) -> list[int]:
        if not -1.0 <= steering <= 1.0:
            raise ValueError("steering must be in [-1, 1]")
        if ticks < 1:
            raise ValueError("ticks must be positive")
        direction = (steering > 0) - (steering < 0)
        duty = abs(float(steering))
        if direction != self._direction:
            self._error = 0.0
            self._direction = direction
        if direction == 0:
            return [0] * ticks
        if duty == 1.0:
            return [direction] * ticks
        result: list[int] = []
        for _ in range(ticks):
            self._error += duty
            pulse = self._error >= 1.0
            if pulse:
                self._error -= 1.0
            result.append(direction if pulse else 0)
        return result


def decode_pwm_level(level: int, levels: int) -> float:
    if levels < 3 or levels % 2 == 0 or not 0 <= level < levels:
        raise ValueError("invalid PWM level or resolution")
    return -1.0 + 2.0 * level / (levels - 1)
