from __future__ import annotations

import json
from dataclasses import fields, replace

from polybot.env import RewardConfig, summer_1_pace_reward_config
from polybot.training.reward_profiles import RewardProfileStore


def test_custom_reward_profiles_round_trip_every_reward_field(tmp_path) -> None:
    store = RewardProfileStore(tmp_path)
    expected = replace(summer_1_pace_reward_config(), progress_per_m=12.5)
    path = store.save("My Pace Profile", expected)
    assert path.name == "my-pace-profile.json"
    actual = store.load("My Pace Profile")
    assert actual == expected
    assert {field.name for field in fields(RewardConfig)} == set(
        json.loads(path.read_text(encoding="utf-8"))["rewards"]
    )
    assert "My Pace Profile" in store.names()
