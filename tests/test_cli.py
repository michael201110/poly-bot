from types import SimpleNamespace

from polybot.cli import _configure_saved_overlays


def test_saved_ppo_overlays_are_restored_to_runner_and_policy():
    runner = SimpleNamespace(
        config=SimpleNamespace(algorithm="ppo"),
        _ppo_air_brake_overlays=[],
        _ppo_speed_bias_schedule=[],
    )
    model = SimpleNamespace()
    metadata = SimpleNamespace(
        policy_overlays=[{"kind": "air_brake", "start": 0.7, "end": 0.8, "duty": 1.0}],
        speed_bias_schedule=[[0.5, 0.7, 0.01]],
    )

    _configure_saved_overlays(runner, model, metadata)

    assert runner._ppo_air_brake_overlays == metadata.policy_overlays
    assert runner._ppo_speed_bias_schedule == metadata.speed_bias_schedule
    assert model.policy_overlays == metadata.policy_overlays
    assert model.speed_bias_schedule == metadata.speed_bias_schedule


def test_saved_ppo_overlays_can_configure_runner_before_model_load():
    runner = SimpleNamespace(
        config=SimpleNamespace(algorithm="ppo"),
        _ppo_air_brake_overlays=[],
        _ppo_speed_bias_schedule=[],
    )
    metadata = SimpleNamespace(
        policy_overlays=[{"kind": "air_brake", "start": 0.7, "end": 0.8, "duty": 1.0}],
        speed_bias_schedule=[[0.5, 0.7, 0.01]],
    )

    _configure_saved_overlays(runner, None, metadata)

    assert runner._ppo_air_brake_overlays == metadata.policy_overlays
    assert runner._ppo_speed_bias_schedule == metadata.speed_bias_schedule


def test_empty_tqc_metadata_does_not_clear_serialized_speed_bias_schedule():
    runner = SimpleNamespace(config=SimpleNamespace(algorithm="tqc"))
    model = SimpleNamespace(
        speed_bias_schedule=[[0.5, 0.75, 0.1]],
    )
    metadata = SimpleNamespace(policy_overlays=[], speed_bias_schedule=[])

    _configure_saved_overlays(runner, model, metadata)

    assert model.policy_overlays == []
    assert model.speed_bias_schedule == [[0.5, 0.75, 0.1]]
