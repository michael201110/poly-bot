from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport


def test_captured_initialization_reference_is_sent_on_later_resets():
    line = {"schema": "polybot.racing-line.v1", "points": [], "prefix_actions": [[0, 1, 0]]}

    class BootstrapTransport(MockSimulatorTransport):
        def _reset(self, params):
            result = super()._reset(params)
            if params.get("export_reference"):
                result["info"]["initialized_racing_line"] = line
            return result

    transport = BootstrapTransport()
    captured = []
    env = PolyTrackEnv(transport, on_racing_line=captured.append)
    try:
        env.reset(seed=1)
        env.reset(seed=2)
        hellos = [request for request in transport.command_log if request["op"] == "hello"]
        resets = [request for request in transport.command_log if request["op"] == "reset"]
        assert len(hellos) == 2
        assert hellos[1]["params"]["racing_line"] == line
        assert captured == [line]
        assert resets[0]["params"]["export_reference"] is True
        assert resets[1]["params"]["export_reference"] is False
    finally:
        env.close()


def test_existing_training_environment_switches_line_only_at_reset():
    original = {"schema": "polybot.racing-line.v1", "points": []}
    promoted = {**original, "prefix_actions": [[0, 1, 0]]}
    selected = [original]
    transport = MockSimulatorTransport()
    env = PolyTrackEnv(transport, racing_line=original, racing_line_provider=lambda: selected[0])
    try:
        env.reset(seed=1)
        selected[0] = promoted
        assert env.racing_line is original
        env.reset(seed=2)
        hellos = [request for request in transport.command_log if request["op"] == "hello"]
        assert hellos[-1]["params"]["racing_line"] == promoted
        assert env.racing_line is promoted
    finally:
        env.close()
