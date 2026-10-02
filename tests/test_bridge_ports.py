from pathlib import Path

MOD_ROOT = Path(__file__).resolve().parents[1] / "pml-mod"


def test_worker_parallel_ports_are_validated_and_loopback_only(tmp_path):
    import subprocess

    worker = (MOD_ROOT / "0.1.32" / "worker_runtime.js").read_text()
    code = worker.split("        const requestedBridgePort", 1)[1].split("        const fixedDtSeconds", 1)[0]
    code = "const requestedBridgePort" + code
    checks = """
const cases = [[undefined,8765],[8766,8766],[-1,8765],[65536,8765],['evil.example',8765],[1.5,8765]];
for (const [value, expected] of cases) {
  const r = {polybotBridgePort:value};
  const actual = (() => { CODE; return bridgeUrl; })();
  if (actual !== 'ws://127.0.0.1:' + expected) throw new Error(actual);
}
""".replace("CODE", code)
    path = tmp_path / "ports.js"
    path.write_text(checks)
    subprocess.run(["node", str(path)], check=True, capture_output=True, text=True)
