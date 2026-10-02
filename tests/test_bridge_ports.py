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


def test_native_instance_guard_is_scoped_to_each_simulator_port(tmp_path):
    import json
    import subprocess

    main = (MOD_ROOT / "0.1.33" / "main.mod.js").read_text()
    main = "class PolyBotBridgeMod" + main.split("class PolyBotBridgeMod", 1)[1]
    main = main.replace("export const polyMod", "const polyMod")
    checks = """
class PolyMod {}
const MixinType = {INSERT:1,REPLACEBETWEEN:2};
const polybotWorkerInjection = () => {};
MAIN
function instanceChannel(port) {
  globalThis.location = {href:'https://web.polymodloader.com/?polybotPort='+port};
  const mixins = [];
  polyMod.preInit({registerSimWorkerMixin() {}, registerGlobalMixin(m) {mixins.push(m);}});
  const mixin = mixins.find(m => m.tokenStart === '"polytrack-single-instance"');
  if (!mixin || mixin.tokenEnd !== mixin.tokenStart) throw new Error('missing native guard');
  return JSON.parse(mixin.func);
}
if (instanceChannel(8765) !== instanceChannel(8765)) throw new Error('same port must conflict');
if (instanceChannel(8765) === instanceChannel(8766)) throw new Error('independent ports conflict');
if (instanceChannel(-1) !== instanceChannel(8765)) throw new Error('invalid port not defaulted');
""".replace("MAIN", main)
    path = tmp_path / "instance-channels.js"
    path.write_text(checks)
    subprocess.run(["node", str(path)], check=True, capture_output=True, text=True)
    manifest = json.loads((MOD_ROOT / "manifest.json").read_text())
    assert manifest["latest"]["0.6.3"] == "0.1.33"
    assert (MOD_ROOT / "0.1.33" / "worker_runtime.js").read_bytes() == (
        MOD_ROOT / "0.1.32" / "worker_runtime.js"
    ).read_bytes()
