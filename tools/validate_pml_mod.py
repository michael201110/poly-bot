"""Validate PolyBot manifests and optional raw PolyModLoader game bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Raw upstream bytes, before PolyModLoader applies its built-in mixins.
# Source revisions and reproduction commands: docs/game-integration.md.
PINNED_HASHES = {
    "0.6.2": (
        "12b084a9159e29fb3e1dd1b96eeffcf873f00baa00df9bacef0c4a29bd6c554b",
        "7ff0bc02f1b42b55ad38c140872cc81182b3e0632888c5376427d73b74e40783",
    ),
    "0.6.3": (
        "8ba0cd41f146ebc05f961a8d55ca9f7e939792b2581739ba1d1cba64beac2bb2",
        "8004a526fcb61bc8d202b307112aad704f27328045fdbce161b866ebd242ada5",
    ),
}
DEFAULT_GAME_VERSION = "0.6.3"
RAW_WORKER_CONSTRUCTOR = 'new Worker("simulation_worker.bundle.js")'
PML_WORKER_CONSTRUCTOR = "new Worker(ActivePolyModLoader.getSimURL())"

WORKER_TOKENS = ("const r = i.data;",)
MAIN_TOKENS = (
    PML_WORKER_CONSTRUCTOR,
    "this.setCarState(e, !1);",
    '(0, l.GG)(this, Ue, null, "f"),',
    "submitLeaderboard(e, t, n, i, r, a, s, o) {",
    "submitUserProfile(e, t, n, i) {",
    "verifyRecordings(e, t, n, i, r) {",
    "getIceServers() {",
    "createMultiplayerHostWebSocket() {",
    "createMultiplayerJoinWebSocket() {",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_tokens(path: Path, tokens: tuple[str, ...], *, source: str) -> list[str]:
    failures = []
    for token in tokens:
        count = source.count(token)
        if count != 1:
            failures.append(f"{path}: expected one {token!r} anchor, found {count}")
    return failures


def validate(
    worker: Path, main: Path, *, game_version: str = DEFAULT_GAME_VERSION,
    require_pinned_hash: bool = True,
) -> list[str]:
    """Return compatibility failures; an empty result means validation passed."""

    if game_version not in PINNED_HASHES:
        return [f"unsupported PolyTrack version: {game_version}"]
    main_source = main.read_text(encoding="utf-8")
    # Reproduce PML's built-in URL replacement before checking our mod's anchors.
    main_source = main_source.replace(RAW_WORKER_CONSTRUCTOR, PML_WORKER_CONSTRUCTOR)
    failures = [
        *_validate_tokens(worker, WORKER_TOKENS, source=worker.read_text(encoding="utf-8")),
        *_validate_tokens(main, MAIN_TOKENS, source=main_source),
    ]
    if require_pinned_hash:
        worker_hash = _sha256(worker)
        main_hash = _sha256(main)
        pinned_worker, pinned_main = PINNED_HASHES[game_version]
        if worker_hash != pinned_worker:
            failures.append(
                f"{worker}: SHA-256 {worker_hash} is not the pinned {game_version} worker hash"
            )
        if main_hash != pinned_main:
            failures.append(
                f"{main}: SHA-256 {main_hash} is not the pinned {game_version} main hash"
            )
    return failures


def _validate_manifests(repository: Path) -> list[str]:
    failures = []
    mod_root = repository / "pml-mod"
    manifest = json.loads((mod_root / "manifest.json").read_text(encoding="utf-8"))
    latest = manifest.get("latest", {})
    for game_version in dict.fromkeys((*PINNED_HASHES, *latest)):
        if game_version not in PINNED_HASHES:
            failures.append(f"no bundle checks defined for PolyTrack {game_version}")
        version = latest.get(game_version)
        if not isinstance(version, str):
            failures.append(f"manifest does not map PolyTrack {game_version} to a mod version")
            continue
        try:
            version_manifest = json.loads(
                (mod_root / version / "version.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            failures.append(f"cannot read mod {version} manifest: {error}")
            continue
        if game_version not in version_manifest.get("targets", []):
            failures.append(f"mod {version} does not target PolyTrack {game_version}")
        main_file = mod_root / version / str(version_manifest.get("main"))
        if not main_file.is_file():
            failures.append(f"missing mod entry point: {main_file}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help="path to simulation_worker.bundle.js")
    parser.add_argument("--main", type=Path, help="path to main.bundle.js")
    parser.add_argument("--game-version", choices=PINNED_HASHES, default=DEFAULT_GAME_VERSION)
    parser.add_argument(
        "--anchors-only",
        action="store_true",
        help="accept another bundle hash if every exact source anchor still matches once",
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    failures = _validate_manifests(repository)
    if (args.worker is None) != (args.main is None):
        parser.error("--worker and --main must be supplied together")
    if args.worker is not None and args.main is not None:
        failures.extend(
            validate(
                args.worker, args.main, game_version=args.game_version,
                require_pinned_hash=not args.anchors_only,
            )
        )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    if args.worker is None:
        print("PolyBot PML manifests are valid. Game bundles were not checked.")
    else:
        checks = "anchors only (hashes not checked)" if args.anchors_only else "hashes and anchors"
        print(f"PolyBot PML manifests and PolyTrack {args.game_version} bundle {checks} passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
