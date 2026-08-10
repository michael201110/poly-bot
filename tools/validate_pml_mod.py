"""Validate PolyBot's source anchors against a local PolyTrack 0.6.2 build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PINNED_WORKER_SHA256 = "2a7e2be7306af8a23488bfe2c07cd32df2ac728bc679163327cb3938eb3b5ef5"
PINNED_MAIN_SHA256 = "e5687766fde6f5bf483bb316ea5c6d55aa9b9edc64e55cf5e527807a81a5c006"

WORKER_TOKENS = ("const r = i.data;",)
MAIN_TOKENS = (
    "this.setCarState(e, !1);",
    "recordFrame(e, t) {",
    "submitLeaderboard(e, t, n, i, r, a, s, o) {",
    "submitUserProfile(e, t, n, i) {",
    "verifyRecordings(e, t, n, i, r) {",
    "getIceServers() {",
    "createMultiplayerHostWebSocket() {",
    "createMultiplayerJoinWebSocket() {",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_tokens(path: Path, tokens: tuple[str, ...]) -> list[str]:
    source = path.read_text(encoding="utf-8")
    failures = []
    for token in tokens:
        count = source.count(token)
        if count != 1:
            failures.append(f"{path}: expected one {token!r} anchor, found {count}")
    return failures


def validate(worker: Path, main: Path, *, require_pinned_hash: bool = True) -> list[str]:
    """Return compatibility failures; an empty result means validation passed."""

    failures = [*_validate_tokens(worker, WORKER_TOKENS), *_validate_tokens(main, MAIN_TOKENS)]
    if require_pinned_hash:
        worker_hash = _sha256(worker)
        main_hash = _sha256(main)
        if worker_hash != PINNED_WORKER_SHA256:
            failures.append(
                f"{worker}: SHA-256 {worker_hash} is not the pinned 0.6.2 worker hash"
            )
        if main_hash != PINNED_MAIN_SHA256:
            failures.append(f"{main}: SHA-256 {main_hash} is not the pinned 0.6.2 main hash")
    return failures


def _validate_manifests(repository: Path) -> list[str]:
    failures = []
    mod_root = repository / "pml-mod"
    manifest = json.loads((mod_root / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("latest", {}).get("0.6.2")
    if not isinstance(version, str):
        return ["pml-mod/manifest.json does not map PolyTrack 0.6.2 to a mod version"]
    version_manifest = json.loads(
        (mod_root / version / "version.json").read_text(encoding="utf-8")
    )
    if version_manifest.get("targets") != ["0.6.2"]:
        failures.append("the current mod version must target only PolyTrack 0.6.2")
    main_file = mod_root / version / str(version_manifest.get("main"))
    if not main_file.is_file():
        failures.append(f"missing mod entry point: {main_file}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help="path to simulation_worker.bundle.js")
    parser.add_argument("--main", type=Path, help="path to main.bundle.js")
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
            validate(args.worker, args.main, require_pinned_hash=not args.anchors_only)
        )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("PolyBot PML manifest and bundle anchors are compatible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
