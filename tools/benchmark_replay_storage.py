"""Compare lossless replay codecs using copied recorded buffers or synthetic arrays.

Never rewrites a source replay. Temporary files include fsync; decoded pickle
bytes must have the same SHA256 for every codec. Reports are not CI thresholds.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import pickle
import platform
import subprocess
import tempfile
import time
import tracemalloc
import types
from pathlib import Path

import numpy as np

from polybot.training import replay_storage

try:
    from compression import zstd
except ImportError:
    from backports import zstd


def codec(raw, name, mode):
    if name.startswith("gzip"):
        return gzip.GzipFile(fileobj=raw, mode=mode, compresslevel=int(name[-1]), mtime=0)
    level = int(name[4])
    options = {zstd.CompressionParameter.compression_level: level, zstd.CompressionParameter.checksum_flag: 1}
    if name.endswith("-2w"):
        options[zstd.CompressionParameter.nb_workers] = 2
    return zstd.ZstdFile(raw, mode, options=options if mode == "wb" else None)


class BufferModel:
    def __init__(self, buffer):
        self.buffer = buffer

    def save_replay_buffer(self, stream):
        from stable_baselines3.common.save_util import save_to_pkl
        save_to_pkl(stream, self.buffer)

    def load_replay_buffer(self, stream):
        from stable_baselines3.common.save_util import load_from_pkl
        self.restored = load_from_pkl(stream)


def production_comparison(buffer, path, size, repeats, digest):
    baseline = types.ModuleType("baseline_replay_storage")
    exec(subprocess.check_output([
        "git", "show", "5115240:src/polybot/training/replay_storage.py",
    ], text=True), baseline.__dict__)
    result = []
    for name, module in (("baseline-gzip3", baseline), ("current-zstd3-2w", replay_storage)):
        model = BufferModel(buffer)
        saves, loads = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            module.save_replay_buffer(model, path)
            saves.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            module.load_replay_buffer(model, path)
            loads.append((time.perf_counter() - start) * 1000)
            del model.restored
        tracemalloc.start()
        module.save_replay_buffer(model, path)
        _, save_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.start()
        module.load_replay_buffer(model, path)
        _, load_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del model.restored
        with path.open("rb") as raw:
            magic = raw.read(4)
            raw.seek(0)
            with (gzip.GzipFile(fileobj=raw) if magic[:2] == b"\x1f\x8b" else zstd.ZstdFile(raw)) as decoded:
                hasher = hashlib.sha256()
                while block := decoded.read(1024 * 1024):
                    hasher.update(block)
        assert hasher.hexdigest() == digest
        result.append({"capacity": size, "codec": name, "compressed_bytes": path.stat().st_size,
                       "save_median_ms": float(np.median(saves)), "save_p95_ms": float(np.percentile(saves, 95)),
                       "load_median_ms": float(np.median(loads)), "load_p95_ms": float(np.percentile(loads, 95)),
                       "save_python_peak_bytes": save_peak, "load_python_peak_bytes": load_peak,
                       "serialized_sha256": hasher.hexdigest()})
        print(json.dumps(result[-1]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 50000, 250000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-comparison", action="store_true",
                        help="benchmark real SB3 storage APIs and memory")
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original = None
    if args.source:
        with args.source.open("rb") as stream:
            magic = stream.read(4)
            stream.seek(0)
            if magic[:2] == b"\x1f\x8b":
                with gzip.GzipFile(fileobj=stream) as decoded:
                    original = pickle.load(decoded)
            elif magic == b"\x28\xb5\x2f\xfd":
                with zstd.ZstdFile(stream) as decoded:
                    original = pickle.load(decoded)
            else:
                original = pickle.load(stream)
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "source": str(args.source), "repeats": args.repeats, "cases": []}
    with tempfile.TemporaryDirectory(prefix="polybot-codecs-") as directory:
        path = Path(directory) / "replay"
        for size in args.sizes:
            if original is not None:
                if size > original.buffer_size:
                    raise ValueError("size exceeds recorded capacity; no fabricated schema padding")
                buffer = copy.copy(original)
                for key, value in vars(original).items():
                    if isinstance(value, np.ndarray) and value.shape[0] == original.buffer_size:
                        setattr(buffer, key, np.ascontiguousarray(value[:size]))
                buffer.buffer_size, buffer.pos, buffer.full = size, 0, True
            else:
                rng = np.random.default_rng(17)
                buffer = {"observations": rng.uniform(-1, 1, (size, 121)).astype(np.float32),
                          "next_observations": rng.uniform(-1, 1, (size, 121)).astype(np.float32),
                          "actions": rng.uniform(-1, 1, (size, 2)).astype(np.float32),
                          "rewards": rng.normal(size=(size, 1)).astype(np.float32)}
            start = time.perf_counter()
            payload = pickle.dumps(buffer, protocol=pickle.HIGHEST_PROTOCOL)
            serialization_ms = (time.perf_counter() - start) * 1000
            digest = hashlib.sha256(payload).hexdigest()
            if args.production_comparison:
                del payload
                report["cases"].extend(production_comparison(buffer, path, size, args.repeats, digest))
                continue
            for name in ("gzip1", "gzip3", "gzip6", "zstd1", "zstd3", "zstd6", "zstd3-2w"):
                saves, loads = [], []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    with path.open("wb") as raw:
                        with codec(raw, name, "wb") as stream:
                            # Stream pickle directly, rather than writing the
                            # pre-materialized payload used only for the oracle.
                            pickle.dump(buffer, stream, protocol=pickle.HIGHEST_PROTOCOL)
                        raw.flush()
                        os.fsync(raw.fileno())
                    saves.append((time.perf_counter() - start) * 1000)
                    start = time.perf_counter()
                    with path.open("rb") as raw, codec(raw, name, "rb") as stream:
                        decoded = stream.read()
                    loads.append((time.perf_counter() - start) * 1000)
                    assert hashlib.sha256(decoded).hexdigest() == digest
                    del decoded
                case = {"capacity": size, "codec": name, "raw_bytes": len(payload),
                        "serialized_sha256": digest, "serialization_ms": serialization_ms,
                        "compressed_bytes": path.stat().st_size,
                        "save_median_ms": float(np.median(saves)), "save_p95_ms": float(np.percentile(saves, 95)),
                        "decode_median_ms": float(np.median(loads)), "decode_p95_ms": float(np.percentile(loads, 95))}
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
            del buffer, payload
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
